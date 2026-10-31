// agility: cs_6_10
// Grouped TQ2_T expert GEMM: unchanged params, bindings, write set and grid
// (ceil(rows / 32), expert, projection). Each group covers 16 grouped slots
// by 32 output rows. All 128 threads stage/decode; two wave32 matrices
// compute the output while the other threads help hide staging latency.
// Codebook values are exactly FP16. Keep each row's scale outside its
// 128-K block product, so only the FP32 activation needs FP16 high/residual
// decomposition. Two FP32-accumulating products recover that input to first
// order without rounding scaled weights to FP16 or a third matrix product.
// Fixed block order and matrix shape apply equally to full/partial tiles.
#include <hlsl/dx/linalg.h>

using namespace dx::linalg;
using TileA = Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave>;
using TileB = Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave>;
using TileC = Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave>;

#define WG 128
#define KT 128
#define TOKENS 16
#define ROWS 32
#define BLOCK_BYTES 34

groupshared half x_hi[TOKENS * KT];
groupshared half x_lo[TOKENS * KT];
// Unscaled W[row][k], loaded as column-major B[k][row].
groupshared half weights[ROWS * KT];
groupshared uint book[2048];
groupshared uint grouped_slots[TOKENS];
groupshared float scales[ROWS];
groupshared float result[2 * 256];

uint wbyte(uint z, uint i) {
    const uint word = z == 0u ? ldw(1, i >> 2u) : ldw(2, i >> 2u);
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}
float whalf(uint z, uint i) {
    return f16tof32((z == 0u ? ldw(1, i >> 2u) : ldw(2, i >> 2u)) >> ((i & 2u) * 8u));
}

[WaveSize(32)]
[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint n = ldu(5, wg.y);
    if (n == 0u) return;
    const uint rows = param(0);
    const uint blocks = param(1) / KT;
    const uint row_bytes = blocks * BLOCK_BYTES;
    const uint row0 = wg.x * ROWS;
    const uint wave = li / 32u;
    for (uint i = li; i < 2048u; i += WG) book[i] = ldw(7, i);

    for (uint tile = 0u; tile < n; tile += TOKENS) {
        if (li < TOKENS) {
            grouped_slots[li] = tile + li < n ? ldu(6, wg.y * param(7) + tile + li) : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        TileC total = TileC::Splat(0.0f);
        for (uint block = 0u; block < blocks; block++) {
            for (uint i = li; i < TOKENS * KT; i += WG) {
                const uint t = i / KT;
                float v = 0.0f;
                if (tile + t < n) {
                    const uint slot = grouped_slots[t];
                    const uint xrow = param(3) == 0u ? slot / param(2) : slot;
                    v = ldf(3, xrow * param(4) + block * KT + i % KT);
                }
                const half hi = (half)v;
                x_hi[i] = hi;
                x_lo[i] = (half)(v - (float)hi);
            }
            if (li < ROWS) {
                const uint j = row0 + li;
                scales[li] = j < rows ? whalf(wg.z, (wg.y * rows + j) * row_bytes + block * BLOCK_BYTES) : 0.0f;
            }
            // A step decodes four consecutive codebook values. The previous
            // stream byte wraps across all 32 steps of the trellis block.
            for (uint i = li; i < ROWS * 32u; i += WG) {
                const uint r = i / 32u;
                const uint step = i & 31u;
                const uint j = row0 + r;
                float4 values = 0.0f;
                if (j < rows) {
                    const uint bo = (wg.y * rows + j) * row_bytes + block * BLOCK_BYTES;
                    const uint state = (wbyte(wg.z, bo + 2u + ((step + 31u) & 31u)) << 8u) | wbyte(wg.z, bo + 2u + step);
                    const uint h = (state & 0x7fffu) * 0x9e3779b1u;
                    const uint wa = book[h >> 21u];
                    const uint wb = book[(h >> 10u) & 2047u];
                    values = float4(f16tof32(wa), f16tof32(wa >> 16u), f16tof32(wb), f16tof32(wb >> 16u));
                }
                const uint o = r * KT + step * 4u;
                [unroll] for (uint q = 0u; q < 4u; q++) weights[o + q] = (half)values[q];
            }
            GroupMemoryBarrierWithGroupSync();
            if (wave < 2u) {
                TileC part_hi = TileC::Splat(0.0f);
                TileC part_lo = TileC::Splat(0.0f);
                [unroll] for (uint ks = 0u; ks < KT; ks += 16u) {
                    TileA ah = TileA::Load(x_hi, ks, KT, MatrixLayout::RowMajor);
                    TileA al = TileA::Load(x_lo, ks, KT, MatrixLayout::RowMajor);
                    TileB b = TileB::Load(weights, wave * 16u * KT + ks, KT, MatrixLayout::ColMajor);
                    part_hi.MultiplyAccumulate(ah, b);
                    part_lo.MultiplyAccumulate(al, b);
                }
                // Matrix coordinates are (token, output row), independent of
                // the hardware's distribution of elements across wave lanes.
                for (uint e = 0u; e < total.Length(); e++) {
                    const uint2 rc = total.GetCoordinate(e);
                    precise float contribution = scales[wave * 16u + rc.y] * (part_hi.Get(e) + part_lo.Get(e));
                    precise float value = total.Get(e) + contribution;
                    total.Set(e, value);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (wave < 2u) total.Store(result, wave * 256u, 16, MatrixLayout::RowMajor);
        GroupMemoryBarrierWithGroupSync();
        for (uint e = li; e < TOKENS * ROWS; e += WG) {
            const uint t = e / ROWS;
            const uint r = e % ROWS;
            const uint j = row0 + r;
            if (tile + t < n && j < rows) {
                const uint o = (r / 16u) * 256u + t * 16u + r % 16u;
                stf(4, grouped_slots[t] * param(5) + wg.z * param(6) + j, result[o]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
