// agility: cs_6_10
// Grouped TQ2_T expert GEMM with the production params, bindings and write set.
// The host still dispatches (ceil(rows / 32), expert, projection). Even x
// groups cover 64 rows; odd x groups return. Four wave32 matrices compute
// 16 routed slots each, reusing the activation tile across all four waves.
// Keep the complete 128-K block and the production high/residual products,
// accumulation order and post-block scale application. No scaled FP16 weights.
// Shared storage is exactly 32 KiB: 8 KiB activations, 16 KiB weights and
// 8 KiB codebook. Wave-local scales and direct coordinate-based output stores
// replace shared scales, grouped-slot metadata and result staging.
#include <hlsl/dx/linalg.h>

using namespace dx::linalg;
using TileA = Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave>;
using TileB = Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave>;
using TileC = Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave>;

#define WG 128
#define KT 128
#define TOKENS 16
#define ROWS 64
#define BLOCK_BYTES 34

groupshared half x_hi[TOKENS * KT];
groupshared half x_lo[TOKENS * KT];
// Unscaled W[row][k], loaded as column-major B[k][row].
groupshared half weights[ROWS * KT];
groupshared uint book[2048];

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
    if ((wg.x & 1u) != 0u) return;
    const uint n = ldu(5, wg.y);
    if (n == 0u) return;
    const uint rows = param(0);
    const uint blocks = param(1) / KT;
    const uint row_bytes = blocks * BLOCK_BYTES;
    const uint row0 = wg.x * 32u;
    const uint wave = li / 32u;
    const uint lane = WaveGetLaneIndex();
    for (uint i = li; i < 2048u; i += WG) book[i] = ldw(7, i);
    GroupMemoryBarrierWithGroupSync();

    for (uint tile = 0u; tile < n; tile += TOKENS) {
        TileC total = TileC::Splat(0.0f);
        for (uint block = 0u; block < blocks; block++) {
            for (uint i = li; i < TOKENS * KT; i += WG) {
                const uint t = i / KT;
                float v = 0.0f;
                if (tile + t < n) {
                    // With KT == WG, t is uniform across the entire group in
                    // each iteration. Metadata reads can broadcast/cache;
                    // retaining a shared slot array would exceed 32 KiB.
                    const uint slot = ldu(6, wg.y * param(7) + tile + t);
                    const uint xrow = param(3) == 0u ? slot / param(2) : slot;
                    v = ldf(3, xrow * param(4) + block * KT + i % KT);
                }
                const half hi = (half)v;
                x_hi[i] = hi;
                x_lo[i] = (half)(v - (float)hi);
            }
            // Lanes 0..15 own one output-row scale each. All lanes remain
            // active, so GetCoordinate(e).y can select its exact row scale.
            float row_scale = 0.0f;
            const uint scale_row = row0 + wave * 16u + lane;
            if (lane < 16u && scale_row < rows) {
                row_scale = whalf(wg.z, (wg.y * rows + scale_row) * row_bytes + block * BLOCK_BYTES);
            }
            // Every wave visits all 32 consecutive steps of one row. Exchange
            // the preceding stream byte instead of loading it a second time;
            // lane 0 reads lane 31, preserving the trellis block's wraparound.
            for (uint i = li; i < ROWS * 32u; i += WG) {
                const uint r = i / 32u;
                const uint step = lane;
                const uint j = row0 + r;
                float4 values = 0.0f;
                if (j < rows) {
                    const uint bo = (wg.y * rows + j) * row_bytes + block * BLOCK_BYTES;
                    const uint mine = wbyte(wg.z, bo + 2u + step);
                    const uint prev = WaveReadLaneAt(mine, (lane + 31u) & 31u);
                    const uint state = (prev << 8u) | mine;
                    const uint h = (state & 0x7fffu) * 0x9e3779b1u;
                    const uint wa = book[h >> 21u];
                    const uint wb = book[(h >> 10u) & 2047u];
                    values = float4(f16tof32(wa), f16tof32(wa >> 16u), f16tof32(wb), f16tof32(wb >> 16u));
                }
                const uint o = r * KT + step * 4u;
                [unroll] for (uint q = 0u; q < 4u; q++) weights[o + q] = (half)values[q];
            }
            GroupMemoryBarrierWithGroupSync();
            TileC part_hi = TileC::Splat(0.0f);
            TileC part_lo = TileC::Splat(0.0f);
            [unroll] for (uint ks = 0u; ks < KT; ks += 16u) {
                TileA ah = TileA::Load(x_hi, ks, KT, MatrixLayout::RowMajor);
                TileA al = TileA::Load(x_lo, ks, KT, MatrixLayout::RowMajor);
                TileB b = TileB::Load(weights, wave * 16u * KT + ks, KT, MatrixLayout::ColMajor);
                part_hi.MultiplyAccumulate(ah, b);
                part_lo.MultiplyAccumulate(al, b);
            }
            for (uint e = 0u; e < total.Length(); e++) {
                const uint2 rc = total.GetCoordinate(e);
                const float scale = WaveReadLaneAt(row_scale, rc.y);
                precise float contribution = scale * (part_hi.Get(e) + part_lo.Get(e));
                precise float value = total.Get(e) + contribution;
                total.Set(e, value);
            }
            // No wave may replace the staging buffers until every wave has
            // finished its matrix loads. This also makes the next token tile
            // safe without an extra barrier around direct output stores.
            GroupMemoryBarrierWithGroupSync();
        }
        for (uint e = 0u; e < total.Length(); e++) {
            const uint2 rc = total.GetCoordinate(e);
            const uint j = row0 + wave * 16u + rc.y;
            if (tile + rc.x < n && j < rows) {
                const uint slot = ldu(6, wg.y * param(7) + tile + rc.x);
                stf(4, slot * param(5) + wg.z * param(6) + j, total.Get(e));
            }
        }
    }
}