// agility: cs_6_10
// Single-product grouped TQ2_T expert GEMM. FP16-rounded activations and
// scaled FP16-dequantized weights, FP32 accumulation, no residual products.
// Preserve the optimized 64-output layout: host grid (ceil(rows / 32),
// expert, projection), odd x groups return, four wave32 output matrices.
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

groupshared half xs[TOKENS * KT];
// Scaled W[row][k], loaded as column-major B[k][row].
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
                    const uint slot = ldu(6, wg.y * param(7) + tile + t);
                    const uint xrow = param(3) == 0u ? slot / param(2) : slot;
                    v = ldf(3, xrow * param(4) + block * KT + i % KT);
                }
                xs[i] = (half)v;
            }
            // Keep the optimized lane-exchange trellis decoder. The scale
            // is uniform within each row's wave and precedes FP16 rounding.
            for (uint i = li; i < ROWS * 32u; i += WG) {
                const uint r = i / 32u;
                const uint step = lane;
                const uint j = row0 + r;
                float4 values = 0.0f;
                float scale = 0.0f;
                if (j < rows) {
                    const uint bo = (wg.y * rows + j) * row_bytes + block * BLOCK_BYTES;
                    scale = whalf(wg.z, bo);
                    const uint mine = wbyte(wg.z, bo + 2u + step);
                    const uint prev = WaveReadLaneAt(mine, (lane + 31u) & 31u);
                    const uint state = (prev << 8u) | mine;
                    const uint h = (state & 0x7fffu) * 0x9e3779b1u;
                    const uint wa = book[h >> 21u];
                    const uint wb = book[(h >> 10u) & 2047u];
                    values = float4(f16tof32(wa), f16tof32(wa >> 16u), f16tof32(wb), f16tof32(wb >> 16u));
                }
                const uint o = r * KT + step * 4u;
                [unroll] for (uint q = 0u; q < 4u; q++) weights[o + q] = (half)(scale * values[q]);
            }
            GroupMemoryBarrierWithGroupSync();
            TileC part = TileC::Splat(0.0f);
            [unroll] for (uint ks = 0u; ks < KT; ks += 16u) {
                TileA a = TileA::Load(xs, ks, KT, MatrixLayout::RowMajor);
                TileB b = TileB::Load(weights, wave * 16u * KT + ks, KT, MatrixLayout::ColMajor);
                part.MultiplyAccumulate(a, b);
            }
            // Materialize the block sum before replacing shared tiles. Keeping
            // the native matrix accumulator live across trellis decode crashes
            // the tested NVIDIA driver during pipeline compilation.
            for (uint e = 0u; e < total.Length(); e++) {
                precise float value = total.Get(e) + part.Get(e);
                total.Set(e, value);
            }
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
