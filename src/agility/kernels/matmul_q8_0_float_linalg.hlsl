// agility: cs_6_10
// Compensated Q8_0 projection: FP16 activation hi+lo, exact FP16 int8 weights,
// FP32 accumulation. Appended to common.hlsli. Bindings/Params/grid remain
// P.MatmulQuant: x=1, read-only repacked weights=2, output=3 (only write),
// ceil(n/64) x ceil(rows/16) x split_count; k and k_slice are multiples of 32.
#include <hlsl/dx/linalg.h>

using namespace dx::linalg;
using TileA = Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave>;
using TileB = Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave>;
using TileC = Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave>;

static const uint TT = 16u;
static const uint TN = 64u;
static const uint TK = 64u;
// Keep shared rows 16-byte aligned without mapping every weight row to
// the same memory banks during cooperative staging.
static const uint SK = TK + 8u;
static const uint WG = 128u;

// A Q8_0 block has one FP16 scale for 32 signed bytes. The bytes are exact
// FP16 integers, so factor the scale outside each 32-position dot product.
// This avoids dequantized-weight rounding and its third correction MMA.
// Activation hi+lo retains roughly FP16 precision squared, subject to FP16
// overflow/subnormal limits. Total groupshared storage is 14,336 bytes.
groupshared half xs_hi[TT * SK];
groupshared half xs_lo[TT * SK];
groupshared half ws[TN * SK];
groupshared float scales[TN * (TK / 32u)];

[WaveSize(32)]
[numthreads(128, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint rows = param(0), n = param(1), k = param(2);
    const uint col0 = param(3), out_stride = param(4);
    const uint kb = wg.z * param(6), ke = min(k, kb + param(6));
    const uint j0 = wg.x * TN, t0 = wg.y * TT;
    const uint wave = li / 32u;
    const uint row_words = ((k + k / 16u + 15u) & ~15u) / 4u;
    TileC total = TileC::Splat(0.0f);

    for (uint k0 = kb; k0 < ke; k0 += TK) {
        for (uint e = li; e < TT * TK; e += WG) {
            const uint t = t0 + e / TK, c = k0 + e % TK;
            float v = 0.0f;
            if (t < rows && c < ke) v = ldf(1, t * k + c);
            const half hi = (half)v;
            precise float residual = v - (float)hi;
            const uint at = (e / TK) * SK + e % TK;
            xs_hi[at] = hi;
            xs_lo[at] = (half)residual;
        }

        // One thread per row/32-value half. Invalid columns and the final
        // 32-position K half are zero-filled, including unaligned split starts.
        const uint r = li / 2u, half_index = li % 2u;
        const uint j = j0 + r, kk = k0 + half_index * 32u;
        if (j < n && kk < ke) {
            const uint wbase = j * row_words;
            const uint block = kk / 32u;
            const uint packed_scales = ldw(2, wbase + k / 4u + block / 2u);
            scales[li] = f16tof32((packed_scales >> ((block & 1u) * 16u)) & 0xffffu);
            [unroll] for (uint q = 0u; q < 8u; q++) {
                const uint word = ldw(2, wbase + kk / 4u + q);
                const uint o = r * SK + half_index * 32u + q * 4u;
                ws[o] = (half)(asint(word << 24u) >> 24u);
                ws[o + 1u] = (half)(asint(word << 16u) >> 24u);
                ws[o + 2u] = (half)(asint(word << 8u) >> 24u);
                ws[o + 3u] = (half)(asint(word) >> 24u);
            }
        } else {
            scales[li] = 0.0f;
            [unroll] for (uint c = 0u; c < 32u; c++) {
                ws[r * SK + half_index * 32u + c] = (half)0.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Each wave owns 16 tokens x 16 outputs. Separate high/residual
        // sums avoid swallowing activation corrections before scaling.
        // Get/Set operate on register-local accumulator elements: no block
        // results need a groupshared round trip or an additional barrier.
        [unroll] for (uint block = 0u; block < TK / 32u; block++) {
            TileC part_hi = TileC::Splat(0.0f);
            TileC part_lo = TileC::Splat(0.0f);
            [unroll] for (uint c = 0u; c < 32u; c += 16u) {
                const uint ks = block * 32u + c;
                TileA a_hi = TileA::Load(xs_hi, ks, SK, MatrixLayout::RowMajor);
                TileA a_lo = TileA::Load(xs_lo, ks, SK, MatrixLayout::RowMajor);
                TileB b = TileB::Load(ws, wave * 16u * SK + ks, SK, MatrixLayout::ColMajor);
                part_hi.MultiplyAccumulate(a_hi, b);
                part_lo.MultiplyAccumulate(a_lo, b);
            }
            [loop] for (uint e = 0u; e < total.Length(); e++) {
                const uint2 rc = total.GetCoordinate(e);
                const float scale = scales[(wave * 16u + rc.y) * (TK / 32u) + block];
                precise float dot = part_hi.Get(e) + part_lo.Get(e);
                precise float product = dot * scale;
                precise float v = total.Get(e) + product;
                total.Set(e, v);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Store register-local matrix elements without a shared output plane.
    for (uint e = 0u; e < total.Length(); e++) {
        const uint2 rc = total.GetCoordinate(e);
        const uint t = t0 + rc.x, j = j0 + wave * 16u + rc.y;
        if (t < rows && j < n) {
            const uint o = wg.z * param(7) + t * out_stride + col0 + j;
            precise float v = total.Get(e);
            if (param(5) != 0u) v = ldf(3, o) + v;
            stf(3, o, v);
        }
    }
}