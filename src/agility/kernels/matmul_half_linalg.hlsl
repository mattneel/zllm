// agility: cs_6_10
// Half-weight projection with unchanged params, bindings and grid.
// SINGLE_PRODUCT selects one FP16 activation product; otherwise retain
// activation hi+residual products. Both modes round decoded weights to FP16.
#include <hlsl/dx/linalg.h>

using namespace dx::linalg;
using TileA = Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave>;
using TileB = Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave>;
using TileC = Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave>;

static const uint TT = 16u;
static const uint TN = 64u;
static const uint TK = 32u;
static const uint SK = TK + 8u;
static const uint WG = 128u;
groupshared half xs_hi[TT * SK];
#if !SINGLE_PRODUCT
groupshared half xs_lo[TT * SK];
#endif
groupshared half ws[TN * SK];

[WaveSize(32)]
[numthreads(128, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint rows = param(0), n = param(1), k = param(2);
    const uint col0 = param(3), out_stride = param(4);
    const uint j0 = wg.x * TN, t0 = wg.y * TT;
    const uint wave = li / 32u;
    TileC total_hi = TileC::Splat(0.0f);
#if !SINGLE_PRODUCT
    TileC total_lo = TileC::Splat(0.0f);
#endif
    for (uint k0 = 0u; k0 < k; k0 += TK) {
        for (uint e = li; e < TT * TK; e += WG) {
            const uint t = t0 + e / TK, c = k0 + e % TK;
            float v = 0.0f;
            if (t < rows && c < k) v = ldf(1, t * k + c);
            const half hi = (half)v;
            const uint at = (e / TK) * SK + e % TK;
            xs_hi[at] = hi;
#if !SINGLE_PRODUCT
            precise float residual = v - (float)hi;
            xs_lo[at] = (half)residual;
#endif
        }
        for (uint e = li; e < TN * (TK / 2u); e += WG) {
            const uint r = e / (TK / 2u), c = e % (TK / 2u), j = j0 + r;
            uint word = 0u;
            if (j < n && k0 + 2u * c < k) word = ldw(2, (j * k + k0) / 2u + c);
            ws[r * SK + 2u * c] = (half)half_lo(word);
            ws[r * SK + 2u * c + 1u] = (half)half_hi(word);
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint ks = 0u; ks < TK; ks += 16u) {
            TileA a_hi = TileA::Load(xs_hi, ks, SK, MatrixLayout::RowMajor);
            TileB b = TileB::Load(ws, wave * 16u * SK + ks, SK, MatrixLayout::ColMajor);
            total_hi.MultiplyAccumulate(a_hi, b);
#if !SINGLE_PRODUCT
            TileA a_lo = TileA::Load(xs_lo, ks, SK, MatrixLayout::RowMajor);
            total_lo.MultiplyAccumulate(a_lo, b);
#endif
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint e = 0u; e < total_hi.Length(); e++) {
        const uint2 rc = total_hi.GetCoordinate(e);
        const uint t = t0 + rc.x, j = j0 + wave * 16u + rc.y;
        if (t < rows && j < n) {
            const uint o = t * out_stride + col0 + j;
            precise float v = total_hi.Get(e);
#if !SINGLE_PRODUCT
            v = v + total_lo.Get(e);
#endif
            if (param(5) != 0u) v = ldf(3, o) + v;
            stf(3, o, v);
        }
    }
}
