// Tiled half-weight matrix multiply with optional residual add
// (kernels/matmul_half.wgsl); half_lo/half_hi come from half/*.hlsli.
static const uint TT = 16u;
static const uint TN = 64u;
static const uint TK = 32u;
static const uint WG = 256u;
groupshared float xs[TT * TK];
groupshared float ws[TN * (TK + 1u)];

[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint rows = param(0), n = param(1), k = param(2);
    const uint col0 = param(3), out_stride = param(4);
    const uint j0 = wg.x * TN, t0 = wg.y * TT;
    const uint tt = li / 16u, tn = li % 16u;
    precise float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k0 = 0u; k0 < k; k0 += TK) {
        for (uint e = li; e < TT * TK; e += WG) {
            const uint r = e / TK, c = e % TK, t = t0 + r;
            xs[e] = t < rows ? ldf(1, t * k + k0 + c) : 0.0f;
        }
        for (uint e = li; e < TN * (TK / 2u); e += WG) {
            const uint r = e / (TK / 2u), c = e % (TK / 2u), j = j0 + r;
            uint word = 0u;
            if (j < n) word = ldw(2, (j * k + k0) / 2u + c);
            ws[r * (TK + 1u) + 2u * c] = half_lo(word);
            ws[r * (TK + 1u) + 2u * c + 1u] = half_hi(word);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint c = 0u; c < TK; c++) {
            const float xv = xs[tt * TK + c];
            for (uint q = 0u; q < 4u; q++) {
                acc[q] += xv * ws[(tn + 16u * q) * (TK + 1u) + c];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const uint t = t0 + tt;
    if (t >= rows) return;
    for (uint q = 0u; q < 4u; q++) {
        const uint j = j0 + tn + 16u * q;
        if (j < n) {
            const uint o = t * out_stride + col0 + j;
            if (param(5) != 0u) stf(3, o, ldf(3, o) + acc[q]);
            else stf(3, o, acc[q]);
        }
    }
}
