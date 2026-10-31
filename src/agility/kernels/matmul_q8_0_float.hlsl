// Tiled Q8_0-weight multiply with unrounded f32 activations.
// See kernels/matmul_q8_0_float.wgsl for the repacked layout and split-k ABI.
// Bindings: x at 1, weights at 2, output at 3; Params is P.MatmulQuant.
static const uint TT = 16u;
static const uint TN = 64u;
static const uint TK = 32u;
static const uint WG = 256u;
groupshared float xs[TT * TK];
groupshared precise float ws[TN * (TK + 1u)];

[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint rows = param(0), n = param(1), k = param(2);
    const uint col0 = param(3), out_stride = param(4);
    const uint kb = wg.z * param(6), ke = min(k, kb + param(6));
    const uint j0 = wg.x * TN, t0 = wg.y * TT;
    const uint tt = li / 16u, tn = li % 16u;
    const uint row_words = ((k + k / 16u + 15u) & ~15u) / 4u;
    precise float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k0 = kb; k0 < ke; k0 += TK) {
        for (uint e = li; e < TT * TK; e += WG) {
            const uint r = e / TK, c = e % TK, t = t0 + r;
            xs[e] = t < rows ? ldf(1, t * k + k0 + c) : 0.0f;
        }
        for (uint e = li; e < TN * (TK / 4u); e += WG) {
            const uint r = e / (TK / 4u), c = e % (TK / 4u), j = j0 + r;
            uint word = 0u;
            float scale = 0.0f;
            if (j < n) {
                const uint base = j * row_words;
                word = ldw(2, base + k0 / 4u + c);
                const uint block = k0 / TK;
                const uint scales = ldw(2, base + k / 4u + block / 2u);
                scale = f16tof32((scales >> ((block & 1u) * 16u)) & 0xffffu);
            }
            const uint o = r * (TK + 1u) + 4u * c;
            precise float4 decoded = scale * float4(
                asint(word << 24u) >> 24u,
                asint(word << 16u) >> 24u,
                asint(word << 8u) >> 24u,
                asint(word) >> 24u);
            ws[o] = decoded.x;
            ws[o + 1u] = decoded.y;
            ws[o + 2u] = decoded.z;
            ws[o + 3u] = decoded.w;
        }
        GroupMemoryBarrierWithGroupSync();
        // A scalar recurrence per output, unchanged by token tile occupancy.
        for (uint c = 0u; c < TK; c++) {
            const float xv = xs[tt * TK + c];
            for (uint q = 0u; q < 4u; q++) {
                precise float product = xv * ws[(tn + 16u * q) * (TK + 1u) + c];
                acc[q] += product;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const uint t = t0 + tt;
    if (t >= rows) return;
    for (uint q = 0u; q < 4u; q++) {
        const uint j = j0 + tn + 16u * q;
        if (j < n) {
            const uint o = wg.z * param(7) + t * out_stride + col0 + j;
            precise float v = acc[q];
            if (param(5) != 0u) v = ldf(3, o) + v;
            stf(3, o, v);
        }
    }
}
