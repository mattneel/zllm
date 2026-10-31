// Mean each newly completed cache block in position order, RMSNorm with
// k_norm, and NeoX RoPE at its first position (kernels/qsa_pool.wgsl).
#define WG 256
groupshared precise float partial[WG];
groupshared precise float vals[WG];

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint i = lid3.x;
    const uint r = param(2);
    const uint d = param(1);
    const uint b = ldu(5, 1) / r + wg.x;
    if ((b + 1u) * r > ldu(5, 1) + ldu(5, 0)) return;
    precise float v = 0.0f;
    if (i < d) {
        for (uint j = 0u; j < r; j++) {
            const uint bits = ldu(1, ((b * r + j) * d + i) / 2u);
            const float2 pair = float2(f16tof32(bits & 0xffffu), f16tof32(bits >> 16u));
            v += (i & 1u) == 1u ? pair.y : pair.x;
        }
        v = v / float(r);
    }
    partial[i] = v * v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s /= 2u) {
        if (i < s) partial[i] += partial[i + s];
        GroupMemoryBarrierWithGroupSync();
    }
    precise float scale = 1.0f / sqrt(partial[0] / float(d) + paramf(6));
    if (i < d) vals[i] = v * scale * ldwf(2, i);
    GroupMemoryBarrierWithGroupSync();
    const uint half_dim = param(5) / 2u;
    if (i < half_dim) {
        const uint pair = 2u * (b * r * half_dim + i);
        const float2 cs = float2(ldwf(3, pair), ldwf(3, pair + 1u));
        const float x0 = vals[i];
        const float x1 = vals[i + half_dim];
        vals[i] = x0 * cs.x - x1 * cs.y;
        vals[i + half_dim] = x0 * cs.y + x1 * cs.x;
    }
    GroupMemoryBarrierWithGroupSync();
    if (i < d) stf(4, b * d + i, vals[i]);
}
