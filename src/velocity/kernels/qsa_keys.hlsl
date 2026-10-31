// RMSNorm and NeoX RoPE the indexer's query heads in place; cache its raw key
// as nearest-even f16 pairs (kernels/qsa_keys.wgsl). pack_f16 is prefixed.
#define WG 256
groupshared precise float partial[WG];
groupshared precise float vals[WG];

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint i = lid3.x;
    const uint x = wg.x;
    const uint t = wg.y;
    const uint d = param(1);
    const uint pos = ldu(5, 1) + t;
    const uint offset = t * param(7) + x * d;
    if (x == param(0)) {
        if (i < d / 2u) {
            stu(4, pos * (d / 2u) + i, pack_f16(float2(ldf(1, offset + 2u * i), ldf(1, offset + 2u * i + 1u))));
        }
        return;
    }
    precise float v = 0.0f;
    if (i < d) v = ldf(1, offset + i);
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
        const uint pair = 2u * (pos * half_dim + i);
        const float2 cs = float2(ldwf(3, pair), ldwf(3, pair + 1u));
        const float x0 = vals[i];
        const float x1 = vals[i + half_dim];
        vals[i] = x0 * cs.x - x1 * cs.y;
        vals[i + half_dim] = x0 * cs.y + x1 * cs.x;
    }
    GroupMemoryBarrierWithGroupSync();
    if (i < d) stf(1, offset + i, vals[i]);
}
