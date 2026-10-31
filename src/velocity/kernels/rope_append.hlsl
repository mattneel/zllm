// Normalize and rotate Q/K heads, then append K/V to the f32 or nearest-even
// f16 cache; supports gated queries and NeoX or normal rotary pairing
// (kernels/rope_append.wgsl), composed after f16_round.hlsli.
static const uint WG = 256u;
groupshared float partial[WG];
groupshared float vals[WG];

[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint i = lid3.x, x = wg.x, t = wg.y;
    const uint heads = param(0), kv_heads = param(1), hd = param(2);
    const uint n_rot = param(3), row = param(4);
    const bool is_q = x < heads;
    const uint g = x - (is_q ? 0u : heads);
    const uint q_head = param(9) != 0u ? 2u * hd : hd;
    const uint head_base = t * row + (is_q ? x * q_head : param(5) + g * hd);
    float v = 0.0f;
    if (i < hd) v = ldf(1, head_base + i);
    partial[i] = v * v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s /= 2u) {
        if (i < s) partial[i] += partial[i + s];
        GroupMemoryBarrierWithGroupSync();
    }
    precise float scale = 1.0f / sqrt(partial[0] / float(hd) + paramf(8));
    if (i < hd) {
        if (param(10) != 0u) {
            precise float normalized = v * scale * (is_q ? ldwf(2, i) : ldwf(3, i));
            vals[i] = normalized;
        } else {
            vals[i] = v;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    const uint rot_half = n_rot / 2u, pos = ldu(7, 1) + t;
    if (i < rot_half) {
        const uint cs_base = 2u * (pos * rot_half + i);
        const float2 cs = float2(ldwf(4, cs_base), ldwf(4, cs_base + 1u));
        const bool normal = param(11) != 0u;
        const uint a = normal ? 2u * i : i;
        const uint b = normal ? 2u * i + 1u : i + rot_half;
        const float x0 = vals[a], x1 = vals[b];
        precise float rotated0 = x0 * cs.x - x1 * cs.y;
        precise float rotated1 = x0 * cs.y + x1 * cs.x;
        vals[a] = rotated0;
        vals[b] = rotated1;
    }
    GroupMemoryBarrierWithGroupSync();
    if (is_q) {
        if (i < hd) stf(1, head_base + i, vals[i]);
        return;
    }
    const uint width = kv_heads * hd, e = pos * width + g * hd;
    const uint vb = t * row + param(6) + g * hd;
    if (i < hd / 2u) {
        const float k0 = vals[2u * i], k1 = vals[2u * i + 1u];
        const float v0 = ldf(1, vb + 2u * i), v1 = ldf(1, vb + 2u * i + 1u);
        if (param(7) != 0u) {
            stu(5, e / 2u + i, pack_f16(float2(k0, k1)));
            stu(6, e / 2u + i, pack_f16(float2(v0, v1)));
        } else {
            stu(5, e + 2u * i, asuint(k0));
            stu(5, e + 2u * i + 1u, asuint(k1));
            stu(6, e + 2u * i, asuint(v0));
            stu(6, e + 2u * i + 1u, asuint(v1));
        }
    }
}
