// Normalize and rotate Q/K heads, then append K/V to the cache in its format
// (param 7: f32, nearest-even f16, or row-scaled FP8 with scales from word
// param 12); supports gated queries and NeoX or normal rotary pairing
// (kernels/rope_append.wgsl), composed after f16_round.hlsli and fp8.hlsli.
static const uint WG = 256u;
groupshared float partial[WG];
groupshared float vals[WG];

// The largest `a` over the group's threads. Every thread calls it.
float row_max(float a, uint i) {
    partial[i] = a;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s /= 2u) {
        if (i < s) partial[i] = max(partial[i], partial[i + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    const float m = partial[0];
    GroupMemoryBarrierWithGroupSync();
    return m;
}

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
    if (param(7) == 2u) {
        const int k_exp = fp8_scale_exp(row_max(i < hd ? abs(vals[i]) : 0.0f, i));
        const int v_exp = fp8_scale_exp(row_max(i < hd ? abs(ldf(1, vb + min(i, hd - 1u))) : 0.0f, i));
        if (i < hd / 4u) {
            uint kw = 0u, vw = 0u;
            for (uint j = 0u; j < 4u; j++) {
                kw |= fp8_bits(vals[4u * i + j] * pow2(-k_exp)) << (8u * j);
                vw |= fp8_bits(ldf(1, vb + 4u * i + j) * pow2(-v_exp)) << (8u * j);
            }
            stu(5, e / 4u + i, kw);
            stu(6, e / 4u + i, vw);
        }
        if (i == 0u) {
            stu(5, param(12) + pos * kv_heads + g, asuint(pow2(k_exp)));
            stu(6, param(12) + pos * kv_heads + g, asuint(pow2(v_exp)));
        }
        return;
    }
    if (i < hd / 2u) {
        const float k0 = vals[2u * i], k1 = vals[2u * i + 1u];
        const float v0 = ldf(1, vb + 2u * i), v1 = ldf(1, vb + 2u * i + 1u);
        if (param(7) == 1u) {
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
