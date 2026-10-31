// Rounds activation rows to Q8_K or Q8_0 from quantize_q8.wgsl, with
// optional RMSNorm or SiLU(gate)*up, exact division and pinned x86 CPU
// tie-even rounding (Q8_0's scalar reference instead rounds ties away).
groupshared float2 squares[64];
groupshared float row_scale;
groupshared float4 vals[64];
groupshared float2 best[64];
float4 load_x(uint v) { return asfloat(rw(1).Load4(base(1) + 16u * v)); }
float4 load_nw(uint v) { return asfloat(ro(4).Load4(base(4) + 16u * v)); }
float f16_rne(float x) {
    uint u = asuint(x);
    if (((u >> 23u) & 0xffu) < 113u) return round(x * 16777216.0) * (1.0 / 16777216.0);
    return asfloat((u + 0xfffu + ((u >> 13u) & 1u)) & 0xffffe000u);
}
float round_even(float y) {
    return float(int(asuint(y + 12582912.0)) - 0x4b400000);
}
[numthreads(64, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    uint k = param(0), block = param(1), mode = param(2);
    uint row = wg.y;
    uint xbase = row * (k / 4u) + wg.x * 64u;
    precise float4 v;
    if (mode == 2u) {
        uint src = row * (k / 2u) + wg.x * 64u + li;
        float4 g = load_x(src);
        v = g / (1.0 + exp(-g)) * load_x(src + k / 4u);
    } else v = load_x(xbase + li);
    if (mode == 1u) {
        float2 sum = 0.0;
        for (uint i = li; i < k / 4u; i += 64u) {
            float4 r = load_x(row * (k / 4u) + i);
            precise float4 square = r * r;
            sum = norm_add(sum, square.x);
            sum = norm_add(sum, square.y);
            sum = norm_add(sum, square.z);
            sum = norm_add(sum, square.w);
        }
        squares[li] = sum;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 32u; s > 0u; s /= 2u) {
            if (li < s) squares[li] = norm_merge(squares[li], squares[li + s]);
            GroupMemoryBarrierWithGroupSync();
        }
        if (li == 0u) row_scale = norm_scale(squares[0], k, paramf(3));
        GroupMemoryBarrierWithGroupSync();
        v = v * row_scale * load_nw(wg.x * 64u + li);
    }
    vals[li] = v;
    float4 a = abs(v);
    float2 m = float2(a.x, 0.0);
    if (a.y > m.x) m = float2(a.y, 1.0);
    if (a.z > m.x) m = float2(a.z, 2.0);
    if (a.w > m.x) m = float2(a.w, 3.0);
    best[li] = float2(m.x, float(4u * li) + m.y);
    GroupMemoryBarrierWithGroupSync();
    uint span = block / 4u;
    for (uint s = span / 2u; s > 0u; s /= 2u) {
        if (li % span < s) {
            float2 o = best[li + s], b = best[li];
            if (o.x > b.x || (o.x == b.x && o.y < b.y)) best[li] = o;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float2 lead = best[li - li % span];
    precise float4 q = 0.0;
    float d = 0.0;
    if (block == 256u) {
        if (lead.x != 0.0) {
            uint at = uint(lead.y);
            float vmax = vals[at / 4u][at % 4u];
            float iscale = div_rn(-127.0, vmax);
            d = div_rn(1.0, iscale);
            q = min(127.0, float4(round_even(iscale * v.x), round_even(iscale * v.y), round_even(iscale * v.z), round_even(iscale * v.w)));
        }
    } else {
        if (lead.x != 0.0) {
            float d32 = div_rn(lead.x, 127.0);
            float id = div_rn(127.0, lead.x);
            d = f16_rne(d32);
            q = float4(round_even(v.x * id), round_even(v.y * id), round_even(v.z * id), round_even(v.w * id));
        }
    }
    int4 qi = int4(q);
    stu(2, xbase + li, (uint(qi.x) & 0xffu) | ((uint(qi.y) & 0xffu) << 8u) | ((uint(qi.z) & 0xffu) << 16u) | ((uint(qi.w) & 0xffu) << 24u));
    if (li % span == 0u) stf(3, (row * k + wg.x * 256u + 4u * li) / block, d);
}
