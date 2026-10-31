// Quantized prefill multiply from matmul_quant.wgsl, with integer unit
// dots, per-half scaling, activation minima correction and split-k output.
static const uint TM = 32u, TN = 128u, WG = 128u;
groupshared uint4 xq[8u * TM / 4u];
groupshared float xd[TM];
groupshared uint4 wq[8u * TN / 4u];
groupshared float4 s0[TN / 4u], s1[TN / 4u], mn[TN / 4u];
struct Unit { uint4 lo; uint4 hi; float3 s; };
uint2 fetch_x(uint t0, uint k0, uint li) {
    uint2 r = 0u;
    for (uint i = 0u; i < 2u; i++) {
        uint e = li + i * WG, t = t0 + e / 8u;
        if (t < param(0)) r[i] = ldu(1, t * (param(2) / 4u) + k0 / 4u + e % 8u);
    }
    return r;
}
float fetch_dx(uint t0, uint k0, uint li) {
    uint t = t0 + li;
    if (li >= TM || t >= param(0)) return 0.0;
    return ldf(4, t * (param(2) / ACT_BLOCK) + k0 / ACT_BLOCK);
}
Unit fetch_w(uint j, uint row_bytes, uint k0) {
    Unit r = (Unit)0;
    if (j < param(1)) {
        Head h = head(j * row_bytes, k0 / 32u);
        r.lo = quad_i8(h, 0u); r.hi = quad_i8(h, 4u); r.s = scales(h);
    }
    return r;
}
[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    uint kb = wg.z * param(6), ke = min(param(2), kb + param(6));
    uint n0 = wg.x * TN, t0 = wg.y * TM;
    uint tg = li % (TM / 4u), ng = li / (TM / 4u);
    uint row_bytes = (param(2) / BLOCK_ELEMS) * BLOCK_BYTES;
    uint wj = n0 + li;
    precise float4 acc[8] = (float4[8])0;
    uint2 nx = fetch_x(t0, kb, li);
    float nd = fetch_dx(t0, kb, li);
    Unit nw = fetch_w(wj, row_bytes, kb);
    for (uint k0 = kb; k0 < ke; k0 += 32u) {
        for (uint i = 0u; i < 2u; i++) {
            uint e = li + i * WG, m = e / 8u, q = e % 8u;
            xq[(q * TM + m) / 4u][m % 4u] = nx[i];
        }
        if (li < TM) xd[li] = nd;
        for (uint e = 0u; e < 4u; e++) {
            wq[(e * TN + li) / 4u][li % 4u] = nw.lo[e];
            wq[((4u + e) * TN + li) / 4u][li % 4u] = nw.hi[e];
        }
        s0[li / 4u][li % 4u] = nw.s.x;
        s1[li / 4u][li % 4u] = nw.s.y;
        mn[li / 4u][li % 4u] = nw.s.z;
        GroupMemoryBarrierWithGroupSync();
        if (k0 + 32u < ke) {
            nx = fetch_x(t0, k0 + 32u, li);
            nd = fetch_dx(t0, k0 + 32u, li);
            nw = fetch_w(wj, row_bytes, k0 + 32u);
        }
        float4 d = float4(xd[4u * tg], xd[4u * tg + 1u], xd[4u * tg + 2u], xd[4u * tg + 3u]);
        for (uint h = 0u; h < 2u; h++) {
            int4 isum[8] = (int4[8])0;
            for (uint e4 = 0u; e4 < 4u; e4++) {
                uint e = 4u * h + e4;
                uint4 xv = xq[e * TM / 4u + tg];
                uint4 w0 = wq[e * TN / 4u + 2u * ng];
                uint4 w1 = wq[e * TN / 4u + 2u * ng + 1u];
                for (uint a = 0u; a < 4u; a++) {
                    isum[2u * a] += int4(dot_i8(xv[a], w0.x), dot_i8(xv[a], w0.y), dot_i8(xv[a], w0.z), dot_i8(xv[a], w0.w));
                    isum[2u * a + 1u] += int4(dot_i8(xv[a], w1.x), dot_i8(xv[a], w1.y), dot_i8(xv[a], w1.z), dot_i8(xv[a], w1.w));
                }
            }
            float4 sa = s0[2u * ng], sb = s0[2u * ng + 1u];
            if (h == 1u) { sa = s1[2u * ng]; sb = s1[2u * ng + 1u]; }
            for (uint a = 0u; a < 4u; a++) {
                acc[2u * a] += d[a] * (sa * float4(isum[2u * a]));
                acc[2u * a + 1u] += d[a] * (sb * float4(isum[2u * a + 1u]));
            }
        }
        if (HAS_MIN) {
            float4 ma = mn[2u * ng], mb = mn[2u * ng + 1u];
            for (uint a = 0u; a < 4u; a++) {
                int sum = 0;
                for (uint e = 0u; e < 8u; e++) sum += dot_i8(xq[e * TM / 4u + tg][a], 0x01010101u);
                precise float dm = d[a] * float(sum);
                acc[2u * a] -= dm * ma;
                acc[2u * a + 1u] -= dm * mb;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint a = 0u; a < 4u; a++) {
        uint t = t0 + 4u * tg + a;
        if (t >= param(0)) break;
        for (uint c = 0u; c < 8u; c++) {
            uint j = n0 + 8u * ng + c;
            if (j < param(1)) {
                uint o = wg.z * param(7) + t * param(4) + param(3) + j;
                precise float v = acc[2u * a + c / 4u][c % 4u];
                if (param(5) != 0u) v = ldf(3, o) + v;
                stf(3, o, v);
            }
        }
    }
}
