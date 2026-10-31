// Q6_K decode dots from quant/q6_k_dot.wgsl, eight parts per block.
static const uint PARTS = 8u;
static const uint XT = 256u;
groupshared float4 xs[512];
float block_dot(uint row_byte, uint blk, uint xb4, uint p) {
    uint b = row_byte + blk * BLOCK_BYTES;
    uint n = p / 4u, cp = (p / 2u) % 2u, hh = p % 2u;
    uint4 qlv = wvec((b + 64u * n + 32u * cp + 16u * hh) >> 4u);
    uint4 qhv = wvec((b + 128u + 32u * n + 16u * hh) >> 4u);
    uint4 sv = wvec((b + 192u) >> 4u);
    float d = f16tof32(wvec((b + 208u) >> 4u).x & 0xffffu);
    precise float acc = 0.0;
    for (uint e2 = 0u; e2 < 2u; e2++) {
        uint c = cp + 2u * e2;
        uint si = 8u * n + hh + 2u * c;
        precise float ds = d * float(sbyte16(sv, si));
        for (uint k = 0u; k < 4u; k++) {
            uint e = (k + p) % 4u;
            uint q = ((qlv[e] >> (4u * e2)) & 0x0f0f0f0fu) | (((qhv[e] >> (2u * c)) & 0x03030303u) << 4u);
            precise float4 wv = ds * (float4(unpack_u8(q)) - 32.0);
            acc += dot(wv, xs[xb4 + 32u * n + 8u * c + 4u * hh + e]);
        }
    }
    return acc;
}
float2 block_dot2(uint row_byte, uint blk, uint xb4, uint p) {
    uint b = row_byte + blk * BLOCK_BYTES;
    uint n = p / 4u, cp = (p / 2u) % 2u, hh = p % 2u;
    uint4 qlv = wvec((b + 64u * n + 32u * cp + 16u * hh) >> 4u);
    uint4 qhv = wvec((b + 128u + 32u * n + 16u * hh) >> 4u);
    uint4 sv = wvec((b + 192u) >> 4u);
    float d = f16tof32(wvec((b + 208u) >> 4u).x & 0xffffu);
    precise float2 acc = 0.0;
    for (uint e2 = 0u; e2 < 2u; e2++) {
        uint c = cp + 2u * e2;
        uint si = 8u * n + hh + 2u * c;
        precise float ds = d * float(sbyte16(sv, si));
        for (uint k = 0u; k < 4u; k++) {
            uint e = (k + p) % 4u;
            uint q = ((qlv[e] >> (4u * e2)) & 0x0f0f0f0fu) | (((qhv[e] >> (2u * c)) & 0x03030303u) << 4u);
            precise float4 wv = ds * (float4(unpack_u8(q)) - 32.0);
            uint i = xb4 + 32u * n + 8u * c + 4u * hh + e;
            acc.x += dot(wv, xs[i]);
            acc.y += dot(wv, xs[i + XT]);
        }
    }
    return acc;
}
