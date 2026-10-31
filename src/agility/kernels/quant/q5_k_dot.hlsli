// Q5_K decode dots from quant/q5_k_dot.wgsl, including the fifth bits.
static const uint PARTS = 8u;
static const uint XT = 256u;
groupshared float4 xs[512];
float block_dot(uint row_byte, uint blk, uint xb4, uint p) {
    uint b = row_byte + blk * BLOCK_BYTES;
    uint c = p / 2u;
    uint4 hv = wvec(b >> 4u);
    uint4 qh = wvec((b + 16u + 16u * (p % 2u)) >> 4u);
    uint4 qv = wvec((b + 48u + 16u * p) >> 4u);
    float2 dm = unpack_half2(hv.x);
    uint2 s1 = scale_min_words(hv.yzw, 2u * c);
    uint2 s2 = scale_min_words(hv.yzw, 2u * c + 1u);
    precise float d1 = dm.x * float(s1.x), m1 = dm.y * float(s1.y);
    precise float d2 = dm.x * float(s2.x), m2 = dm.y * float(s2.y);
    uint xbase = xb4 + 16u * c + 4u * (p % 2u);
    precise float acc = 0.0;
    for (uint k = 0u; k < 4u; k++) {
        uint e = (k + p) % 4u;
        uint q1 = (qv[e] & 0x0f0f0f0fu) | (((qh[e] >> (2u * c)) & 0x01010101u) << 4u);
        uint q2 = ((qv[e] >> 4u) & 0x0f0f0f0fu) | (((qh[e] >> (2u * c + 1u)) & 0x01010101u) << 4u);
        precise float4 w1 = d1 * float4(unpack_u8(q1)) - m1;
        precise float4 w2 = d2 * float4(unpack_u8(q2)) - m2;
        acc += dot(w1, xs[xbase + e]);
        acc += dot(w2, xs[xbase + 8u + e]);
    }
    return acc;
}
float2 block_dot2(uint row_byte, uint blk, uint xb4, uint p) {
    uint b = row_byte + blk * BLOCK_BYTES;
    uint c = p / 2u;
    uint4 hv = wvec(b >> 4u);
    uint4 qh = wvec((b + 16u + 16u * (p % 2u)) >> 4u);
    uint4 qv = wvec((b + 48u + 16u * p) >> 4u);
    float2 dm = unpack_half2(hv.x);
    uint2 s1 = scale_min_words(hv.yzw, 2u * c);
    uint2 s2 = scale_min_words(hv.yzw, 2u * c + 1u);
    precise float d1 = dm.x * float(s1.x), m1 = dm.y * float(s1.y);
    precise float d2 = dm.x * float(s2.x), m2 = dm.y * float(s2.y);
    uint xbase = xb4 + 16u * c + 4u * (p % 2u);
    precise float2 acc = 0.0;
    for (uint k = 0u; k < 4u; k++) {
        uint e = (k + p) % 4u;
        uint q1 = (qv[e] & 0x0f0f0f0fu) | (((qh[e] >> (2u * c)) & 0x01010101u) << 4u);
        uint q2 = ((qv[e] >> 4u) & 0x0f0f0f0fu) | (((qh[e] >> (2u * c + 1u)) & 0x01010101u) << 4u);
        precise float4 w1 = d1 * float4(unpack_u8(q1)) - m1;
        precise float4 w2 = d2 * float4(unpack_u8(q2)) - m2;
        acc.x += dot(w1, xs[xbase + e]);
        acc.x += dot(w2, xs[xbase + 8u + e]);
        acc.y += dot(w1, xs[xbase + XT + e]);
        acc.y += dot(w2, xs[xbase + XT + 8u + e]);
    }
    return acc;
}
