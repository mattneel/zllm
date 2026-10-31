// Repacked Q3_K decode dots from quant/q3_k_dot.wgsl.
static const uint PARTS = 8u;
static const uint XT = 256u;
groupshared float4 xs[512];
float block_dot(uint row_byte, uint blk, uint xb4, uint p) {
    Head h = head(row_byte + blk * BLOCK_BYTES, p);
    precise float acc = 0.0;
    for (uint k = 0u; k < 8u; k++) {
        uint e = (k + p) % 8u;
        uint c = ((h.lo[e / 4u] >> (2u * (e % 4u))) & 0x03030303u) | (((h.hi >> e) & 0x01010101u) << 2u);
        float dl = e >= 4u ? h.dl1 : h.dl0;
        precise float4 wv = dl * (float4(unpack_u8(c)) - 4.0);
        acc += dot(wv, xs[xb4 + 8u * p + e]);
    }
    return acc;
}
float2 block_dot2(uint row_byte, uint blk, uint xb4, uint p) {
    Head h = head(row_byte + blk * BLOCK_BYTES, p);
    precise float2 acc = 0.0;
    for (uint k = 0u; k < 8u; k++) {
        uint e = (k + p) % 8u;
        uint c = ((h.lo[e / 4u] >> (2u * (e % 4u))) & 0x03030303u) | (((h.hi >> e) & 0x01010101u) << 2u);
        float dl = e >= 4u ? h.dl1 : h.dl0;
        precise float4 wv = dl * (float4(unpack_u8(c)) - 4.0);
        uint i = xb4 + 8u * p + e;
        acc.x += dot(wv, xs[i]);
        acc.y += dot(wv, xs[i + XT]);
    }
    return acc;
}
