// Q8_0 decode dots from quant/q8_0_dot.wgsl: eight parts per block.
static const uint PARTS = 8u;
// Declared before the dots because composition puts this fragment first.
static const uint XT = 256u;
groupshared float4 xs[512];
float block_dot(uint row_byte, uint blk, uint xb4, uint p) {
    uint b = row_byte + blk * BLOCK_BYTES;
    int4 q = unpack_i8(w32(b + 2u + 4u * p));
    precise float4 wv = float4(q) * whalf(b);
    precise float acc = dot(wv, xs[xb4 + p]);
    return acc;
}
float2 block_dot2(uint row_byte, uint blk, uint xb4, uint p) {
    uint b = row_byte + blk * BLOCK_BYTES;
    int4 q = unpack_i8(w32(b + 2u + 4u * p));
    precise float4 wv = float4(q) * whalf(b);
    precise float2 acc = float2(dot(wv, xs[xb4 + p]), dot(wv, xs[xb4 + XT + p]));
    return acc;
}
