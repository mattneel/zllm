// Q8_0 decode dots from quant/q8_0_dot.wgsl, on split rows: two parts per
// block, 16 weights each from one aligned 16-byte load.
static const uint PARTS = 2u;
// Declared before the dots because composition puts this fragment first.
static const uint XT = 256u;
groupshared float4 xs[512];
float block_dot(uint row_byte, uint blk, uint xb4, uint p) {
    uint4 q = wvec((row_byte + 32u * blk) / 16u + p);
    float d = whalf(row_byte + row_k + 2u * blk);
    precise float s = 0.0;
    [unroll] for (uint e = 0u; e < 4u; e++) {
        precise float4 wv = float4(unpack_i8(q[e])) * d;
        s += dot(wv, xs[xb4 + 4u * p + e]);
    }
    return s;
}
float2 block_dot2(uint row_byte, uint blk, uint xb4, uint p) {
    uint4 q = wvec((row_byte + 32u * blk) / 16u + p);
    float d = whalf(row_byte + row_k + 2u * blk);
    precise float2 s = 0.0;
    [unroll] for (uint e = 0u; e < 4u; e++) {
        precise float4 wv = float4(unpack_i8(q[e])) * d;
        s += float2(dot(wv, xs[xb4 + 4u * p + e]), dot(wv, xs[xb4 + XT + 4u * p + e]));
    }
    return s;
}
