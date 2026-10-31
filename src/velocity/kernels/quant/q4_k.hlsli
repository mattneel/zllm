// Q4_K units and scales from quant/q4_k.wgsl.
static const uint BLOCK_ELEMS = 256u;
static const uint BLOCK_BYTES = 144u;
uint row_stride(uint k) { return (k / BLOCK_ELEMS) * BLOCK_BYTES; }
static const uint ACT_BLOCK = 256u;
static const bool HAS_MIN = true;
uint2 scale_min_words(uint3 sw, uint j) {
    if (j < 4u) return uint2(byte12(sw, j) & 63u, byte12(sw, j + 4u) & 63u);
    return uint2((byte12(sw, j + 4u) & 0xfu) | ((byte12(sw, j - 4u) >> 6u) << 4u),
                 (byte12(sw, j + 4u) >> 4u) | ((byte12(sw, j) >> 6u) << 4u));
}
struct Head { uint qs; uint shift; float d1; float m1; };
Head head(uint row_byte, uint u) {
    uint b = row_byte + (u / 8u) * BLOCK_BYTES;
    uint s = u % 8u;
    uint4 hv = wvec(b >> 4u);
    float2 dm = unpack_half2(hv.x);
    uint2 sm = scale_min_words(hv.yzw, s);
    Head h; h.qs = b + 16u + 32u * (s / 2u); h.shift = 4u * (s % 2u);
    h.d1 = dm.x * float(sm.x); h.m1 = dm.y * float(sm.y); return h;
}
float3 scales(Head h) { return float3(h.d1, h.d1, h.m1); }
uint4 quad_i8(Head h, uint i0) {
    return (wvec((h.qs + 4u * i0) >> 4u) >> h.shift) & 0x0f0f0f0fu;
}
