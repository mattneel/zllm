// Q8_0 units and scales from quant/q8_0.wgsl (34-byte blocks).
static const uint BLOCK_ELEMS = 32u;
static const uint BLOCK_BYTES = 34u;
static const uint ACT_BLOCK = 32u;
static const bool HAS_MIN = false;
struct Head { uint qs; float d; };
Head head(uint row_byte, uint u) {
    uint b = row_byte + u * BLOCK_BYTES;
    Head h; h.qs = b + 2u; h.d = whalf(b); return h;
}
float3 scales(Head h) { return float3(h.d, h.d, 0.0); }
uint4 quad_i8(Head h, uint i0) {
    return uint4(w32(h.qs + 4u * i0), w32(h.qs + 4u * i0 + 4u), w32(h.qs + 4u * i0 + 8u), w32(h.qs + 4u * i0 + 12u));
}
