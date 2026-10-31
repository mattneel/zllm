// Repacked Q3_K units and scales from quant/q3_k.wgsl.
static const uint BLOCK_ELEMS = 256u;
static const uint BLOCK_BYTES = 128u;
static const uint ACT_BLOCK = 256u;
static const bool HAS_MIN = false;
struct Head { uint2 lo; uint hi; float dl0; float dl1; };
Head head(uint row_byte, uint u) {
    uint4 v = wvec((row_byte + 16u * u) >> 4u);
    float d = f16tof32(v.w >> 16u);
    int sc0 = asint(v.w << 24u) >> 24u;
    int sc1 = asint(v.w << 16u) >> 24u;
    Head h; h.lo = v.xy; h.hi = v.z; h.dl0 = d * float(sc0); h.dl1 = d * float(sc1); return h;
}
float3 scales(Head h) { return float3(h.dl0, h.dl1, 0.0); }
uint4 quad_i8(Head h, uint i0) {
    uint l = h.lo[i0 / 4u];
    uint4 r;
    for (uint e = 0u; e < 4u; e++) {
        uint c = ((l >> (2u * e)) & 0x03030303u) | (((h.hi >> (i0 + e)) & 0x01010101u) << 2u);
        r[e] = (c + 0x7c7c7c7cu) ^ 0x80808080u;
    }
    return r;
}
