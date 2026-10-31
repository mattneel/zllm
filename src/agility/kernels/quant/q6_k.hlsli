// Padded Q6_K units and scales from quant/q6_k.wgsl.
static const uint BLOCK_ELEMS = 256u;
static const uint BLOCK_BYTES = 224u;
uint row_stride(uint k) { return (k / BLOCK_ELEMS) * BLOCK_BYTES; }
static const uint ACT_BLOCK = 256u;
static const bool HAS_MIN = false;
struct Head { uint ql; uint qh; uint low_shift; uint high_shift; float ds0; float ds1; };
int sbyte16(uint4 s, uint j) { return asint((s[j / 4u] >> (8u * (j % 4u))) << 24u) >> 24u; }
Head head(uint row_byte, uint u) {
    uint b = row_byte + (u / 8u) * BLOCK_BYTES;
    uint s = u % 8u;
    uint n = s / 4u;
    uint c = s % 4u;
    float d = f16tof32(wvec((b + 208u) >> 4u).x & 0xffffu);
    uint4 sv = wvec((b + 192u) >> 4u);
    uint si = 8u * n + 2u * c;
    Head h; h.ql = b + 64u * n + 32u * (c % 2u); h.qh = b + 128u + 32u * n;
    h.low_shift = 4u * (c / 2u); h.high_shift = 2u * c;
    h.ds0 = d * float(sbyte16(sv, si)); h.ds1 = d * float(sbyte16(sv, si + 1u)); return h;
}
float3 scales(Head h) { return float3(h.ds0, h.ds1, 0.0); }
uint4 quad_i8(Head h, uint i0) {
    uint4 lv = wvec((h.ql + 4u * i0) >> 4u);
    uint4 hv = wvec((h.qh + 4u * i0) >> 4u);
    uint4 q = ((lv >> h.low_shift) & 0x0f0f0f0fu) | (((hv >> h.high_shift) & 0x03030303u) << 4u);
    return (q + 0x60606060u) ^ 0x80808080u;
}
