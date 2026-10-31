// IQ4_XS block layout and nonlinear signed-byte codebook from iq4_xs.wgsl.
static const uint BLOCK_ELEMS = 256u;
static const uint BLOCK_BYTES = 144u; // 136 file bytes, padded by repad.
static const uint ACT_BLOCK = 256u;
static const bool HAS_MIN = false;
static const int IQ4_VALUES[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
uint row_stride(uint k) { return (k / BLOCK_ELEMS) * BLOCK_BYTES; }
struct Head { uint qs; float ds; };
Head head(uint row_byte, uint u) {
    uint block_bytes = row_k == 0u ? 136u : BLOCK_BYTES;
    uint b = row_byte + (u / 8u) * block_bytes;
    uint s = u % 8u;
    uint lo = (wbyte(b + 4u + s / 2u) >> (4u * (s % 2u))) & 15u;
    uint hi = (wword(b >> 2u) >> (16u + 2u * s)) & 3u;
    int scale = int(lo | (hi << 4u)) - 32;
    Head h; h.qs = b + 8u + 16u * s; h.ds = whalf(b) * float(scale); return h;
}
float3 scales(Head h) { return float3(h.ds, h.ds, 0.0); }
uint4 quad_i8(Head h, uint i0) {
    uint4 q;
    for (uint e = 0u; e < 4u; e++) {
        uint indices = (wword((h.qs >> 2u) + e) >> (i0 >= 4u ? 4u : 0u)) & 0x0f0f0f0fu;
        uint4 v = unpack_u8(indices);
        q[e] = (uint(IQ4_VALUES[v.x]) & 0xffu) | ((uint(IQ4_VALUES[v.y]) & 0xffu) << 8u) |
            ((uint(IQ4_VALUES[v.z]) & 0xffu) << 16u) | ((uint(IQ4_VALUES[v.w]) & 0xffu) << 24u);
    }
    return q;
}
