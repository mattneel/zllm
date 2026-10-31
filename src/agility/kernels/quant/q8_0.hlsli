// Q8_0 from quant/q8_0.wgsl: uploaded rows are split (repack_q8_0.hlsl): the
// row's k values as i8, then its k/32 scales as f16, padded to 16 bytes. Rows
// read as in the file (row_k == 0) keep 34-byte blocks.
static const uint BLOCK_ELEMS = 32u;
static const uint BLOCK_BYTES = 34u;
static const uint ACT_BLOCK = 32u;
static const bool HAS_MIN = false;
struct Head { uint qs; float d; };
uint row_stride(uint k) { return (k + k / 16u + 15u) & ~15u; }
Head head(uint row_byte, uint u) {
    Head h;
    if (row_k == 0u) {
        uint b = row_byte + u * BLOCK_BYTES;
        h.qs = b + 2u; h.d = whalf(b);
    } else {
        h.qs = row_byte + 32u * u; h.d = whalf(row_byte + row_k + 2u * u);
    }
    return h;
}
float3 scales(Head h) { return float3(h.d, h.d, 0.0); }
uint4 quad_i8(Head h, uint i0) {
    return uint4(w32(h.qs + 4u * i0), w32(h.qs + 4u * i0 + 4u), w32(h.qs + 4u * i0 + 8u), w32(h.qs + 4u * i0 + 12u));
}
