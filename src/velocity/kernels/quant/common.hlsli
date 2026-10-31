// Quantized weight reads and packed-byte helpers from quant/common.wgsl.
// Weights occupy binding 2; all offsets below are relative to its view.
uint wword(uint q) { return ldw(2, q); }
uint4 wvec(uint q) { return ro(2).Load4(base(2) + 16u * q); }
uint wbyte(uint i) { return (wword(i >> 2u) >> ((i & 3u) * 8u)) & 0xffu; }
float whalf(uint i) { return f16tof32(wbyte(i) | (wbyte(i + 1u) << 8u)); }
float2 unpack_half2(uint v) { return float2(f16tof32(v & 0xffffu), f16tof32(v >> 16u)); }
uint w32(uint i) {
    uint q = i >> 2u;
    if ((i & 2u) == 0u) return wword(q);
    return (wword(q) >> 16u) | (wword(q + 1u) << 16u);
}
uint byte12(uint3 s, uint j) { return (s[j / 4u] >> (8u * (j % 4u))) & 0xffu; }
uint4 unpack_u8(uint v) { return (v.xxxx >> uint4(0u, 8u, 16u, 24u)) & 0xffu; }
int4 unpack_i8(uint v) { return asint(v.xxxx << uint4(24u, 16u, 8u, 0u)) >> 24u; }
// Integer products and sums are exact; match WGSL dot4I8Packed without waves.
int dot_i8(uint a, uint b) {
    int4 p = unpack_i8(a) * unpack_i8(b);
    return p.x + p.y + p.z + p.w;
}
