// Unpack BF16 weights, low half first (kernels/half/bf16.wgsl).
float half_lo(uint word) { return asfloat(word << 16u); }
float half_hi(uint word) { return asfloat(word & 0xffff0000u); }
