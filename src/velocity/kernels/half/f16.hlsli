// Unpack F16 weights, low half first (kernels/half/f16.wgsl).
float half_lo(uint word) { return f16tof32(word & 0xffffu); }
float half_hi(uint word) { return f16tof32(word >> 16u); }
