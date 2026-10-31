// Round f32 to f16 bits, nearest even, retaining NaN and signed infinity
// (kernels/f16_round.wgsl); shared by kernels storing packed f16.
uint f16_bits(float x) {
    const uint b = asuint(x);
    const uint sign = (b >> 16u) & 0x8000u;
    const uint mag = b & 0x7fffffffu;
    if (mag > 0x7f800000u) return sign | 0x7e00u;
    const int e = int(mag >> 23u) - 112;
    if (e >= 31) return sign | 0x7c00u;
    if (e > 0) {
        const uint v = (uint(e) << 23u) | (mag & 0x7fffffu);
        return sign | ((v + 0xfffu + ((v >> 13u) & 1u)) >> 13u);
    }
    if (e < -10) return sign;
    const uint s = (mag & 0x7fffffu) | 0x800000u;
    const uint k = uint(14 - e);
    const uint q = s >> k;
    const uint rem = s & ((1u << k) - 1u);
    const uint halfway = 1u << (k - 1u);
    const bool up = rem > halfway || (rem == halfway && (q & 1u) == 1u);
    return sign | (q + (up ? 1u : 0u));
}

uint pack_f16(float2 v) {
    return f16_bits(v.x) | (f16_bits(v.y) << 16u);
}
