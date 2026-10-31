// FP8 (E4M3FN) for the KV cache, as kernels/fp8.wgsl: 1 sign, 4 exponent
// (bias 7) and 3 mantissa bits; largest finite 448; four values a word,
// lowest byte first; each (position, KV head) row divided by its scale, the
// smallest power of two with amax / scale ≤ 448, so it applies exactly.

static const float FP8_MAX = 448.0f;

// 2^k for k in [−126, 127], exactly.
float pow2(int k) { return asfloat(uint(k + 127) << 23); }

// The scale's exponent for a row whose largest magnitude is m (448 = 1.75 ·
// 2^8); 0 for a zero (or subnormal, flushed) m.
int fp8_scale_exp(float m) {
    const uint bits = asuint(m);
    const int e = int((bits >> 23) & 255u);
    if (e == 0) return 0;
    return e - 127 - 8 + ((bits & 0x7fffffu) > 0x600000u ? 1 : 0);
}

// |x| ≤ 448 → E4M3FN bits, rounded to nearest, ties to even.
uint fp8_bits(float x) {
    const uint sign = (asuint(x) >> 24) & 0x80u;
    const float a = min(abs(x), FP8_MAX);
    if (a < 0.015625f) {
        // Subnormal (and zero): a · 2^9 is exact; codes 0..8.
        const float s = a * 512.0f;
        uint q = uint(floor(s));
        const float r = s - float(q);
        if (r > 0.5f || (r == 0.5f && (q & 1u) == 1u)) q += 1u;
        return sign | q;
    }
    const uint ab = asuint(a);
    uint e = (ab >> 23) - 120u;
    const uint mant = ab & 0x7fffffu;
    uint q = mant >> 20;
    const uint rem = mant & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (q & 1u) == 1u)) q += 1u;
    if (q == 8u) {
        q = 0u;
        e += 1u;
    }
    return sign | min((e << 3) | q, 0x7eu);
}

float fp8_value(uint b) {
    const uint e = (b >> 3) & 15u;
    const float m = float(b & 7u);
    const float mag = e == 0u ? m * 0.001953125f : ldexp(1.0f + m / 8.0f, float(int(e) - 7));
    return (b & 0x80u) != 0u ? -mag : mag;
}

// Value i of a word-packed FP8 array, from the word holding it.
float fp8_at(uint w, uint i) {
    return fp8_value((w >> (8u * (i & 3u))) & 0xffu);
}
