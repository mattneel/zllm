// Shared by the kernels that store or read the FP8 KV cache: composed before
// them (kernels.zig). FP8 is E4M3FN: 1 sign, 4 exponent (bias 7) and 3
// mantissa bits; no infinities, largest finite 448, subnormals below 2^-6.
// The cache stores each (position, KV head) row divided by its scale, four
// values a word (lowest byte first), and the scale as f32 after the values
// (Params.scales). A scale is a power of two, the smallest with amax /
// scale ≤ 448, so applying it is exact on every backend (division need not
// be correctly rounded on a GPU) and the bytes match ref/fp8.zig's.

const FP8_MAX: f32 = 448.0;

// 2^k for k in [−126, 127], exactly.
fn pow2(k: i32) -> f32 {
    return bitcast<f32>(u32(k + 127) << 23u);
}

// The scale's exponent for a row whose largest magnitude is m: the
// smallest k with m ≤ 448 · 2^k (448 = 1.75 · 2^8); 0 for a zero row (or a
// subnormal m, which GPUs may flush).
fn fp8_scale_exp(m: f32) -> i32 {
    let bits = bitcast<u32>(m);
    let e = i32((bits >> 23u) & 255u);
    if (e == 0) {
        return 0;
    }
    return e - 127 - 8 + select(0, 1, (bits & 0x7fffffu) > 0x600000u);
}

// |x| ≤ 448 → E4M3FN bits, rounded to nearest, ties to even.
fn fp8_bits(x: f32) -> u32 {
    let bits = bitcast<u32>(x);
    let sign = (bits >> 24u) & 0x80u;
    let a = min(abs(x), FP8_MAX);
    if (a < 0x1p-6f) {
        // Subnormal (and zero): a · 2^9 is exact; codes 0..8, where 8 is
        // the smallest normal.
        let s = a * 512.0;
        var q = u32(floor(s));
        let r = s - f32(q);
        if (r > 0.5 || (r == 0.5 && (q & 1u) == 1u)) {
            q += 1u;
        }
        return sign | q;
    }
    let ab = bitcast<u32>(a);
    var e = (ab >> 23u) - 120u; // biased exponent: (e32 − 127) + 7
    let mant = ab & 0x7fffffu;
    var q = mant >> 20u;
    let rem = mant & 0xfffffu;
    if (rem > 0x80000u || (rem == 0x80000u && (q & 1u) == 1u)) {
        q += 1u;
    }
    if (q == 8u) {
        q = 0u;
        e += 1u;
    }
    return sign | min((e << 3u) | q, 0x7eu);
}

fn fp8_value(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = f32(b & 7u);
    let mag = select(ldexp(1.0 + m / 8.0, i32(e) - 7), m * 0x1p-9f, e == 0u);
    return select(mag, -mag, (b & 0x80u) != 0u);
}

// Value i of a word-packed FP8 array.
fn fp8_at(words: u32, i: u32) -> f32 {
    return fp8_value((words >> (8u * (i & 3u))) & 0xffu);
}
