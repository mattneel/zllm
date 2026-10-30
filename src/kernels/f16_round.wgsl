// Shared by the kernels that store f16: composed before them (kernels.zig).

// f32 → f16 bits, rounded to nearest even; overflow to ±inf, NaN kept.
fn f16_bits(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let sign = (b >> 16u) & 0x8000u;
    let mag = b & 0x7fffffffu;
    if (mag > 0x7f800000u) {
        return sign | 0x7e00u;
    }
    let e = i32(mag >> 23u) - 112;
    if (e >= 31) {
        return sign | 0x7c00u;
    }
    if (e > 0) {
        // Normal: drop 13 mantissa bits; a carry moves into the exponent
        // (up to inf).
        let v = (u32(e) << 23u) | (mag & 0x7fffffu);
        return sign | ((v + 0xfffu + ((v >> 13u) & 1u)) >> 13u);
    }
    if (e < -10) {
        return sign;
    }
    // Subnormal: the significand in units of 2^-24.
    let s = (mag & 0x7fffffu) | 0x800000u;
    let k = u32(14 - e);
    let q = s >> k;
    let rem = s & ((1u << k) - 1u);
    let half = 1u << (k - 1u);
    let up = rem > half || (rem == half && (q & 1u) == 1u);
    return sign | (q + select(0u, 1u, up));
}

fn pack_f16(v: vec2f) -> u32 {
    return f16_bits(v.x) | (f16_bits(v.y) << 16u);
}
