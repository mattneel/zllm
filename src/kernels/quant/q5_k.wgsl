// Q5_K: blocks of 256 = d, dmin (f16), scales[12], qh[32] (fifth bits),
// qs[128]. As Q4_K, plus 16 when bit s of qh[l] is set.
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 176u;

fn scale_min_k4(sb: u32, j: u32) -> vec2u {
    if (j < 4u) {
        return vec2u(wbyte(sb + j) & 63u, wbyte(sb + j + 4u) & 63u);
    }
    return vec2u((wbyte(sb + j + 4u) & 0xfu) | ((wbyte(sb + j - 4u) >> 6u) << 4u),
                 (wbyte(sb + j + 4u) >> 4u) | ((wbyte(sb + j) >> 6u) << 4u));
}

fn unit(row_byte: u32, u: u32) -> array<f32, 32> {
    let b = row_byte + (u / 8u) * BLOCK_BYTES;
    let s = u % 8u;
    let sm = scale_min_k4(b + 4u, s);
    let d1 = whalf(b) * f32(sm.x);
    let m1 = whalf(b + 2u) * f32(sm.y);
    let q0 = b + 48u + 32u * (s / 2u);
    let shift = 4u * (s % 2u);
    var y: array<f32, 32>;
    for (var l = 0u; l < 32u; l++) {
        let high = select(0u, 16u, ((wbyte(b + 16u + l) >> s) & 1u) != 0u);
        y[l] = d1 * f32(((wbyte(q0 + l) >> shift) & 0xfu) + high) - m1;
    }
    return y;
}

