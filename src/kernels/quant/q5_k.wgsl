// Q5_K: blocks of 256 = d, dmin (f16), scales[12], qh[32] (fifth bits),
// qs[128]. As Q4_K, plus 16 when bit s of qh[l] is set.
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 176u;

// scale_min_k4, as for Q4_K.
fn scale_min_words(sw: vec3u, j: u32) -> vec2u {
    if (j < 4u) {
        return vec2u(byte12(sw, j) & 63u, byte12(sw, j + 4u) & 63u);
    }
    return vec2u((byte12(sw, j + 4u) & 0xfu) | ((byte12(sw, j - 4u) >> 6u) << 4u),
                 (byte12(sw, j + 4u) >> 4u) | ((byte12(sw, j) >> 6u) << 4u));
}

struct Head {
    qs: u32,
    qh: u32,
    shift: u32,
    bit: u32,
    d1: f32,
    m1: f32,
}

fn head(row_byte: u32, u: u32) -> Head {
    let b = row_byte + (u / 8u) * BLOCK_BYTES;
    let s = u % 8u;
    let hv = w[b >> 4u];
    let dm = unpack2x16float(hv.x);
    let sm = scale_min_words(hv.yzw, s);
    return Head(b + 48u + 32u * (s / 2u), b + 16u, 4u * (s % 2u), s, dm.x * f32(sm.x), dm.y * f32(sm.y));
}

const ACT_BLOCK: u32 = 256u;
const HAS_MIN: bool = true;

fn scales(h: Head) -> vec3f {
    return vec3f(h.d1, h.d1, h.m1);
}

fn quad_i8(h: Head, i0: u32) -> vec4u {
    let qv = w[(h.qs + 4u * i0) >> 4u];
    let hv = w[(h.qh + 4u * i0) >> 4u];
    return ((qv >> vec4u(h.shift)) & vec4u(0x0f0f0f0fu)) | (((hv >> vec4u(h.bit)) & vec4u(0x01010101u)) << vec4u(4u));
}
