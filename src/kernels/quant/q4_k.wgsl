// Q4_K: blocks of 256 = d, dmin (f16), scales[12] (6-bit scale and min per
// 32), qs[128]. Unit s of a block: bytes 32·(s/2).. of qs, the low nibble
// for even s and the high one for odd s; y = d·sc·q − dmin·m.
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 144u;

// scale_min_k4: the 6-bit scale and min of sub-block j, from the 12 scale
// bytes held as three words.
fn scale_min_words(sw: vec3u, j: u32) -> vec2u {
    if (j < 4u) {
        return vec2u(byte12(sw, j) & 63u, byte12(sw, j + 4u) & 63u);
    }
    return vec2u((byte12(sw, j + 4u) & 0xfu) | ((byte12(sw, j - 4u) >> 6u) << 4u),
                 (byte12(sw, j + 4u) >> 4u) | ((byte12(sw, j) >> 6u) << 4u));
}

struct Head {
    qs: u32,
    shift: u32,
    d1: f32,
    m1: f32,
}

fn head(row_byte: u32, u: u32) -> Head {
    let b = row_byte + (u / 8u) * BLOCK_BYTES;
    let s = u % 8u;
    let hv = w[b >> 4u];
    let dm = unpack2x16float(hv.x);
    let sm = scale_min_words(hv.yzw, s);
    return Head(b + 16u + 32u * (s / 2u), 4u * (s % 2u), dm.x * f32(sm.x), dm.y * f32(sm.y));
}

fn quad(h: Head, i0: u32) -> array<vec4f, 4> {
    let qv = w[(h.qs + 4u * i0) >> 4u];
    var r: array<vec4f, 4>;
    for (var e = 0u; e < 4u; e++) {
        r[e] = h.d1 * vec4f(unpack4xU8((qv[e] >> h.shift) & 0x0f0f0f0fu)) - h.m1;
    }
    return r;
}
