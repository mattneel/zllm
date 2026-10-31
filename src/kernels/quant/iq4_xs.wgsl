// IQ4_XS: 256 weights in d(f16), scales_h(u16), scales_l[4], qs[128].
// Each 32-weight unit has one signed 6-bit scale (stored biased by 32)
// and 16 bytes of indices: low nibbles first, then high nibbles. The
// nonlinear codebook is ggml's kvalues_iq4nl, not a linear signed nibble.
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 144u; // 136 file bytes, padded by repad.
const ACT_BLOCK: u32 = 256u;
const HAS_MIN: bool = false;
const IQ4_VALUES = array<i32, 16>(-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);

fn row_stride(k: u32) -> u32 {
    return (k / BLOCK_ELEMS) * BLOCK_BYTES;
}

struct Head {
    qs: u32,
    ds: f32,
}

fn head(row_byte: u32, u: u32) -> Head {
    // Explicit embedding strides select raw GGUF rows, as for Q8_0.
    let block_bytes = select(BLOCK_BYTES, 136u, row_k == 0u);
    let b = row_byte + (u / 8u) * block_bytes;
    let s = u % 8u;
    let lo = (wbyte(b + 4u + s / 2u) >> (4u * (s % 2u))) & 15u;
    let hi = (wword(b >> 2u) >> (16u + 2u * s)) & 3u;
    let scale = i32(lo | (hi << 4u)) - 32;
    return Head(b + 8u + 16u * s, whalf(b) * f32(scale));
}

fn scales(h: Head) -> vec3f {
    return vec3f(h.ds, h.ds, 0.0);
}

fn quad_i8(h: Head, i0: u32) -> vec4u {
    var q: vec4u;
    for (var e = 0u; e < 4u; e++) {
        let indices = (wword((h.qs >> 2u) + e) >> select(0u, 4u, i0 >= 4u)) & 0x0f0f0f0fu;
        let v = unpack4xU8(indices);
        q[e] = pack4xI8(vec4i(IQ4_VALUES[v.x], IQ4_VALUES[v.y], IQ4_VALUES[v.z], IQ4_VALUES[v.w]));
    }
    return q;
}
