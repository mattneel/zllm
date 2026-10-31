// Q8_0: blocks of 32 = d (f16), qs[32] (i8), 34 bytes in the file. Uploaded
// rows are split (repack_q8_0.wgsl): the row's k values as i8, then its k/32
// scales as f16, then zero padding to 16 bytes, so every unit's values take
// aligned loads and the rows stay 16-byte aligned. Rows read as in the file
// (row_k == 0) keep 34-byte blocks.
const BLOCK_ELEMS: u32 = 32u;
const BLOCK_BYTES: u32 = 34u;

struct Head {
    qs: u32,
    d: f32,
}

fn row_stride(k: u32) -> u32 {
    return (k + k / 16u + 15u) & ~15u;
}

fn head(row_byte: u32, u: u32) -> Head {
    if (row_k == 0u) {
        let b = row_byte + u * BLOCK_BYTES;
        return Head(b + 2u, whalf(b));
    }
    return Head(row_byte + 32u * u, whalf(row_byte + row_k + 2u * u));
}

// Activations round to Q8_0 blocks for these weights.
const ACT_BLOCK: u32 = 32u;
const HAS_MIN: bool = false;

fn scales(h: Head) -> vec3f {
    return vec3f(h.d, h.d, 0.0);
}

fn quad_i8(h: Head, i0: u32) -> vec4u {
    return vec4u(w32(h.qs + 4u * i0), w32(h.qs + 4u * i0 + 4u), w32(h.qs + 4u * i0 + 8u), w32(h.qs + 4u * i0 + 12u));
}
