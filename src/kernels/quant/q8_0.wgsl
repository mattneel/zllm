// Q8_0: blocks of 32 = d (f16), qs[32] (i8). One block is one unit. Blocks
// stay 34 bytes (kernels.blockStride), so reads are 32-bit.
const BLOCK_ELEMS: u32 = 32u;
const BLOCK_BYTES: u32 = 34u;

struct Head {
    qs: u32,
    d: f32,
}

fn head(row_byte: u32, u: u32) -> Head {
    let b = row_byte + u * BLOCK_BYTES;
    return Head(b + 2u, whalf(b));
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
