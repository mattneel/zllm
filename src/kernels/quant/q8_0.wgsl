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

fn quad(h: Head, i0: u32) -> array<vec4f, 4> {
    var r: array<vec4f, 4>;
    for (var e = 0u; e < 4u; e++) {
        let q = bitcast<vec4i>(vec4u(w32(h.qs + 4u * (i0 + e))) << vec4u(24u, 16u, 8u, 0u)) >> vec4u(24u);
        r[e] = vec4f(q) * h.d;
    }
    return r;
}
