// Q8_0 for the decode multiply: 8 threads per block, a word (4 weights)
// each.
const PARTS: u32 = 8u;

fn block_dot(row_byte: u32, blk: u32, xb4: u32, p: u32) -> f32 {
    let b = row_byte + blk * BLOCK_BYTES;
    let q = bitcast<vec4i>(vec4u(w32(b + 2u + 4u * p)) << vec4u(24u, 16u, 8u, 0u)) >> vec4u(24u);
    return dot(vec4f(q) * whalf(b), xs[xb4 + p]);
}

// The same for two tokens: the second's x starts XT vectors on.
fn block_dot2(row_byte: u32, blk: u32, xb4: u32, p: u32) -> vec2f {
    let b = row_byte + blk * BLOCK_BYTES;
    let q = bitcast<vec4i>(vec4u(w32(b + 2u + 4u * p)) << vec4u(24u, 16u, 8u, 0u)) >> vec4u(24u);
    let wv = vec4f(q) * whalf(b);
    return vec2f(dot(wv, xs[xb4 + p]), dot(wv, xs[xb4 + XT + p]));
}
