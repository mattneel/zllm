// Q8_0 for the decode multiply, on split rows (q8_0.wgsl): 2 threads per
// block, 16 weights each from one aligned 16-byte load.
const PARTS: u32 = 2u;

fn block_dot(row_byte: u32, blk: u32, xb4: u32, p: u32) -> f32 {
    let q = w[(row_byte + 32u * blk) / 16u + p];
    let d = whalf(row_byte + row_k + 2u * blk);
    var s = 0.0;
    for (var e = 0u; e < 4u; e++) {
        let v = bitcast<vec4i>(vec4u(q[e]) << vec4u(24u, 16u, 8u, 0u)) >> vec4u(24u);
        s += dot(vec4f(v) * d, xs[xb4 + 4u * p + e]);
    }
    return s;
}

// The same for two tokens: the second's x starts XT vectors on.
fn block_dot2(row_byte: u32, blk: u32, xb4: u32, p: u32) -> vec2f {
    let q = w[(row_byte + 32u * blk) / 16u + p];
    let d = whalf(row_byte + row_k + 2u * blk);
    var s = vec2f(0.0);
    for (var e = 0u; e < 4u; e++) {
        let v = bitcast<vec4i>(vec4u(q[e]) << vec4u(24u, 16u, 8u, 0u)) >> vec4u(24u);
        let wv = vec4f(v) * d;
        s += vec2f(dot(wv, xs[xb4 + 4u * p + e]), dot(wv, xs[xb4 + XT + 4u * p + e]));
    }
    return s;
}
