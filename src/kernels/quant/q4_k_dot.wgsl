// Q4_K for the decode multiply: 8 threads per block. Part p reads the
// block's first 16 bytes (d, dmin, scales) and qs bytes 16p..16p+15:
// positions l = 16(p%2).. of chunk c = p/2, low nibbles for unit 2c, high
// for unit 2c+1.
const PARTS: u32 = 8u;

fn block_dot(row_byte: u32, blk: u32, xb4: u32, p: u32) -> f32 {
    let b = row_byte + blk * BLOCK_BYTES;
    let c = p / 2u;
    let hv = w[b >> 4u];
    let qv = w[(b + 16u + 16u * p) >> 4u];
    let dm = unpack2x16float(hv.x);
    let s1 = scale_min_words(hv.yzw, 2u * c);
    let s2 = scale_min_words(hv.yzw, 2u * c + 1u);
    let d1 = dm.x * f32(s1.x);
    let m1 = dm.y * f32(s1.y);
    let d2 = dm.x * f32(s2.x);
    let m2 = dm.y * f32(s2.y);
    let base = xb4 + 16u * c + 4u * (p % 2u);
    var acc = 0.0;
    for (var e = 0u; e < 4u; e++) {
        acc += dot(d1 * vec4f(unpack4xU8(qv[e] & 0x0f0f0f0fu)) - m1, xs[base + e]);
        acc += dot(d2 * vec4f(unpack4xU8((qv[e] >> 4u) & 0x0f0f0f0fu)) - m2, xs[base + 8u + e]);
    }
    return acc;
}

