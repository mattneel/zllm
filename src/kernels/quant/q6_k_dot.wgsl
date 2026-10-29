// Q6_K for the decode multiply: 8 threads per (padded) block. Part p: half
// n = p/4, quarter pair cp = (p/2)%2, positions l = 16(p%2)..: 16 ql bytes
// (low nibbles quarter cp, high nibbles quarter cp+2) and the qh bytes for
// the same positions: 32 weights.
const PARTS: u32 = 8u;

fn block_dot(row_byte: u32, blk: u32, xb4: u32, p: u32) -> f32 {
    let b = row_byte + blk * BLOCK_BYTES;
    let n = p / 4u;
    let cp = (p / 2u) % 2u;
    let hh = p % 2u;
    let qlv = w[(b + 64u * n + 32u * cp + 16u * hh) >> 4u];
    let qhv = w[(b + 128u + 32u * n + 16u * hh) >> 4u];
    let sv = w[(b + 192u) >> 4u];
    let d = unpack2x16float(w[(b + 208u) >> 4u].x).x;
    var acc = 0.0;
    for (var e2 = 0u; e2 < 2u; e2++) {
        let c = cp + 2u * e2;
        let si = 8u * n + hh + 2u * c;
        let ds = d * f32(sbyte16(sv, si));
        for (var e = 0u; e < 4u; e++) {
            let q = ((qlv[e] >> (4u * e2)) & 0x0f0f0f0fu) | (((qhv[e] >> (2u * c)) & 0x03030303u) << 4u);
            acc += dot(ds * (vec4f(unpack4xU8(q)) - 32.0), xs[xb4 + 32u * n + 8u * c + 4u * hh + e]);
        }
    }
    return acc;
}

