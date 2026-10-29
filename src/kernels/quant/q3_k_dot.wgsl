// Q3_K for the decode multiply: 4 threads per (padded, 16-byte aligned)
// block. Part p = 2n + h reads 16 bytes of qs (half n, positions
// l = 16h..16h+15), the hmask bytes for the same positions, and the
// scales with d: 64 weights over the four 2-bit planes j, at block
// positions 128n + 32j + l.
const PARTS: u32 = 4u;

fn block_dot(row_byte: u32, blk: u32, xb4: u32, p: u32) -> f32 {
    let b = row_byte + blk * BLOCK_BYTES;
    let n = p / 2u;
    let h = p % 2u;
    let qv = w[(b + 32u + 32u * n + 16u * h) >> 4u];
    let hv = w[(b + 16u * h) >> 4u];
    let sv = w[(b + 96u) >> 4u];
    let sw = sv.xyz;
    let d_all = unpack2x16float(sv.w).x;
    var acc = 0.0;
    for (var j = 0u; j < 4u; j++) {
        let dl = d_all * f32(scale3(sw, 8u * n + 2u * j + h));
        for (var e = 0u; e < 4u; e++) {
            // q + 4·hbit for four positions at once, one byte each.
            let c = ((qv[e] >> (2u * j)) & 0x03030303u) | (((hv[e] >> (4u * n + j)) & 0x01010101u) << 2u);
            acc += dot(dl * (vec4f(unpack4xU8(c)) - 4.0), xs[xb4 + 32u * n + 8u * j + 4u * h + e]);
        }
    }
    return acc;
}

