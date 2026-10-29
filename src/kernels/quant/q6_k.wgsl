// Q6_K: blocks of 256 = ql[128] (low 4 bits), qh[64] (high 2 bits),
// scales[16] (i8, one per 16), d (f16). Unit s: half n = s/4 of the block,
// quarter c = s%4 of that half; y = d·sc·(q − 32).
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 210u;

fn unit(row_byte: u32, u: u32) -> array<f32, 32> {
    let b = row_byte + (u / 8u) * BLOCK_BYTES;
    let s = u % 8u;
    let n = s / 4u;
    let c = s % 4u;
    let d = whalf(b + 208u);
    let ql0 = b + 64u * n + 32u * (c % 2u);
    let low_shift = 4u * (c / 2u);
    let qh0 = b + 128u + 32u * n;
    var y: array<f32, 32>;
    for (var l = 0u; l < 32u; l++) {
        let q = ((wbyte(ql0 + l) >> low_shift) & 0xfu) | (((wbyte(qh0 + l) >> (2u * c)) & 3u) << 4u);
        let sc = wi8(b + 192u + 8u * n + l / 16u + 2u * c);
        y[l] = d * f32(sc) * f32(i32(q) - 32);
    }
    return y;
}

