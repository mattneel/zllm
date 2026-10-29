// Q3_K: blocks of 256 = hmask[32] (third bits), qs[64] (2 bits each),
// scales[12] (16 6-bit scales), d (f16). Unit s: half n = s/4, shift
// j = s%4 within it; weight l takes bits 2j of qs[32n + l], minus 4 unless
// bit 4n + j of hmask[l] is set, times d·(scale − 32) for scale
// 8n + 2j + l/16.
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 110u;

// Scale i (0..16): low 4 bits from scale bytes 0..8, high 2 bits from 8..12.
fn scale3(sb: u32, i: u32) -> i32 {
    let k = i % 4u;
    let g = i / 4u;
    let low = (wbyte(sb + 4u * (g % 2u) + k) >> (4u * (g / 2u))) & 0xfu;
    let high = (wbyte(sb + 8u + k) >> (2u * g)) & 3u;
    return i32(low | (high << 4u)) - 32;
}

fn unit(row_byte: u32, u: u32) -> array<f32, 32> {
    let b = row_byte + (u / 8u) * BLOCK_BYTES;
    let s = u % 8u;
    let n = s / 4u;
    let j = s % 4u;
    let d_all = whalf(b + 108u);
    let dl0 = d_all * f32(scale3(b + 96u, 8u * n + 2u * j));
    let dl1 = d_all * f32(scale3(b + 96u, 8u * n + 2u * j + 1u));
    let bit = 4u * n + j;
    var y: array<f32, 32>;
    for (var l = 0u; l < 32u; l++) {
        let q = i32((wbyte(b + 32u + 32u * n + l) >> (2u * j)) & 3u);
        let sub = select(4, 0, ((wbyte(b + l) >> bit) & 1u) != 0u);
        y[l] = select(dl0, dl1, l >= 16u) * f32(q - sub);
    }
    return y;
}

