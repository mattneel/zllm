// Q8_0: blocks of 32 = d (f16), qs[32] (i8). One block is one unit.
const BLOCK_ELEMS: u32 = 32u;
const BLOCK_BYTES: u32 = 34u;

fn unit(row_byte: u32, u: u32) -> array<f32, 32> {
    let b = row_byte + u * BLOCK_BYTES;
    let d = whalf(b);
    var y: array<f32, 32>;
    for (var l = 0u; l < 32u; l++) {
        y[l] = f32(wi8(b + 2u + l)) * d;
    }
    return y;
}

