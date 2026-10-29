// F16 weights for the half-precision kernels (matvec_half, matmul_half,
// embed_half): two values per word, the low half first.
fn half_lo(word: u32) -> f32 {
    return unpack2x16float(word).x;
}

fn half_hi(word: u32) -> f32 {
    return unpack2x16float(word).y;
}

