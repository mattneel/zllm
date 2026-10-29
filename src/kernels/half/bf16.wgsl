// BF16 weights for the half-precision kernels (matvec_half, matmul_half,
// embed_half): two values per word, the low half first.
fn half_lo(word: u32) -> f32 {
    return bitcast<f32>(word << 16u);
}

fn half_hi(word: u32) -> f32 {
    return bitcast<f32>(word & 0xffff0000u);
}

