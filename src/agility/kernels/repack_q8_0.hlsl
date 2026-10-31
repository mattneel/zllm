// Q8_0 rows as in the GGUF file (34-byte blocks) to split rows, as
// kernels/repack_q8_0.wgsl: the k values as i8, then the k/32 scales as f16,
// padded to 16 bytes. One thread per destination word, the grid folded into
// y past 65,535 groups. Params: rows, k. Bindings: src (read-only), dst.

uint src_byte(uint i) { return (ldw(1, i >> 2u) >> ((i & 3u) * 8u)) & 0xffu; }
uint src_half(uint i) { return src_byte(i) | (src_byte(i + 1u) << 8u); }

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint rows = param(0), k = param(1);
    const uint stride = (k + k / 16u + 15u) & ~15u;
    const uint i = id.x + id.y * num_groups().x * 256u;
    if (i >= rows * (stride / 4u)) return;
    const uint r = i / (stride / 4u);
    const uint o = 4u * (i % (stride / 4u));
    const uint raw = r * (k / 32u) * 34u;
    const uint blocks = k / 32u;
    uint word = 0u;
    if (o < k) {
        const uint b = raw + (o / 32u) * 34u + 2u + o % 32u;
        word = src_byte(b) | (src_byte(b + 1u) << 8u) | (src_byte(b + 2u) << 16u) | (src_byte(b + 3u) << 24u);
    } else if (o < k + k / 16u) {
        const uint s = (o - k) / 2u;
        word = src_half(raw + s * 34u);
        if (s + 1u < blocks) word |= src_half(raw + (s + 1u) * 34u) << 16u;
    }
    stu(2, i, word);
}
