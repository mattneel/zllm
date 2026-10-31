// Copy packed quantized blocks to aligned, zero-padded blocks
// (kernels/repad.wgsl).
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x + id.y * num_groups().x * 256u;
    const uint src_bytes = param(1), dst_bytes = param(2);
    const uint words = dst_bytes / 4u;
    if (i >= param(0) * words) return;
    const uint blk = i / words, off = 4u * (i % words);
    if (off >= src_bytes) {
        stu(2, i, 0u);
        return;
    }
    const uint s = blk * src_bytes + off, q = s >> 2u;
    uint v = ldu(1, q);
    if ((s & 2u) != 0u) v = (v >> 16u) | (ldu(1, q + 1u) << 16u);
    const uint own = min(4u, src_bytes - off);
    if (own < 4u) v &= (1u << (8u * own)) - 1u;
    stu(2, i, v);
}
