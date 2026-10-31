// attn_split's sparse key selection: the tail and selected block prefixes
// (kernels/attn_sparse.wgsl). Bindings 6: selection, 7: Qsa uniform.
bool included(uint row, uint key) {
    const uint visible = ldu(5, 1) + row + 1u;
    const uint width = ldu(7, 3);
    if (visible <= width) return true;
    const uint ratio = ldu(7, 2);
    const uint tail = visible / ratio * ratio;
    if (key >= tail) return true;
    return (key % ratio) < ldu(6, row * ldu(7, 4) + key / ratio);
}
