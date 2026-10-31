// attn_split's dense key selection: every causal key (kernels/attn_dense.wgsl).
bool included(uint row, uint key) {
    return true;
}
