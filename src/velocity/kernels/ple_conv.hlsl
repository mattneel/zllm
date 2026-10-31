// PLE causal dilated convolution, SiLU residual update and in-place history
// advance, one channel per thread (kernels/ple_conv.wgsl).

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint ch = id.x;
    const uint n = param(0);
    if (ch >= n) return;
    const uint taps = param(1);
    const uint dilation = param(2);
    const uint rows = (taps - 1u) * dilation;
    // Binding 6 is the GPU-produced Step uniform; m is its first word.
    const uint m = ldu(6, 0u);
    for (uint t = 0u; t < m; t++) {
        precise float acc = 0.0f;
        for (uint k = 0u; k < taps; k++) {
            const uint back = (taps - 1u - k) * dilation;
            float v = 0.0f;
            if (t >= back) v = ldf(2, (t - back) * n + ch);
            else v = ldf(4, (rows - back + t) * n + ch);
            acc += ldwf(3, ch * taps + k) * v;
        }
        precise float update = ldf(1, t * n + ch) + acc / (1.0f + exp(-acc));
        precise float result = ldf(5, t * n + ch) + update;
        stf(5, t * n + ch, result);
    }
    for (uint i = 0u; i < rows; i++) {
        const uint j = m + i;
        if (j < rows) stf(4, i * n + ch, ldf(4, j * n + ch));
        else stf(4, i * n + ch, ldf(2, (j - rows) * n + ch));
    }
}
