// Inject a block's output into each residual stream with the mixer's
// 2 * sigmoid(inject / hc) weights (kernels/hc_combine.wgsl).

[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint dim = param(0);
    const uint hc = param(1);
    const uint t = wg.y;
    const uint i = wg.x * 256u + li;
    if (i >= dim) return;
    const float b = ldf(1, t * dim + i);
    for (uint c = 0u; c < hc; c++) {
        precise float w = 2.0f / (1.0f + exp(-ldf(2, t * param(2) + param(3) + c) / float(hc)));
        const uint index = (t * hc + c) * dim + i;
        precise float result = ldf(3, index) + b * w;
        stf(3, index, result);
    }
}
