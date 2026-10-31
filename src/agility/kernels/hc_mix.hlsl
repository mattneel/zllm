// Hyper-connection stream mix: sum(xn * sigmoid(gate)) / hc
// (kernels/hc_mix.wgsl). Grid: columns, tokens.

[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint dim = param(0);
    const uint hc = param(1);
    const uint t = wg.y;
    const uint i = wg.x * 256u + li;
    if (i >= dim) return;
    const uint start = t * hc * dim + i;
    precise float acc = 0.0f;
    for (uint c = 0u; c < hc; c++) {
        const float g = ldf(2, start + c * dim);
        acc += ldf(1, start + c * dim) / (1.0f + exp(-g));
    }
    stf(3, t * dim + i, acc / float(hc));
}
