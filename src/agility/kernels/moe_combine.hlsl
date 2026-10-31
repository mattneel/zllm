// Weighted routed-expert sum plus sigmoid-gated shared-expert output
// (kernels/moe_combine.wgsl). All inputs are GPU-produced activations.

[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint n_used = param(0);
    const uint dim = param(1);
    const uint t = wg.y;
    const uint i = wg.x * 256u + li;
    if (i >= dim) return;
    precise float acc = 0.0f;
    for (uint k = 0u; k < n_used; k++) {
        acc += ldf(2, t * n_used + k) * ldf(1, (t * n_used + k) * dim + i);
    }
    const float g = 1.0f / (1.0f + exp(-ldf(4, t)));
    precise float result = acc + g * ldf(3, t * dim + i);
    stf(5, t * dim + i, result);
}
