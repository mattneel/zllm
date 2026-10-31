// Per-head in-place RMSNorm times learned weight and SiLU or sigmoid gate
// from the fused input projection (kernels/gated_norm.wgsl).

#define WG 256
groupshared float partial[WG];

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint v_heads = param(0);
    const uint dim = param(1);
    const float eps = paramf(2);
    const uint z_row = param(3);
    const uint z_off = param(4);
    const uint sigmoid = param(5);
    const uint b = lid3.x;
    const uint rowbase = (wg.y * v_heads + wg.x) * dim;
    const bool in_head = b < dim;
    precise float v = 0.0f;
    if (in_head) v = ldf(1, rowbase + b);
    partial[b] = v * v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s /= 2u) {
        if (b < s) {
            precise float sum = partial[b] + partial[b + s];
            partial[b] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (in_head) {
        const float zz = ldf(2, wg.y * z_row + z_off + wg.x * dim + b);
        precise float r = 1.0f / sqrt(partial[0] / float(dim) + eps);
        precise float gate = sigmoid != 0u ? 1.0f / (1.0f + exp(-zz))
            : zz / (1.0f + exp(-zz));
        precise float result = v * r * ldwf(3, b) * gate;
        stf(1, rowbase + b, result);
    }
}
