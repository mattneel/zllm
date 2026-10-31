// Router and shared-gate logits, eight 32-lane dot products per group
// (kernels/moe_logits.wgsl). Bindings: router, shared_gate, x, logits.

#define WG 256
#define LANES 32
#define GROUPS (WG / LANES)
#define MAX_COLS 2560

groupshared float xs[MAX_COLS];
groupshared float red[WG];

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint n_expert = param(0);
    const uint cols = param(2);
    const uint t = wg.y;
    const uint lane = li % LANES;
    const uint group = li / LANES;
    for (uint i = li; i < cols; i += WG) {
        xs[i] = ldf(3, t * param(3) + i);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint e = wg.x * GROUPS + group;
    precise float acc = 0.0f;
    if (e < n_expert) {
        for (uint i = lane; i < cols; i += LANES) {
            acc += ldwf(1, e * cols + i) * xs[i];
        }
    } else if (e == n_expert) {
        for (uint i = lane; i < cols; i += LANES) {
            acc += ldwf(2, i) * xs[i];
        }
    }
    red[li] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = LANES / 2u; s > 0u; s /= 2u) {
        if (lane < s) red[li] += red[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u && e <= n_expert) {
        stf(4, t * (n_expert + 1u) + e, red[li]);
    }
}
