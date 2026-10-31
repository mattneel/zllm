// Softmax routing, repeated top-k argmax (lower expert index wins ties),
// normalized weights and shared-gate passthrough (kernels/moe_route.wgsl).

#define WG 256
#define MAX_EXPERT 512

groupshared float probs[MAX_EXPERT];
groupshared float red[WG];
groupshared uint redi[WG];
groupshared float chosen[16];

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint n_expert = param(0);
    const uint n_used = param(1);
    const uint t = wg.x;
    const uint row = t * (n_expert + 1u);
    for (uint e = li; e < n_expert; e += WG) probs[e] = ldf(1, row + e);
    if (li == 0u) stf(4, t, ldf(1, row + n_expert));
    GroupMemoryBarrierWithGroupSync();

    float m = -3.0e38f;
    for (uint e = li; e < n_expert; e += WG) m = max(m, probs[e]);
    red[li] = m;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s /= 2u) {
        if (li < s) red[li] = max(red[li], red[li + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    m = red[0];
    GroupMemoryBarrierWithGroupSync();
    precise float sum = 0.0f;
    for (uint e = li; e < n_expert; e += WG) {
        const float v = exp(probs[e] - m);
        probs[e] = v;
        sum += v;
    }
    red[li] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s /= 2u) {
        if (li < s) red[li] += red[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    sum = red[0];
    GroupMemoryBarrierWithGroupSync();
    for (uint e = li; e < n_expert; e += WG) probs[e] = probs[e] / sum;
    GroupMemoryBarrierWithGroupSync();

    precise float total = 0.0f;
    for (uint k = 0u; k < n_used; k++) {
        float best = -1.0f;
        uint best_i = 0xffffffffu;
        for (uint e = li; e < n_expert; e += WG) {
            if (probs[e] > best) {
                best = probs[e];
                best_i = e;
            }
        }
        red[li] = best;
        redi[li] = best_i;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG / 2u; s > 0u; s /= 2u) {
            if (li < s) {
                const float ov = red[li + s];
                const uint oi = redi[li + s];
                if (ov > red[li] || (ov == red[li] && oi < redi[li])) {
                    red[li] = ov;
                    redi[li] = oi;
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        const float gb = red[0];
        const uint gi = redi[0];
        total += gb;
        if (li == 0u) {
            stu(2, t * n_used + k, gi);
            chosen[k] = gb;
            probs[gi] = -1.0f;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (li < n_used) {
        precise float w = chosen[li] / max(total, 6.103515625e-5f);
        const float scale = paramf(4);
        if (scale != 0.0f && scale != 1.0f) w *= scale;
        stf(3, t * n_used + li, w);
    }
}
