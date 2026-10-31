// Score each complete block as the sum of positive query-head dot products
// with its pooled key, skipping rows selecting all cells (kernels/qsa_score.wgsl).
#define WG 256
#define MAX_Q 2048
groupshared float q[MAX_Q];

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint t = wg.y;
    const uint d = param(1);
    const uint visible = ldu(4, 1) + t + 1u;
    if (visible <= param(3)) return;
    for (uint k = lid3.x; k < param(0) * d; k += WG) q[k] = ldf(1, t * param(7) + k);
    GroupMemoryBarrierWithGroupSync();
    const uint b = wg.x * WG + lid3.x;
    if (b >= visible / param(2)) return;
    precise float s = 0.0f;
    for (uint h = 0u; h < param(0); h++) {
        precise float product = 0.0f;
        for (uint i = 0u; i < d; i++) product += q[h * d + i] * ldf(2, b * d + i);
        s += max(product, 0.0f);
    }
    stf(3, t * param(4) + b, s);
}
