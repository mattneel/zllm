// First index of the maximum logit in one speculative draft row, written
// to the requested token slot (kernels/draft.wgsl).

#define WG 256
groupshared float best_v[WG];
groupshared uint best_i[WG];

[numthreads(WG, 1, 1)]
void main(uint3 lid3 : SV_GroupThreadID) {
    const uint vocab = param(0);
    const uint row = param(1);
    const uint slot = param(2);
    const uint lid = lid3.x;
    float v = 0.0f;
    uint at = 0xffffffffu;
    for (uint i = lid; i < vocab; i += WG) {
        const float x = ldf(1, row + i);
        if (at == 0xffffffffu || x > v) {
            v = x;
            at = i;
        }
    }
    best_v[lid] = v;
    best_i[lid] = at;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = WG / 2u; stride > 0u; stride /= 2u) {
        if (lid < stride) {
            const float ov = best_v[lid + stride];
            const uint oi = best_i[lid + stride];
            const bool none = best_i[lid] == 0xffffffffu;
            if (oi != 0xffffffffu && (none || ov > best_v[lid] || (ov == best_v[lid] && oi < best_i[lid]))) {
                best_v[lid] = ov;
                best_i[lid] = oi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0u) stu(2, slot, best_i[0]);
}
