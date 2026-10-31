// Select the highest-score blocks, ties by lower index, cutting the last
// block to fit the cell budget (kernels/qsa_select.wgsl). One group per row.
#define WG 256
groupshared uint red[WG];

uint count_at_least(uint offset, uint nb, uint at, uint lid) {
    uint c = 0u;
    for (uint b = lid; b < nb; b += WG) c += ldu(1, offset + b) >= at ? 1u : 0u;
    red[lid] = c;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s /= 2u) {
        if (lid < s) red[lid] += red[lid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    // workgroupUniformLoad: synchronize both before and after the shared read.
    GroupMemoryBarrierWithGroupSync();
    const uint total = red[0];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint lid = lid3.x;
    const uint t = wg.y;
    const uint r = param(2);
    const uint visible = ldu(3, 1) + t + 1u;
    if (visible <= param(3)) return;
    const uint nb = visible / r;
    const uint cells = param(3) - (visible - nb * r);
    const uint k = (cells + r - 1u) / r;
    const uint last = cells - (k - 1u) * r;
    const uint offset = t * param(4);

    uint at = 0u;
    for (int bit = 31; bit >= 0; bit--) {
        const uint candidate = at | (1u << uint(bit));
        if (count_at_least(offset, nb, candidate, lid) >= k) at = candidate;
    }
    const uint above = count_at_least(offset, nb, at + 1u, lid);
    const uint ties = k - above;

    const uint chunk = (nb + WG - 1u) / WG;
    const uint lo = min(lid * chunk, nb);
    const uint hi = min(lo + chunk, nb);
    uint mine = 0u;
    for (uint b = lo; b < hi; b++) mine += ldu(1, offset + b) == at ? 1u : 0u;
    red[lid] = mine;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 1u; s < WG; s *= 2u) {
        uint add = 0u;
        if (lid >= s) add = red[lid - s];
        GroupMemoryBarrierWithGroupSync();
        red[lid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint rank = red[lid] - mine;
    for (uint b = lo; b < hi; b++) {
        const uint bits = ldu(1, offset + b);
        uint take = 0u;
        if (bits > at) {
            take = r;
        } else if (bits == at) {
            if (rank < ties) take = rank + 1u == ties ? last : r;
            rank += 1u;
        }
        stu(2, offset + b, take);
    }
}
