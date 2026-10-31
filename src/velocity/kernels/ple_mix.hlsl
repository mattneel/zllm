// PLE key/query RMS-normalized gate, gated values and convolution-input
// normalization, one token stream per group (kernels/ple_mix.wgsl).

#define WG 256
groupshared float3 red[WG];

float3 total(float3 v, uint li) {
    red[li] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s /= 2u) {
        if (li < s) red[li] += red[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float3 r = red[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint c = wg.x;
    const uint t = wg.y;
    const uint e = param(0);
    const float eps = paramf(2);
    const uint seg = (t * param(1) + c) * e;
    precise float3 sq = float3(0.0f, 0.0f, 0.0f);
    for (uint i = li; i < e; i += WG) {
        const float k = ldf(1, seg + i);
        const float h = ldf(2, seg + i);
        const float v = ldf(3, t * e + i);
        sq += float3(k * k, h * h, v * v);
    }
    const float3 sums = total(sq, li);
    precise float sk = 1.0f / sqrt(sums.x / float(e) + eps);
    precise float sh = 1.0f / sqrt(sums.y / float(e) + eps);
    precise float d = 0.0f;
    for (uint i = li; i < e; i += WG) {
        d += ldf(1, seg + i) * sk * ldwf(4, c * e + i) * ldf(2, seg + i) * sh * ldwf(5, c * e + i);
    }
    precise float s = total(float3(d, 0.0f, 0.0f), li).x / sqrt(float(e));
    precise float gate = 1.0f / (1.0f + exp(-sign(s) * sqrt(max(abs(s), 1e-6f))));
    precise float sg = 1.0f / sqrt(gate * gate * sums.z / float(e) + eps);
    for (uint i = li; i < e; i += WG) {
        precise float g = ldf(3, t * e + i) * gate;
        stf(7, seg + i, g);
        precise float normed = g * sg * ldwf(6, c * e + i);
        stf(8, seg + i, normed);
    }
}
