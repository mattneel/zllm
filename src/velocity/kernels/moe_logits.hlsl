// Router and shared-gate logits (kernels/moe_logits.wgsl): group (e, t) takes
// row e of token t, its threads striding over the inputs four at a time; wave
// sums, then the eight waves' sums in order. Waves are 32. Bindings: router,
// shared_gate (read-only), x, logits.

#define WG 256

groupshared float wsum[WG / 32];

[WaveSize(32)]
[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint n_expert = param(0);
    const uint c4 = param(2) / 4u;
    const uint e = wg.x;
    const uint t = wg.y;
    const uint xb = base(3) + 16u * t * (param(3) / 4u);
    precise float acc = 0.0f;
    if (e < n_expert) {
        const uint wb = base(1) + 16u * e * c4;
        for (uint i = li; i < c4; i += WG) acc += dot(asfloat(ro(1).Load4(wb + 16u * i)), asfloat(rw(3).Load4(xb + 16u * i)));
    } else {
        for (uint i = li; i < c4; i += WG) acc += dot(asfloat(ro(2).Load4(base(2) + 16u * i)), asfloat(rw(3).Load4(xb + 16u * i)));
    }
    const float s = WaveActiveSum(acc);
    if (WaveIsFirstLane()) wsum[li / 32u] = s;
    GroupMemoryBarrierWithGroupSync();
    if (li == 0u) {
        precise float total = 0.0f;
        for (uint w = 0u; w < WG / 32u; w++) total += wsum[w];
        stf(4, t * (n_expert + 1u) + e, total);
    }
}
