// Merge block softmax partials in block order, optionally applying the
// sigmoid output gate (kernels/attn_combine.wgsl). One thread per dimension.
#define WG 256
#define BLOCK 64

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint d = lid3.x;
    const uint h = wg.x;
    const uint t = wg.y;
    const uint row = param(7) + t;
    const uint hd = param(2);
    if (d >= hd) return;
    const uint keys = ldu(4, 1) + row + 1u;
    const uint blocks = (keys + BLOCK - 1u) / BLOCK;
    const uint base0 = (t * param(0) + h) * param(5) * (hd + 2u);

    precise float mx = -asfloat(0x7f7fffffu);
    for (uint b = 0u; b < blocks; b++) mx = max(mx, ldf(2, base0 + b * (hd + 2u)));
    precise float l = 0.0f;
    precise float o = 0.0f;
    for (uint b = 0u; b < blocks; b++) {
        const uint offset = base0 + b * (hd + 2u);
        precise float w = exp(ldf(2, offset) - mx);
        l += ldf(2, offset + 1u) * w;
        o += ldf(2, offset + 2u + d) * w;
    }
    if (param(8) == 0u) {
        precise float result = o / l;
        stf(3, (row * param(0) + h) * hd + d, result);
        return;
    }
    const float gate = ldf(1, row * param(6) + 2u * h * hd + hd + d);
    precise float result = o / l / (1.0f + exp(-gate));
    stf(3, (row * param(0) + h) * hd + d, result);
}
