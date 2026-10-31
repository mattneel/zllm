// Depthwise causal convolution, SiLU, q/k L2 normalization and DeltaNet gates
// (kernels/delta_prep.wgsl). One workgroup per channel head and token.

#define WG 256
groupshared float partial[WG];

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint channels = param(0);
    const uint taps = param(1);
    const uint key_channels = param(2);
    const uint head = param(3);
    const uint row = param(4);
    const uint v_heads = param(5);
    const uint beta_off = param(6);
    const uint alpha_off = param(7);
    const float eps = paramf(8);
    const uint i = lid3.x;
    const uint c = wg.x;
    const uint t = wg.y;
    const uint groups = channels / head;
    if (c == groups) {
        if (i < v_heads) {
            const uint k = t * v_heads + i;
            stf(8, k, 1.0f / (1.0f + exp(-ldf(1, t * row + beta_off + i))));
            precise float x = ldf(1, t * row + alpha_off + i) + ldwf(5, i);
            precise float sp = x > 20.0f ? x : log(1.0f + exp(x));
            stf(7, k, sp * ldwf(6, i));
        }
        return;
    }
    const uint ch = c * head + i;
    const bool on = i < head;
    precise float y = 0.0f;
    if (on) {
        precise float sum = 0.0f;
        for (uint tap = 0u; tap < taps; tap++) {
            const uint back = taps - 1u - tap;
            float xv;
            if (t >= back) {
                xv = ldf(1, (t - back) * row + ch);
            } else {
                xv = ldf(3, (taps - 1u - (back - t)) * channels + ch);
            }
            sum += ldwf(2, ch * taps + tap) * xv;
        }
        y = sum / (1.0f + exp(-sum));
    }
    if (c * head < key_channels) {
        partial[i] = y * y;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG / 2u; s > 0u; s /= 2u) {
            if (i < s) {
                precise float sum = partial[i] + partial[i + s];
                partial[i] = sum;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        y = y / sqrt(partial[0] + eps);
    }
    if (on) stf(4, t * channels + ch, y);
}
