// Hyper-connection low-rank activation: lo = silu(down / hc), skipping
// the injection columns at each row's end (kernels/hc_act.wgsl).

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= param(0)) return;
    const uint rank = param(2);
    const uint t = i / rank;
    const float v = ldf(1, t * param(3) + i % rank) / float(param(1));
    precise float result = v / (1.0f + exp(-v));
    stf(2, i, result);
}
