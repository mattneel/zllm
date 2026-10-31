// Signed 128-wide Hadamard rotation, optionally after silu(gate) * up
// (kernels/rotate128.wgsl). Bindings: x, signs table, y.

groupshared float buf[128];

[numthreads(64, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint col0 = wg.x * 128u;
    const uint xb = wg.y * param(1);
    for (uint i = li; i < 128u; i += 64u) {
        const uint c = col0 + i;
        precise float v = ldf(1, xb + c);
        if (param(3) != 0u) {
            v = v / (1.0f + exp(-v)) * ldf(1, xb + param(0) + c);
        }
        buf[i] = v * ldwf(2, c);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint width = 1u; width < 128u; width *= 2u) {
        const uint lo = (li / width) * (2u * width) + (li % width);
        const float a = buf[lo];
        const float b = buf[lo + width];
        GroupMemoryBarrierWithGroupSync();
        buf[lo] = a + b;
        buf[lo + width] = a - b;
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint i = li; i < 128u; i += 64u) {
        stf(3, wg.y * param(2) + col0 + i, buf[i] * 0.08838834764831845f);
    }
}
