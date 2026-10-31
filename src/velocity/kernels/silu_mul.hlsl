// act[t][j] = silu(gate[t][j]) * up[t][j] (kernels/silu_mul.wgsl).
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x + id.y * num_groups().x * 256u;
    if (i >= param(0)) return;
    const uint width = param(1), t = i / width, j = i % width;
    const float g = ldf(1, t * 2u * width + j);
    precise float v = g / (1.0f + exp(-g)) * ldf(1, t * 2u * width + width + j);
    stf(2, i, v);
}
