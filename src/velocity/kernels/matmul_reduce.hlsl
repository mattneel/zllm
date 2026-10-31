// Sum split-k partial products, optionally adding the residual
// (kernels/matmul_reduce.wgsl).
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const uint i = id.x + id.y * num_groups().x * 256u;
    const uint rows = param(0), n = param(1), slices = param(2);
    if (i >= rows * n) return;
    const uint t = i / n, j = i % n;
    precise float sum = 0.0f;
    for (uint s = 0u; s < slices; s++) sum += ldf(1, s * param(6) + i);
    const uint o = t * param(4) + param(3) + j;
    if (param(5) != 0u) stf(2, o, ldf(2, o) + sum);
    else stf(2, o, sum);
}
