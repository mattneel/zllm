// y[r] = x[r] / sqrt(mean(x[r]²) + eps) * w for rows of `dim`, one group
// per row; with w_rows > 1, row r takes weight row r mod w_rows
// (kernels/rmsnorm.wgsl). Params: rows, dim, eps, w_rows. Bindings: x, w
// (read-only), y.

#define WG 256
groupshared float partial[WG];

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint dim = param(1);
    const float eps = paramf(2);
    const uint w_rows = param(3);
    const uint r = wg.x;
    const uint row = r * dim;
    float sum = 0.0f;
    for (uint i = lid.x; i < dim; i += WG) {
        const float v = ldf(1, row + i);
        sum += v * v;
    }
    partial[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s /= 2) {
        if (lid.x < s) partial[lid.x] += partial[lid.x + s];
        GroupMemoryBarrierWithGroupSync();
    }
    const float scale = 1.0f / sqrt(partial[0] / float(dim) + eps);
    const uint wb = (r % max(w_rows, 1u)) * dim;
    for (uint i = lid.x; i < dim; i += WG) stf(3, row + i, ldf(1, row + i) * scale * ldwf(2, wb + i));
}
