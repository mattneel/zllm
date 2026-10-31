// y[r] = x[r] / sqrt(mean(x[r]²) + eps) * w for rows of `dim`, one group
// per row; with w_rows > 1, row r takes weight row r mod w_rows
// (kernels/rmsnorm.wgsl). Params: rows, dim, eps, w_rows. Bindings: x, w
// (read-only), y.

#define WG 256
groupshared float2 partial[WG];
groupshared float row_scale;

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint dim = param(1);
    const float eps = paramf(2);
    const uint w_rows = param(3);
    const uint r = wg.x;
    const uint row = r * dim;
    float2 sum = 0.0f;
    for (uint i = lid.x; i < dim; i += WG) {
        const float v = ldf(1, row + i);
        precise float square = v * v;
        sum = norm_add(sum, square);
    }
    partial[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s /= 2) {
        if (lid.x < s) partial[lid.x] = norm_merge(partial[lid.x], partial[lid.x + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid.x == 0u) row_scale = norm_scale(partial[0], dim, eps);
    GroupMemoryBarrierWithGroupSync();
    const uint wb = (r % max(w_rows, 1u)) * dim;
    for (uint i = lid.x; i < dim; i += WG) {
        precise float value = ldf(1, row + i) * row_scale * ldwf(2, wb + i);
        stf(3, row + i, value);
    }
}
