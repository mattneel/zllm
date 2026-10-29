// In place: x ← x / sqrt(Σ x² + eps) for each of the first `heads` heads of
// `dim` floats in every token row of `row_stride` floats (the DeltaNet q and
// k heads, which lead each conv output row). One workgroup per (head, token).

struct Params {
    heads: u32,
    dim: u32,
    row_stride: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let base = wg.y * p.row_stride + wg.x * p.dim;
    var sum = 0.0;
    for (var i = lid.x; i < p.dim; i += WG) {
        let v = x[base + i];
        sum += v * v;
    }
    partial[lid.x] = sum;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lid.x < s) {
            partial[lid.x] += partial[lid.x + s];
        }
        workgroupBarrier();
    }
    let scale = 1.0 / sqrt(partial[0] + p.eps);
    for (var i = lid.x; i < p.dim; i += WG) {
        x[base + i] *= scale;
    }
}
