// y[r] = x[r] / sqrt(mean(x[r]²) + eps) * w for rows of `dim`, one
// workgroup per row.

struct Params {
    rows: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let r = wg.x;
    let base = r * p.dim;
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
    let scale = 1.0 / sqrt(partial[0] / f32(p.dim) + p.eps);
    for (var i = lid.x; i < p.dim; i += WG) {
        y[base + i] = x[base + i] * scale * w[i];
    }
}
