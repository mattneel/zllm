// y[r] = x[r] / sqrt(mean(x[r]²) + eps) * w for rows of `dim`, one
// workgroup per row. With w_rows > 1, row r takes weight row r mod w_rows
// (the hyper-connection streams' own norms); 0 or 1: one weight row.

struct Params {
    rows: u32,
    dim: u32,
    eps: f32,
    w_rows: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const WG: u32 = 256u;
var<workgroup> partial: array<vec2f, WG>;
var<workgroup> row_scale: f32;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let r = wg.x;
    let base = r * p.dim;
    var sum = vec2f(0.0);
    for (var i = lid.x; i < p.dim; i += WG) {
        let v = x[base + i];
        let square = v * v;
        sum = norm_add(sum, square);
    }
    partial[lid.x] = sum;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lid.x < s) {
            partial[lid.x] = norm_merge(partial[lid.x], partial[lid.x + s]);
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        row_scale = norm_scale(partial[0], p.dim, p.eps);
    }
    workgroupBarrier();
    let wb = (r % max(p.w_rows, 1u)) * p.dim;
    for (var i = lid.x; i < p.dim; i += WG) {
        y[base + i] = x[base + i] * row_scale * w[wb + i];
    }
}
