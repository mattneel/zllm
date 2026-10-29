// In place, per head of per-token rows: x ← rmsnorm(x) · w, then NeoX RoPE on
// the first n_rot dims, pairs (i, i + n_rot/2) rotated by the angle of
// position pos0 + t. One workgroup per (head, token).
//
// Head h of token t starts at t · row_stride + h · head_stride, which covers
// both Q (heads interleaved with their output gates) and K. `rope` holds
// (cos, sin) for each (position, pair), computed on the host: WGSL sin/cos
// are only accurate near [−π, π], and these angles reach hundreds of radians.

struct Params {
    heads: u32,
    head_dim: u32,
    n_rot: u32,
    row_stride: u32,
    head_stride: u32,
    pos0: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> rope: array<vec2f>;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let h = wg.x;
    let t = wg.y;
    let base = t * p.row_stride + h * p.head_stride;

    var sum = 0.0;
    for (var i = lid.x; i < p.head_dim; i += WG) {
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
    let scale = 1.0 / sqrt(partial[0] / f32(p.head_dim) + p.eps);
    for (var i = lid.x; i < p.head_dim; i += WG) {
        x[base + i] = x[base + i] * scale * w[i];
    }
    // Rotation pairs cross threads: make the normalized values visible.
    storageBarrier();

    let half = p.n_rot / 2u;
    let row = (p.pos0 + t) * half;
    for (var i = lid.x; i < half; i += WG) {
        let cs = rope[row + i];
        let x0 = x[base + i];
        let x1 = x[base + i + half];
        x[base + i] = x0 * cs.x - x1 * cs.y;
        x[base + i + half] = x0 * cs.y + x1 * cs.x;
    }
}
