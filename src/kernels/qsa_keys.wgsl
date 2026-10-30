// The sparse attention's indexer, part 1, on each step row t's indexer
// projection (`idx` row t: `heads` query heads of `dim`, then the key):
// workgroup x < heads RMSNorms query head x with q_norm and applies NeoX RoPE
// at position pos0 + t, in place; workgroup x = heads stores the raw key in
// the cache as f16 pairs, rounded to nearest (f16_round.wgsl), as the
// reference's cache holds it.

struct Qsa {
    heads: u32,
    dim: u32,
    ratio: u32,
    width: u32,
    stride: u32,
    n_rot: u32,
    eps: f32,
    row: u32,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Qsa;
@group(0) @binding(1) var<storage, read_write> idx: array<f32>;
@group(0) @binding(2) var<storage, read> q_norm: array<f32>;
@group(0) @binding(3) var<storage, read> rope: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> keys: array<u32>;
@group(0) @binding(5) var<uniform> step: Step;

const WG: u32 = 256u;

var<workgroup> partial: array<f32, WG>;
var<workgroup> vals: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let i = lid3.x;
    let x = wg.x;
    let t = wg.y;
    let d = p.dim;
    let pos = step.pos0 + t;
    let base = t * p.row + x * d;
    if (x == p.heads) {
        if (i < d / 2u) {
            keys[pos * (d / 2u) + i] = pack_f16(vec2f(idx[base + 2u * i], idx[base + 2u * i + 1u]));
        }
        return;
    }
    var v = 0.0;
    if (i < d) {
        v = idx[base + i];
    }
    partial[i] = v * v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (i < s) {
            partial[i] += partial[i + s];
        }
        workgroupBarrier();
    }
    let scale = 1.0 / sqrt(partial[0] / f32(d) + p.eps);
    if (i < d) {
        vals[i] = v * scale * q_norm[i];
    }
    workgroupBarrier();
    let half = p.n_rot / 2u;
    if (i < half) {
        let cs = rope[pos * half + i];
        let x0 = vals[i];
        let x1 = vals[i + half];
        vals[i] = x0 * cs.x - x1 * cs.y;
        vals[i + half] = x0 * cs.y + x1 * cs.x;
    }
    workgroupBarrier();
    if (i < d) {
        idx[base + i] = vals[i];
    }
}
