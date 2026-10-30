// The sparse attention's indexer, part 3: each step row t's score for every
// complete block before its tail, the sum over query heads of
// max(q_h · pooled_b, 0), into scores[t · stride + b]. Rows that see at most
// `width` cells select everything and score nothing. A thread per block.

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
@group(0) @binding(1) var<storage, read> idx: array<f32>;
@group(0) @binding(2) var<storage, read> pooled: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;
@group(0) @binding(4) var<uniform> step: Step;

const WG: u32 = 256u;
const MAX_Q: u32 = 2048u;

var<workgroup> q: array<f32, MAX_Q>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let t = wg.y;
    let d = p.dim;
    let visible = step.pos0 + t + 1u;
    if (visible <= p.width) {
        return;
    }
    for (var k = lid3.x; k < p.heads * d; k += WG) {
        q[k] = idx[t * p.row + k];
    }
    workgroupBarrier();
    let b = wg.x * WG + lid3.x;
    if (b >= visible / p.ratio) {
        return;
    }
    var s = 0.0;
    for (var h = 0u; h < p.heads; h++) {
        var dot = 0.0;
        for (var i = 0u; i < d; i++) {
            dot += q[h * d + i] * pooled[b * d + i];
        }
        s += max(dot, 0.0);
    }
    scores[t * p.stride + b] = s;
}
