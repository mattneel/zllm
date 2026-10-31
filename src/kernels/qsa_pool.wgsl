// The sparse attention's indexer, part 2: pools each block of `ratio`
// cached keys this step completes (blocks b with (b + 1) · ratio ≤ pos0 + m).
// The mean of its keys, in position order, is RMSNormed with k_norm and
// rotated (NeoX RoPE) at its first
// position, b · ratio. Workgroup x takes block pos0 / ratio + x.

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
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> k_norm: array<f32>;
@group(0) @binding(3) var<storage, read> rope: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> pooled: array<f32>;
@group(0) @binding(5) var<uniform> step: Step;

const WG: u32 = 256u;

var<workgroup> partial: array<f32, WG>;
var<workgroup> vals: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let i = lid3.x;
    let r = p.ratio;
    let d = p.dim;
    let b = step.pos0 / r + wg.x;
    if ((b + 1u) * r > step.pos0 + step.m) {
        return;
    }
    var v = 0.0;
    if (i < d) {
        for (var j = 0u; j < r; j++) {
            let pair = unpack2x16float(keys[((b * r + j) * d + i) / 2u]);
            v += select(pair.x, pair.y, (i & 1u) == 1u);
        }
        v = v / f32(r);
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
        vals[i] = v * scale * k_norm[i];
    }
    workgroupBarrier();
    let half = p.n_rot / 2u;
    if (i < half) {
        let cs = rope[b * r * half + i];
        let x0 = vals[i];
        let x1 = vals[i + half];
        vals[i] = x0 * cs.x - x1 * cs.y;
        vals[i + half] = x0 * cs.y + x1 * cs.x;
    }
    workgroupBarrier();
    if (i < d) {
        pooled[b * d + i] = vals[i];
    }
}
