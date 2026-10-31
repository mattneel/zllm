// Expert routing, part 1: the router's logits, router · x (f32 weights,
// n_expert rows), then the shared expert's gate logit, shared_gate · x, as
// row n_expert: logits[t · (n_expert + 1) + e]. Workgroup (e, t) takes row
// e of token t: its 256 threads stride over the inputs four at a time, then
// reduce in a fixed tree. cols is a multiple of 4.

struct Params {
    n_expert: u32,
    n_used: u32,
    cols: u32,
    x_stride: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> router: array<vec4f>;
@group(0) @binding(2) var<storage, read> shared_gate: array<vec4f>;
@group(0) @binding(3) var<storage, read> x: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;

const WG: u32 = 256u;

var<workgroup> red: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let e = wg.x;
    let t = wg.y;
    let c4 = p.cols / 4u;
    let xb = t * (p.x_stride / 4u);
    var acc = 0.0;
    if (e < p.n_expert) {
        for (var i = li; i < c4; i += WG) {
            acc += dot(router[e * c4 + i], x[xb + i]);
        }
    } else {
        for (var i = li; i < c4; i += WG) {
            acc += dot(shared_gate[i], x[xb + i]);
        }
    }
    red[li] = acc;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (li < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        logits[t * (p.n_expert + 1u) + e] = red[0];
    }
}
