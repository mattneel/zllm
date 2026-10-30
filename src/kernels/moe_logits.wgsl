// Expert routing, part 1: the router's logits, router · x (f32 weights,
// n_expert rows), then the shared expert's gate logit, shared_gate · x, as
// row n_expert: logits[t · (n_expert + 1) + e]. Each of a workgroup's 8 lane
// groups takes one row, its 32 lanes striding over the inputs, then reduces
// them in a fixed tree; workgroup (x, t) takes rows 8x.. of token t.

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
@group(0) @binding(1) var<storage, read> router: array<f32>;
@group(0) @binding(2) var<storage, read> shared_gate: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> logits: array<f32>;

const WG: u32 = 256u;
const LANES: u32 = 32u;
const GROUPS: u32 = WG / LANES;
const MAX_COLS: u32 = 2560u;

var<workgroup> xs: array<f32, MAX_COLS>;
var<workgroup> red: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    let lane = li % LANES;
    let group = li / LANES;
    for (var i = li; i < p.cols; i += WG) {
        xs[i] = x[t * p.x_stride + i];
    }
    workgroupBarrier();

    let e = wg.x * GROUPS + group;
    var acc = 0.0;
    if (e < p.n_expert) {
        for (var i = lane; i < p.cols; i += LANES) {
            acc += router[e * p.cols + i] * xs[i];
        }
    } else if (e == p.n_expert) {
        for (var i = lane; i < p.cols; i += LANES) {
            acc += shared_gate[i] * xs[i];
        }
    }
    red[li] = acc;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s /= 2u) {
        if (lane < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }
    if (lane == 0u && e <= p.n_expert) {
        logits[t * (p.n_expert + 1u) + e] = red[li];
    }
}
