// Causal grouped-query attention with a sigmoid output gate, for a batch of
// tokens attending to themselves (positions 0..tokens). One workgroup per
// (query head, query token t):
//
//   s_j = q · k_j · scale            for j ≤ t
//   o   = Σ_j softmax(s)_j · v_j
//   out = o · sigmoid(gate)
//
// q and its gate come from the joint projection: head h of token t has q at
// t·q_stride + 2h·head_dim and the gate right after it. K and V rows hold
// kv_heads heads; query head h reads kv head h / (heads / kv_heads).
// Scores live in workgroup memory, so tokens ≤ MAX_KEYS.

struct Params {
    tokens: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qg: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const WG: u32 = 256u;
const MAX_KEYS: u32 = 4096u;
const MAX_HEAD_DIM: u32 = 256u;

var<workgroup> scores: array<f32, MAX_KEYS>;
var<workgroup> qs: array<f32, MAX_HEAD_DIM>;
var<workgroup> partial: array<f32, WG>;

fn reduce_max(lid: u32, x: f32) -> f32 {
    partial[lid] = x;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            partial[lid] = max(partial[lid], partial[lid + s]);
        }
        workgroupBarrier();
    }
    let r = partial[0];
    workgroupBarrier();
    return r;
}

fn reduce_sum(lid: u32, x: f32) -> f32 {
    partial[lid] = x;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            partial[lid] += partial[lid + s];
        }
        workgroupBarrier();
    }
    let r = partial[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let lid = lid3.x;
    let h = wg.x;
    let t = wg.y;
    let hd = p.head_dim;
    let kvh = h / (p.heads / p.kv_heads);
    let q_base = t * (2u * p.heads * hd) + 2u * h * hd;
    let kv_stride = p.kv_heads * hd;

    for (var d = lid; d < hd; d += WG) {
        qs[d] = qg[q_base + d];
    }
    workgroupBarrier();

    var m = -0x1.fffffep+127f; // -FLT_MAX
    for (var j = lid; j <= t; j += WG) {
        let kb = j * kv_stride + kvh * hd;
        var s = 0.0;
        for (var d = 0u; d < hd; d++) {
            s += qs[d] * k[kb + d];
        }
        s *= p.scale;
        scores[j] = s;
        m = max(m, s);
    }
    m = reduce_max(lid, m);

    var sum = 0.0;
    for (var j = lid; j <= t; j += WG) {
        let e = exp(scores[j] - m);
        scores[j] = e;
        sum += e;
    }
    sum = reduce_sum(lid, sum);

    for (var d = lid; d < hd; d += WG) {
        var o = 0.0;
        for (var j = 0u; j <= t; j++) {
            o += scores[j] * v[j * kv_stride + kvh * hd + d];
        }
        let gate = qg[q_base + hd + d];
        out[(t * p.heads + h) * hd + d] = o / sum / (1.0 + exp(-gate));
    }
}
