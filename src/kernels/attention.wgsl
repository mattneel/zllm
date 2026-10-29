// Causal grouped-query attention against the KV cache, with a sigmoid output
// gate. The step's m query tokens sit at positions pos0 .. pos0 + m − 1 and
// their keys and values are already in the cache (kv_append). One workgroup
// per (query head h, query token t), attending to cache positions 0 ..= pos0 + t:
//
//   s_j = q · k_j · scale
//   o   = Σ_j softmax(s)_j · v_j
//   out = o · sigmoid(gate)
//
// Keys are visited in tiles of WG with an online softmax, so the context
// length is unbounded. Every query walks the same tiles from position 0
// whatever m is, so the result does not depend on how the sequence was split
// into steps.
//
// q and its gate come from the joint projection: head h of token t has q at
// t·2·heads·head_dim + 2h·head_dim and the gate right after it. Cache rows hold
// kv_heads heads of head_dim; query head h reads kv head h / (heads / kv_heads).
// The cache stores f16 pairs (pack2x16float) when f16 != 0, else f32 bits.
// head_dim ≤ WG and even.

struct Params {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    f16: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qg: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<u32>;
@group(0) @binding(3) var<storage, read> vc: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> step: Step;

const WG: u32 = 256u;
const MAX_HEAD_DIM: u32 = 256u;

var<workgroup> qs: array<f32, MAX_HEAD_DIM>;
var<workgroup> ps: array<f32, WG>;
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

// Elements (2i, 2i + 1) of a cache.
fn k_pair(i: u32) -> vec2f {
    if (p.f16 != 0u) {
        return unpack2x16float(kc[i]);
    }
    return vec2f(bitcast<f32>(kc[2u * i]), bitcast<f32>(kc[2u * i + 1u]));
}

fn v_at(i: u32) -> f32 {
    if (p.f16 != 0u) {
        let pair = unpack2x16float(vc[i / 2u]);
        return select(pair.x, pair.y, (i & 1u) == 1u);
    }
    return bitcast<f32>(vc[i]);
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
    let keys = step.pos0 + t + 1u;

    for (var d = lid; d < hd; d += WG) {
        qs[d] = qg[q_base + d];
    }
    workgroupBarrier();

    var m_run = 0.0;
    var l_run = 0.0;
    var acc = 0.0;
    for (var j0 = 0u; j0 < keys; j0 += WG) {
        let j = j0 + lid;
        var s = -0x1.fffffep+127f; // -FLT_MAX
        if (j < keys) {
            let kb = (j * kv_stride + kvh * hd) / 2u;
            var dot = 0.0;
            for (var d = 0u; d < hd / 2u; d++) {
                let kk = k_pair(kb + d);
                dot += qs[2u * d] * kk.x + qs[2u * d + 1u] * kk.y;
            }
            s = dot * p.scale;
        }
        let tile_max = reduce_max(lid, s);
        let first = j0 == 0u;
        let m_new = select(max(m_run, tile_max), tile_max, first);
        let alpha = select(exp(m_run - m_new), 0.0, first);
        let e = select(0.0, exp(s - m_new), j < keys);
        ps[lid] = e;
        l_run = l_run * alpha + reduce_sum(lid, e);
        if (lid < hd) {
            let n = min(WG, keys - j0);
            var o = 0.0;
            for (var jj = 0u; jj < n; jj++) {
                o += ps[jj] * v_at((j0 + jj) * kv_stride + kvh * hd + lid);
            }
            acc = acc * alpha + o;
        }
        m_run = m_new;
        workgroupBarrier();
    }

    if (lid < hd) {
        let gate = qg[q_base + hd + lid];
        out[(t * p.heads + h) * hd + lid] = acc / l_run / (1.0 + exp(-gate));
    }
}
