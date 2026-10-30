// The PLE block's gate, per stream c of token t (lifted from gyro-dx's
// ple_mix): k = rmsnorm(key[t][c]) · wk[c], q = rmsnorm(hidden[t][c]) ·
// wq[c], s = k · q / √E, gate = sigmoid(sign(s) · √max(|s|, 1e-6)); then
// gated[t][c] = value[t] · gate and normed[t][c] = rmsnorm(gated) · wc[c]
// (its RMS is |gate| · rms(value)), the convolution's input.
// Grid: x = streams, y = tokens.

struct Params {
    dim: u32,
    hc: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> hidden: array<f32>;
@group(0) @binding(3) var<storage, read> value: array<f32>;
// The norm weights: key, query, conv, each hc · dim.
@group(0) @binding(4) var<storage, read> wk: array<f32>;
@group(0) @binding(5) var<storage, read> wq: array<f32>;
@group(0) @binding(6) var<storage, read> wc: array<f32>;
@group(0) @binding(7) var<storage, read_write> gated: array<f32>;
@group(0) @binding(8) var<storage, read_write> normed: array<f32>;

const WG: u32 = 256u;
var<workgroup> red: array<vec3f, WG>;

fn total(v: vec3f, li: u32) -> vec3f {
    red[li] = v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (li < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let c = wg.x;
    let t = wg.y;
    let e = p.dim;
    let seg = (t * p.hc + c) * e;
    var sq = vec3f(0.0);
    for (var i = li; i < e; i += WG) {
        let k = key[seg + i];
        let h = hidden[seg + i];
        let v = value[t * e + i];
        sq += vec3f(k * k, h * h, v * v);
    }
    let sums = total(sq, li);
    let sk = 1.0 / sqrt(sums.x / f32(e) + p.eps);
    let sh = 1.0 / sqrt(sums.y / f32(e) + p.eps);
    var d = 0.0;
    for (var i = li; i < e; i += WG) {
        d += key[seg + i] * sk * wk[c * e + i] * hidden[seg + i] * sh * wq[c * e + i];
    }
    let s = total(vec3f(d, 0.0, 0.0), li).x / sqrt(f32(e));
    let gate = 1.0 / (1.0 + exp(-sign(s) * sqrt(max(abs(s), 1e-6))));
    let sg = 1.0 / sqrt(gate * gate * sums.z / f32(e) + p.eps);
    for (var i = li; i < e; i += WG) {
        let g = value[t * e + i] * gate;
        gated[seg + i] = g;
        normed[seg + i] = g * sg * wc[c * e + i];
    }
}
