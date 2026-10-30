// Expert routing, part 2, one token per workgroup (lifted from gyro-dx's
// moe_route): over the logits moe_logits computed, softmax, the n_used
// largest (ties to the lower index), their probabilities normalized by their
// sum (clamped below at 2^-14) and scaled by `scale` unless it is 0 or 1.
// Also passes the shared expert's gate logit on to moe_combine.

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
// [token][n_expert + 1] (moe_logits).
@group(0) @binding(1) var<storage, read> logits: array<f32>;
// [token][n_used] expert ids and weights; [token] the shared gate logit.
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> shared_logit: array<f32>;

const WG: u32 = 256u;
const MAX_EXPERT: u32 = 512u;

var<workgroup> probs: array<f32, MAX_EXPERT>;
var<workgroup> red: array<f32, WG>;
var<workgroup> redi: array<u32, WG>;
var<workgroup> chosen: array<f32, 16>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let t = wg.x;
    let row = t * (p.n_expert + 1u);
    for (var e = li; e < p.n_expert; e += WG) {
        probs[e] = logits[row + e];
    }
    if (li == 0u) {
        shared_logit[t] = logits[row + p.n_expert];
    }
    workgroupBarrier();

    // Softmax: the maximum, then the sum of exponentials.
    var m = -3.0e38;
    for (var e = li; e < p.n_expert; e += WG) {
        m = max(m, probs[e]);
    }
    red[li] = m;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (li < s) {
            red[li] = max(red[li], red[li + s]);
        }
        workgroupBarrier();
    }
    m = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var e = li; e < p.n_expert; e += WG) {
        let v = exp(probs[e] - m);
        probs[e] = v;
        sum += v;
    }
    red[li] = sum;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (li < s) {
            red[li] += red[li + s];
        }
        workgroupBarrier();
    }
    sum = red[0];
    workgroupBarrier();
    for (var e = li; e < p.n_expert; e += WG) {
        probs[e] = probs[e] / sum;
    }
    workgroupBarrier();

    // Top n_used by repeated argmax, the lower index on ties.
    var total = 0.0;
    for (var k = 0u; k < p.n_used; k++) {
        var best = -1.0;
        var best_i = 0xffffffffu;
        for (var e = li; e < p.n_expert; e += WG) {
            if (probs[e] > best) {
                best = probs[e];
                best_i = e;
            }
        }
        red[li] = best;
        redi[li] = best_i;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s /= 2u) {
            if (li < s) {
                let ov = red[li + s];
                let oi = redi[li + s];
                if (ov > red[li] || (ov == red[li] && oi < redi[li])) {
                    red[li] = ov;
                    redi[li] = oi;
                }
            }
            workgroupBarrier();
        }
        let gb = red[0];
        let gi = redi[0];
        total += gb;
        if (li == 0u) {
            ids[t * p.n_used + k] = gi;
            chosen[k] = gb;
            probs[gi] = -1.0;
        }
        workgroupBarrier();
    }
    if (li < p.n_used) {
        var w = chosen[li] / max(total, 6.103515625e-5);
        if (p.scale != 0.0 && p.scale != 1.0) {
            w *= p.scale;
        }
        weights[t * p.n_used + li] = w;
    }
}
