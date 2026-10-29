// Samples one token from a row of logits with vLLM's semantics and order:
//
//   1. logit bias: l[tok] += bias
//   2. penalties: repetition r over prompt ∪ output tokens (l > 0 ? l / r :
//      l · r), then l −= frequency · count + presence · [count > 0] over
//      output tokens
//   3. greedy: if `greedy`, the first index of the maximum; done
//   4. temperature: l /= T
//   5. min-p: drop tokens with exp(l − max) < min_p, i.e. p < min_p · p_max
//   6. top-k: keep tokens ≥ the k-th largest remaining value
//   7. top-p: keep the largest tokens whose mass reaches p (a token stays
//      when the mass ranked above it is < p)
//   8. sample by inverse CDF in token-id order at u ∈ [0, 1) from the step
//
// Every filter keeps a top segment of the values, so the kept set is always
// {l : key(l) ≥ tau} intersected with the min-p test, where key() maps f32 to
// an order-preserving u32. Top-k and top-p find their tau by bisection over
// keys, so nothing is sorted; ties at a threshold are all kept.
//
// The token goes to out[0] (out[1] = 1 if it is a stop token) and to
// next[0], the embedding input of the next step; its output count is bumped.
// One workgroup; `work` is scratch for the processed logits.

struct Params {
    vocab: u32,
    n_bias: u32,
    n_stop: u32,
    greedy: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    repetition: f32,
    presence: f32,
    frequency: f32,
    n_prompt: u32,
    stops: array<vec4u, 4>,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
// (token, f32 bits of the bias); tokens are distinct.
@group(0) @binding(2) var<storage, read> bias: array<vec2u>;
@group(0) @binding(3) var<storage, read> prompt_mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> work: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<u32>;
@group(0) @binding(7) var<storage, read_write> next: array<u32>;
@group(0) @binding(8) var<uniform> step: Step;

const WG: u32 = 256u;
const NONE: u32 = 0xffffffffu;
const LOWEST: f32 = -0x1.fffffep+127f; // -FLT_MAX

var<workgroup> rf: array<f32, WG>;
var<workgroup> ru: array<u32, WG>;
var<workgroup> chosen: u32;

fn key(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return select(b ^ 0x80000000u, ~b, (b >> 31u) == 1u);
}

fn sum_f(lid: u32, x: f32) -> f32 {
    rf[lid] = x;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            rf[lid] += rf[lid + s];
        }
        workgroupBarrier();
    }
    let r = workgroupUniformLoad(&rf[0]);
    workgroupBarrier();
    return r;
}

fn sum_u(lid: u32, x: u32) -> u32 {
    ru[lid] = x;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            ru[lid] += ru[lid + s];
        }
        workgroupBarrier();
    }
    let r = workgroupUniformLoad(&ru[0]);
    workgroupBarrier();
    return r;
}

fn max_u(lid: u32, x: u32) -> u32 {
    ru[lid] = x;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            ru[lid] = max(ru[lid], ru[lid + s]);
        }
        workgroupBarrier();
    }
    let r = workgroupUniformLoad(&ru[0]);
    workgroupBarrier();
    return r;
}

// Index of the maximum (first on ties); its value is left in rf[0].
fn argmax(lid: u32, v: f32, i: u32) -> u32 {
    rf[lid] = v;
    ru[lid] = i;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            let ov = rf[lid + s];
            let oi = ru[lid + s];
            if (ov > rf[lid] || (ov == rf[lid] && oi < ru[lid])) {
                rf[lid] = ov;
                ru[lid] = oi;
            }
        }
        workgroupBarrier();
    }
    let r = workgroupUniformLoad(&ru[0]);
    workgroupBarrier();
    return r;
}

// Whether value v survives min-p and the threshold tau.
fn kept(v: f32, top: f32, tau: u32) -> bool {
    return key(v) >= tau && (p.min_p <= 0.0 || exp(v - top) >= p.min_p);
}

fn count_ge(lid: u32, top: f32, tau: u32) -> u32 {
    var n = 0u;
    for (var i = lid; i < p.vocab; i += WG) {
        n += select(0u, 1u, kept(work[i], top, tau));
    }
    return sum_u(lid, n);
}

fn mass_ge(lid: u32, top: f32, tau: u32) -> f32 {
    var s = 0.0;
    for (var i = lid; i < p.vocab; i += WG) {
        let v = work[i];
        if (kept(v, top, tau)) {
            s += exp(v - top);
        }
    }
    return sum_f(lid, s);
}

fn finish(token: u32) {
    var stop = 0u;
    for (var i = 0u; i < p.n_stop; i++) {
        if (p.stops[i / 4u][i % 4u] == token) {
            stop = 1u;
        }
    }
    out[0] = token;
    out[1] = stop;
    next[0] = token;
    counts[token] += 1u;
}

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid3: vec3u) {
    let lid = lid3.x;

    for (var i = lid; i < p.vocab; i += WG) {
        work[i] = logits[i];
    }
    storageBarrier();
    for (var e = lid; e < p.n_bias; e += WG) {
        let b = bias[e];
        work[b.x] += bitcast<f32>(b.y);
    }
    storageBarrier();

    var best = LOWEST;
    var best_i = NONE;
    for (var i = lid; i < p.vocab; i += WG) {
        var v = work[i];
        let n = counts[i];
        if (prompt_mask[i] != 0u || n != 0u) {
            v = select(v * p.repetition, v / p.repetition, v > 0.0);
        }
        v -= p.frequency * f32(n);
        v -= p.presence * select(0.0, 1.0, n != 0u);
        work[i] = v;
        if (v > best) {
            best = v;
            best_i = i;
        }
    }
    let top_i = argmax(lid, best, best_i);
    if (p.greedy != 0u) {
        if (lid == 0u) {
            finish(top_i);
        }
        return;
    }

    let top = workgroupUniformLoad(&rf[0]) / p.temperature;
    for (var i = lid; i < p.vocab; i += WG) {
        work[i] = work[i] / p.temperature;
    }
    storageBarrier();

    // top-k: the largest tau keeping at least k tokens.
    var tau = 0u;
    if (p.top_k > 0u && count_ge(lid, top, 0u) > p.top_k) {
        var lo = 0u;
        var hi = key(top) + 1u;
        loop {
            if (hi - lo <= 1u) {
                break;
            }
            let mid = lo + (hi - lo) / 2u;
            if (count_ge(lid, top, mid) >= p.top_k) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        tau = lo;
    }

    // top-p: the largest tau keeping at least p of the remaining mass.
    if (p.top_p < 1.0) {
        let goal = p.top_p * mass_ge(lid, top, tau);
        var lo = tau;
        var hi = key(top) + 1u;
        loop {
            if (hi - lo <= 1u) {
                break;
            }
            let mid = lo + (hi - lo) / 2u;
            if (mass_ge(lid, top, mid) >= goal) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        tau = lo;
    }

    // Inverse CDF in token order: thread lid owns one contiguous range.
    let span = (p.vocab + WG - 1u) / WG;
    let i0 = min(lid * span, p.vocab);
    let i1 = min(i0 + span, p.vocab);
    var s = 0.0;
    var last = 0u; // 1 + the last kept index in the range, 0 if none
    for (var i = i0; i < i1; i++) {
        let v = work[i];
        if (kept(v, top, tau)) {
            s += exp(v - top);
            last = i + 1u;
        }
    }
    let last_any = max_u(lid, last);
    rf[lid] = s;
    if (lid == 0u) {
        chosen = NONE;
    }
    workgroupBarrier();
    if (lid == 0u) {
        // Exclusive prefix sums, in place, then the total in ru[0] bits.
        var acc = 0.0;
        for (var t = 0u; t < WG; t++) {
            let x = rf[t];
            rf[t] = acc;
            acc += x;
        }
        ru[0] = bitcast<u32>(acc);
    }
    workgroupBarrier();
    let at = step.u * bitcast<f32>(ru[0]);
    let lo = rf[lid];
    if (s > 0.0 && lo <= at && at < lo + s) {
        var c = lo;
        var pick = last - 1u;
        for (var i = i0; i < i1; i++) {
            let v = work[i];
            if (kept(v, top, tau)) {
                c += exp(v - top);
                if (c > at) {
                    pick = i;
                    break;
                }
            }
        }
        chosen = pick;
    }
    workgroupBarrier();
    if (lid == 0u) {
        // Rounding can leave the draw at or past the total: take the last kept token.
        finish(select(chosen, last_any - 1u, chosen == NONE));
    }
}
