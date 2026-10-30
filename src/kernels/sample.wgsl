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
// an order-preserving u32; ties at a threshold are all kept.
//
// The work is split into phases, one dispatch each, selected by `ph.phase`
// (see kernels.sampleSteps for the sequences). Phases over the vocabulary
// run one workgroup per SPAN tokens; the rest run one workgroup. Values pass
// between phases in `st`:
//
//   prep       bias, penalties and temperature into `work`; each
//              workgroup's maximum
//   reduce     the maximum and its first index; greedy finishes here
//   hist/digit top-k by radix select: four rounds of 8-bit digit histograms
//              of the keys, then the digit holding the k-th largest
//   gather     the kept tokens, as candidates (at most CAP)
//   final      top-p over the candidates sorted in workgroup memory, then
//              the draw
//   mass/pick  the draw over the whole vocabulary: each workgroup's kept
//              mass in token order, then the inverse CDF
//   bisect_*   top-p without a small top-k: bisection on the key threshold
//              with each workgroup's mass above it
//
// The token goes to st[0] (st[1] = 1 if it is a stop token) and to next[0],
// the embedding input of the next step; its output count is bumped.
//
//   unbump     undoes the last draw's count: a speculative step's second
//              sample, discarded with its rejected draft

struct Phase {
    phase: u32,
    round: u32,
    _pad0: u32,
    _pad1: u32,
}

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

@group(0) @binding(0) var<uniform> ph: Phase;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
// (token, f32 bits of the bias), sorted by token.
@group(0) @binding(2) var<storage, read> bias: array<vec2u>;
@group(0) @binding(3) var<storage, read> prompt_mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> work: array<f32>;
@group(0) @binding(6) var<storage, read_write> st: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> next: array<u32>;
@group(0) @binding(8) var<uniform> step: Step;
@group(0) @binding(9) var<uniform> p: Params;

const PREP: u32 = 0u;
const REDUCE: u32 = 1u;
const HIST: u32 = 2u;
const DIGIT: u32 = 3u;
const GATHER: u32 = 4u;
const FINAL: u32 = 5u;
const MASS: u32 = 6u;
const PICK: u32 = 7u;
const BISECT_INIT: u32 = 8u;
const BISECT_MASS: u32 = 9u;
const BISECT_STEP: u32 = 10u;
const UNBUMP: u32 = 11u;

const WG: u32 = 256u;
const PER: u32 = 4u;
const SPAN: u32 = WG * PER;
const CAP: u32 = 1024u;
const NONE: u32 = 0xffffffffu;
const LOWEST: f32 = -0x1.fffffep+127f; // -FLT_MAX

// st layout (u32 words).
const S_TOKEN: u32 = 0u;
const S_STOP: u32 = 1u;
const S_TOP: u32 = 2u;
const S_ARGMAX: u32 = 3u;
const S_TAU: u32 = 4u;
const S_PREFIX: u32 = 5u;
const S_LEFT: u32 = 6u;
const S_LO: u32 = 7u;
const S_HI: u32 = 8u;
const S_NCAND: u32 = 9u;
const S_GOAL: u32 = 10u;
const S_SKIP: u32 = 11u;
const S_HIST: u32 = 16u;
const S_PART: u32 = S_HIST + 256u;
// Candidates follow the per-workgroup partials: S_PART + 2 · groups().

var<workgroup> rf: array<f32, WG>;
var<workgroup> ru: array<u32, WG>;
var<workgroup> wh: array<atomic<u32>, 256>;
var<workgroup> sk: array<u32, CAP>;
var<workgroup> si: array<u32, CAP>;
var<workgroup> flag: u32;
var<workgroup> pick_group: u32;
var<workgroup> pick_base: f32;

fn ld(i: u32) -> u32 {
    return atomicLoad(&st[i]);
}

fn sto(i: u32, v: u32) {
    atomicStore(&st[i], v);
}

fn groups() -> u32 {
    return (p.vocab + SPAN - 1u) / SPAN;
}

fn cand_base() -> u32 {
    return S_PART + 2u * groups();
}

fn key(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return select(b ^ 0x80000000u, ~b, (b >> 31u) == 1u);
}

fn top() -> f32 {
    return bitcast<f32>(ld(S_TOP));
}

// Whether value v survives min-p and the threshold tau.
fn kept(v: f32, t: f32, tau: u32) -> bool {
    return key(v) >= tau && (p.min_p <= 0.0 || exp(v - t) >= p.min_p);
}

fn bias_of(i: u32) -> vec2f {
    // (found, bias) by binary search.
    var lo = 0u;
    var hi = p.n_bias;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        let b = bias[mid];
        if (b.x == i) {
            return vec2f(1.0, bitcast<f32>(b.y));
        }
        if (b.x < i) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return vec2f(0.0, 0.0);
}

fn finish(token: u32) {
    var stop = 0u;
    for (var i = 0u; i < p.n_stop; i++) {
        if (p.stops[i / 4u][i % 4u] == token) {
            stop = 1u;
        }
    }
    sto(S_TOKEN, token);
    sto(S_STOP, stop);
    next[0] = token;
    counts[token] += 1u;
}

// Index of the maximum over the workgroup (first on ties); value in rf[0].
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
    return ru[0];
}

// One workgroup's kept mass in token order: thread t sums elements
// [base + t·PER, base + t·PER + PER), then thread 0 adds the 256 sums in
// order. Every phase that recomputes a mass uses this same order.
fn group_mass(lid: u32, g: u32, t: f32, tau: u32) -> f32 {
    let i0 = g * SPAN + lid * PER;
    var s = 0.0;
    for (var j = 0u; j < PER; j++) {
        let i = i0 + j;
        if (i < p.vocab) {
            let v = work[i];
            if (kept(v, t, tau)) {
                s += exp(v - t);
            }
        }
    }
    rf[lid] = s;
    workgroupBarrier();
    var total = 0.0;
    if (lid == 0u) {
        for (var k = 0u; k < WG; k++) {
            total += rf[k];
        }
    }
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let lid = lid3.x;
    let g = wg.x;
    switch ph.phase {
        case PREP: {
            var best = LOWEST;
            var best_i = NONE;
            for (var j = 0u; j < PER; j++) {
                let i = g * SPAN + j * WG + lid;
                if (i < p.vocab) {
                    var v = logits[i];
                    if (p.n_bias > 0u) {
                        let b = bias_of(i);
                        if (b.x != 0.0) {
                            v += b.y;
                        }
                    }
                    let n = counts[i];
                    if (prompt_mask[i] != 0u || n != 0u) {
                        v = select(v * p.repetition, v / p.repetition, v > 0.0);
                    }
                    v -= p.frequency * f32(n);
                    v -= p.presence * select(0.0, 1.0, n != 0u);
                    if (p.greedy == 0u) {
                        v = v / p.temperature;
                    }
                    work[i] = v;
                    if (v > best || (v == best && i < best_i)) {
                        best = v;
                        best_i = i;
                    }
                }
            }
            let bi = argmax(lid, best, best_i);
            if (lid == 0u) {
                sto(S_PART + 2u * g, bitcast<u32>(rf[0]));
                sto(S_PART + 2u * g + 1u, bi);
            }
        }
        case REDUCE: {
            var best = LOWEST;
            var best_i = NONE;
            for (var w = lid; w < groups(); w += WG) {
                let v = bitcast<f32>(ld(S_PART + 2u * w));
                let i = ld(S_PART + 2u * w + 1u);
                if (v > best || (v == best && i < best_i)) {
                    best = v;
                    best_i = i;
                }
            }
            let bi = argmax(lid, best, best_i);
            sto(S_HIST + lid, 0u);
            if (lid == 0u) {
                if (p.greedy != 0u) {
                    finish(bi);
                } else {
                    sto(S_TOP, bitcast<u32>(rf[0]));
                    sto(S_ARGMAX, bi);
                    sto(S_TAU, 0u);
                    sto(S_PREFIX, 0u);
                    sto(S_LEFT, p.top_k);
                    sto(S_NCAND, 0u);
                    sto(S_SKIP, 0u);
                }
            }
        }
        case HIST: {
            if (lid == 0u) {
                flag = ld(S_SKIP);
            }
            if (workgroupUniformLoad(&flag) != 0u) {
                return;
            }
            atomicStore(&wh[lid], 0u);
            workgroupBarrier();
            let t = top();
            let r = ph.round;
            let prefix = ld(S_PREFIX);
            for (var j = 0u; j < PER; j++) {
                let i = g * SPAN + j * WG + lid;
                if (i < p.vocab) {
                    let v = work[i];
                    if (kept(v, t, 0u)) {
                        let k = key(v);
                        var match_ = true;
                        if (r > 0u) {
                            match_ = (k >> (32u - 8u * r)) == prefix;
                        }
                        if (match_) {
                            atomicAdd(&wh[(k >> (24u - 8u * r)) & 255u], 1u);
                        }
                    }
                }
            }
            workgroupBarrier();
            let c = atomicLoad(&wh[lid]);
            if (c > 0u) {
                atomicAdd(&st[S_HIST + lid], c);
            }
        }
        case DIGIT: {
            if (lid == 0u && ld(S_SKIP) == 0u) {
                let r = ph.round;
                var left = ld(S_LEFT);
                var total = 0u;
                if (r == 0u) {
                    for (var d = 0u; d < 256u; d++) {
                        total += ld(S_HIST + d);
                    }
                }
                if (r == 0u && total <= left) {
                    // At most k tokens survive min-p: top-k removes nothing.
                    sto(S_SKIP, 1u);
                } else {
                    var above = 0u;
                    var d = 255u;
                    loop {
                        let c = ld(S_HIST + d);
                        if (above + c >= left || d == 0u) {
                            break;
                        }
                        above += c;
                        d -= 1u;
                    }
                    let prefix = (ld(S_PREFIX) << 8u) | d;
                    sto(S_PREFIX, prefix);
                    sto(S_LEFT, left - above);
                    if (r == 3u) {
                        sto(S_TAU, prefix);
                    }
                }
            }
            workgroupBarrier();
            sto(S_HIST + lid, 0u);
        }
        case GATHER: {
            let t = top();
            let tau = ld(S_TAU);
            let base = cand_base();
            for (var j = 0u; j < PER; j++) {
                let i = g * SPAN + j * WG + lid;
                if (i < p.vocab) {
                    let v = work[i];
                    if (kept(v, t, tau)) {
                        let slot = atomicAdd(&st[S_NCAND], 1u);
                        if (slot < CAP) {
                            sto(base + 2u * slot, i);
                            sto(base + 2u * slot + 1u, bitcast<u32>(v));
                        }
                    }
                }
            }
        }
        case FINAL: {
            let n = min(ld(S_NCAND), CAP);
            let base = cand_base();
            for (var j = lid; j < CAP; j += WG) {
                if (j < n) {
                    si[j] = ld(base + 2u * j);
                    sk[j] = key(bitcast<f32>(ld(base + 2u * j + 1u)));
                } else {
                    si[j] = NONE;
                    sk[j] = 0u;
                }
            }
            workgroupBarrier();
            // Bitonic sort: key descending, then index ascending.
            for (var size = 2u; size <= CAP; size *= 2u) {
                for (var stride = size / 2u; stride > 0u; stride /= 2u) {
                    for (var q = lid; q < CAP / 2u; q += WG) {
                        let a = 2u * stride * (q / stride) + q % stride;
                        let b = a + stride;
                        let up = (a & size) == 0u;
                        let a_first = sk[a] > sk[b] || (sk[a] == sk[b] && si[a] <= si[b]);
                        if (a_first != up) {
                            let tk = sk[a];
                            sk[a] = sk[b];
                            sk[b] = tk;
                            let ti = si[a];
                            si[a] = si[b];
                            si[b] = ti;
                        }
                    }
                    workgroupBarrier();
                }
            }
            // Top-p: keep the largest while the mass ranked above is < p.
            if (lid == 0u) {
                let t = top();
                var kept_n = n;
                if (p.top_p < 1.0) {
                    var z = 0.0;
                    for (var j = 0u; j < n; j++) {
                        z += exp(work[si[j]] - t);
                    }
                    var above = 0.0;
                    var last = 0u;
                    for (var j = 0u; j < n; j++) {
                        if (above >= p.top_p * z) {
                            break;
                        }
                        last = j;
                        above += exp(work[si[j]] - t);
                    }
                    kept_n = last + 1u;
                    while (kept_n < n && sk[kept_n] == sk[last]) {
                        kept_n++;
                    }
                }
                flag = kept_n;
            }
            workgroupBarrier();
            // Drop the rest, then sort the kept by index for the draw.
            let kept_n = workgroupUniformLoad(&flag);
            for (var j = lid; j < CAP; j += WG) {
                if (j >= kept_n) {
                    si[j] = NONE;
                }
            }
            workgroupBarrier();
            for (var size = 2u; size <= CAP; size *= 2u) {
                for (var stride = size / 2u; stride > 0u; stride /= 2u) {
                    for (var q = lid; q < CAP / 2u; q += WG) {
                        let a = 2u * stride * (q / stride) + q % stride;
                        let b = a + stride;
                        let up = (a & size) == 0u;
                        if ((si[a] <= si[b]) != up) {
                            let ti = si[a];
                            si[a] = si[b];
                            si[b] = ti;
                        }
                    }
                    workgroupBarrier();
                }
            }
            if (lid == 0u) {
                let t = top();
                var z = 0.0;
                for (var j = 0u; j < kept_n; j++) {
                    z += exp(work[si[j]] - t);
                }
                let at = step.u * z;
                var c = 0.0;
                var token = si[kept_n - 1u];
                for (var j = 0u; j < kept_n; j++) {
                    c += exp(work[si[j]] - t);
                    if (c > at) {
                        token = si[j];
                        break;
                    }
                }
                finish(token);
            }
        }
        case MASS, BISECT_MASS: {
            var tau = ld(S_TAU);
            if (ph.phase == BISECT_MASS) {
                if (lid == 0u) {
                    flag = select(0u, 1u, ld(S_HI) - ld(S_LO) <= 1u);
                }
                if (workgroupUniformLoad(&flag) != 0u) {
                    return;
                }
                let lo = ld(S_LO);
                tau = lo + (ld(S_HI) - lo) / 2u;
            }
            let m = group_mass(lid, g, top(), tau);
            if (lid == 0u) {
                sto(S_PART + g, bitcast<u32>(m));
            }
        }
        case BISECT_INIT, BISECT_STEP: {
            if (lid == 0u) {
                var z = 0.0;
                for (var w = 0u; w < groups(); w++) {
                    z += bitcast<f32>(ld(S_PART + w));
                }
                if (ph.phase == BISECT_INIT) {
                    sto(S_GOAL, bitcast<u32>(p.top_p * z));
                    sto(S_LO, ld(S_TAU));
                    sto(S_HI, key(top()) + 1u);
                } else {
                    let lo = ld(S_LO);
                    let hi = ld(S_HI);
                    if (hi - lo > 1u) {
                        let mid = lo + (hi - lo) / 2u;
                        if (z >= bitcast<f32>(ld(S_GOAL))) {
                            sto(S_LO, mid);
                        } else {
                            sto(S_HI, mid);
                        }
                    }
                    sto(S_TAU, ld(S_LO));
                }
            }
        }
        case UNBUMP: {
            if (lid == 0u) {
                counts[ld(S_TOKEN)] -= 1u;
            }
        }
        case PICK: {
            let t = top();
            let tau = ld(S_TAU);
            if (lid == 0u) {
                var z = 0.0;
                for (var w = 0u; w < groups(); w++) {
                    z += bitcast<f32>(ld(S_PART + w));
                }
                let at = step.u * z;
                var c = 0.0;
                var found = NONE;
                var last = NONE;
                for (var w = 0u; w < groups(); w++) {
                    let m = bitcast<f32>(ld(S_PART + w));
                    if (m > 0.0) {
                        last = w;
                        if (found == NONE && c <= at && at < c + m) {
                            found = w;
                            pick_base = c;
                        }
                    }
                    c += m;
                }
                if (found == NONE) {
                    // Rounding left the draw at or past the total.
                    found = last;
                    pick_base = c;
                }
                pick_group = found;
            }
            let pg = workgroupUniformLoad(&pick_group);
            // Each thread's sum, in the same order as group_mass.
            let i0 = pg * SPAN + lid * PER;
            var s = 0.0;
            for (var j = 0u; j < PER; j++) {
                let i = i0 + j;
                if (i < p.vocab) {
                    let v = work[i];
                    if (kept(v, t, tau)) {
                        s += exp(v - t);
                    }
                }
            }
            rf[lid] = s;
            workgroupBarrier();
            if (lid == 0u) {
                var z = 0.0;
                for (var w = 0u; w < groups(); w++) {
                    z += bitcast<f32>(ld(S_PART + w));
                }
                let target_at = step.u * z;
                var c = pick_base;
                var token = NONE;
                var last = NONE;
                for (var k = 0u; k < WG && token == NONE; k++) {
                    let base_i = pg * SPAN + k * PER;
                    if (rf[k] > 0.0 && c + rf[k] > target_at) {
                        var cc = c;
                        for (var j = 0u; j < PER; j++) {
                            let i = base_i + j;
                            if (i < p.vocab) {
                                let v = work[i];
                                if (kept(v, t, tau)) {
                                    cc += exp(v - t);
                                    last = i;
                                    if (cc > target_at) {
                                        token = i;
                                        break;
                                    }
                                }
                            }
                        }
                    } else if (rf[k] > 0.0) {
                        for (var j = 0u; j < PER; j++) {
                            let i = base_i + j;
                            if (i < p.vocab && kept(work[i], t, tau)) {
                                last = i;
                            }
                        }
                    }
                    c += rf[k];
                }
                finish(select(token, last, token == NONE));
            }
        }
        default: {}
    }
}
