// Gated delta rule over the step's m tokens. Value head j's state S is
// dim × dim (S[a][b] at state[j·dim² + a·dim + b]); per token t, with q, k,
// v the head's slices of conv row t (q, k already L2-normalized; value head
// j reads key head j mod k_heads):
//
//   S ← e^g · S
//   δ = β · (v − Sᵀk)
//   S ← S + k δᵀ
//   o = Sᵀq / sqrt(dim)
//
// o goes to `out` un-normalized; gated_norm applies the RMSNorm and gate.
//
// It also advances the conv history that delta_prep read: hist becomes the
// last K − 1 raw inputs (rows of the fused projection `proj`) of the
// sequence including this step.
//
// With step.keep = r + 1, the state and the conv history as they stand after
// row r also go to `kept` (the state at 0, the history at p.kept_hist): a
// rejected speculative token after row r rolls back by copying them over.
//
// Column b of S depends only on column b, so a head's columns split across
// workgroups: workgroup (j, c) owns columns [c·COLS, (c+1)·COLS), and thread
// (row group r, column) keeps rows [r·dim/R, (r+1)·dim/R) of its column in
// registers for the whole step. The state is read once and written once per
// step; the two dot products per token reduce over the R row groups.

struct Params {
    k_heads: u32,
    v_heads: u32,
    dim: u32,
    conv_stride: u32,
    taps: u32,
    row: u32,
    kept_hist: u32,
    _pad: u32,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    keep: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> conv: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> state: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> step: Step;
@group(0) @binding(7) var<storage, read> proj: array<f32>;
@group(0) @binding(8) var<storage, read_write> hist: array<f32>;
@group(0) @binding(9) var<storage, read_write> kept: array<f32>;

const WG: u32 = 256u;
const COLS: u32 = 32u;
const R: u32 = WG / COLS;
const MAX_DIM: u32 = 256u;
const MAX_ROWS: u32 = MAX_DIM / R;

var<workgroup> qs: array<f32, MAX_DIM>;
var<workgroup> ks: array<f32, MAX_DIM>;
var<workgroup> red: array<f32, WG>;
var<workgroup> delta: array<f32, COLS>;

// hist[i] ← the input K − 1 − i positions before the step's end, for
// every channel: one thread per channel, oldest slot first, so slots still
// to be read are not yet overwritten. The kept history (as after row
// keep − 1) is taken first, from the history before any slot moves.
fn advance_history(id: u32, threads: u32) {
    let back = p.taps - 1u;
    for (var c = id; c < p.conv_stride; c += threads) {
        if (step.keep != 0u) {
            for (var i = 0u; i < back; i++) {
                let j = step.keep + i;
                var v = 0.0;
                if (j < back) {
                    v = hist[j * p.conv_stride + c];
                } else {
                    v = proj[(j - back) * p.row + c];
                }
                kept[p.kept_hist + i * p.conv_stride + c] = v;
            }
        }
        for (var i = 0u; i < back; i++) {
            let j = step.m + i;
            if (j < back) {
                hist[i * p.conv_stride + c] = hist[j * p.conv_stride + c];
            } else {
                hist[i * p.conv_stride + c] = proj[(j - back) * p.row + c];
            }
        }
    }
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let lid = lid3.x;
    advance_history((wg.y * nwg.x + wg.x) * WG + lid, nwg.x * nwg.y * WG);
    let j = wg.x;
    let col = lid % COLS;
    let r = lid / COLS;
    let d = p.dim;
    let b = wg.y * COLS + col;
    let in_head = b < d;
    let rows = d / R;
    let a0 = r * rows;
    let key_dim = p.k_heads * d;
    let kh = j % p.k_heads;
    let s0 = j * d * d;
    let scale = 1.0 / sqrt(f32(d));

    var s: array<f32, MAX_ROWS>;
    for (var i = 0u; i < rows; i++) {
        s[i] = select(0.0, state[s0 + (a0 + i) * d + b], in_head);
    }

    for (var t = 0u; t < step.m; t++) {
        let row = t * p.conv_stride;
        for (var i = lid; i < d; i += WG) {
            qs[i] = conv[row + kh * d + i];
            ks[i] = conv[row + key_dim + kh * d + i];
        }
        workgroupBarrier();

        let decay = exp(g[t * p.v_heads + j]);
        var stk = 0.0;
        for (var i = 0u; i < rows; i++) {
            s[i] *= decay;
            stk += s[i] * ks[a0 + i];
        }
        red[lid] = stk;
        workgroupBarrier();
        if (r == 0u) {
            var total = 0.0;
            for (var q = 0u; q < R; q++) {
                total += red[q * COLS + col];
            }
            let v = select(0.0, conv[row + 2u * key_dim + j * d + b], in_head);
            delta[col] = (v - total) * beta[t * p.v_heads + j];
        }
        workgroupBarrier();

        let dl = delta[col];
        var o = 0.0;
        for (var i = 0u; i < rows; i++) {
            s[i] += ks[a0 + i] * dl;
            o += s[i] * qs[a0 + i];
        }
        red[lid] = o;
        workgroupBarrier();
        if (r == 0u && in_head) {
            var total = 0.0;
            for (var q = 0u; q < R; q++) {
                total += red[q * COLS + col];
            }
            out[t * p.v_heads * d + j * d + b] = total * scale;
        }
        if (t + 1u == step.keep && in_head) {
            for (var i = 0u; i < rows; i++) {
                kept[s0 + (a0 + i) * d + b] = s[i];
            }
        }
        workgroupBarrier();
    }

    if (in_head) {
        for (var i = 0u; i < rows; i++) {
            state[s0 + (a0 + i) * d + b] = s[i];
        }
    }
}
