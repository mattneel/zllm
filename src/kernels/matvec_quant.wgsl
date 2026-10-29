// out[t][col0 + j] = Σ_i x[t][i] · w[j][i] for quantized weights (the type
// file before this one decodes them): the decode-path multiply, as
// matvec_bf16. A workgroup computes 8 rows for token t = workgroup_id.y;
// each row's 32-weight units are split across 32 lanes, then the lanes'
// partial sums reduce in a fixed tree.
//
// Fused around it, by flag (as matvec_bf16):
//   NORM        x ← rmsnorm(x) · norm_w first; every workgroup reduces the
//               row's sum of squares itself
//   ACCUMULATE  out += result
//   SILU_PAIR   out[j] = silu(w[j] · x) · (w[half + j] · x), j < n
// x is staged in workgroup memory when it fits (k ≤ MAX_STAGED); wider
// inputs are read, and normalized, from storage.

struct Params {
    n: u32,
    k: u32,
    col0: u32,
    out_stride: u32,
    eps: f32,
    flags: u32,
    half: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> norm_w: array<f32>;

const NORM: u32 = 1u;
const ACCUMULATE: u32 = 2u;
const SILU_PAIR: u32 = 4u;

const WG: u32 = 256u;
const LANES: u32 = 32u;
const ROWS: u32 = WG / LANES;
const MAX_STAGED: u32 = 3584u;

var<workgroup> partial: array<f32, WG>;
var<workgroup> partial2: array<f32, WG>;
var<workgroup> xs: array<f32, MAX_STAGED>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let lane = li % LANES;
    let r = li / LANES;
    let j = wg.x * ROWS + r;
    let t = wg.y;
    let xb = t * p.k;
    let staged = p.k <= MAX_STAGED;
    let norm = (p.flags & NORM) != 0u;

    var ss = 0.0;
    for (var i = li; i < p.k; i += WG) {
        let v = x[xb + i];
        if (staged) {
            xs[i] = v;
        }
        ss += v * v;
    }
    var scale = 1.0;
    if (norm) {
        partial[li] = ss;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s /= 2u) {
            if (li < s) {
                partial[li] += partial[li + s];
            }
            workgroupBarrier();
        }
        scale = 1.0 / sqrt(partial[0] / f32(p.k) + p.eps);
        if (staged) {
            for (var i = li; i < p.k; i += WG) {
                xs[i] = xs[i] * scale * norm_w[i];
            }
        }
    }
    workgroupBarrier();

    let pair = (p.flags & SILU_PAIR) != 0u;
    let row_bytes = (p.k / BLOCK_ELEMS) * BLOCK_BYTES;
    var acc = 0.0;
    var acc2 = 0.0;
    if (j < p.n) {
        for (var u = lane; u < p.k / 32u; u += LANES) {
            let wv = unit(j * row_bytes, u);
            var wv2: array<f32, 32>;
            if (pair) {
                wv2 = unit((p.half + j) * row_bytes, u);
            }
            for (var l = 0u; l < 32u; l++) {
                let i = 32u * u + l;
                var xv: f32;
                if (staged) {
                    xv = xs[i];
                } else if (norm) {
                    xv = x[xb + i] * scale * norm_w[i];
                } else {
                    xv = x[xb + i];
                }
                acc += xv * wv[l];
                if (pair) {
                    acc2 += xv * wv2[l];
                }
            }
        }
    }
    partial[li] = acc;
    partial2[li] = acc2;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s /= 2u) {
        if (lane < s) {
            partial[li] += partial[li + s];
            partial2[li] += partial2[li + s];
        }
        workgroupBarrier();
    }
    if (lane == 0u && j < p.n) {
        let o = t * p.out_stride + p.col0 + j;
        var v = partial[li];
        if (pair) {
            v = v / (1.0 + exp(-v)) * partial2[li];
        }
        if ((p.flags & ACCUMULATE) != 0u) {
            v += out[o];
        }
        out[o] = v;
    }
}
