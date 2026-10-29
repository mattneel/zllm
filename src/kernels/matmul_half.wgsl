// out[t][col0 + j] = Σ_i x[t][i] · w[j][i] for t < rows, j < n.
//
// w holds `n` rows of `k` half-precision values (BF16 or F16, by the helper
// file before this one, half/<type>.wgsl; row j contiguous), packed two per u32
// little-endian. It may be one chunk of a larger matrix: `col0` places its
// outputs, `out_stride` is the full output row width. k % TK == 0. With
// `accumulate`, out += result (a residual add).
//
// Tiling: a workgroup computes TT tokens × TN outputs. Each step stages a
// TT×TK tile of x and a TN×TK tile of w (widened to f32) in workgroup memory,
// loading both with consecutive threads on consecutive words.

struct Params {
    rows: u32,
    n: u32,
    k: u32,
    col0: u32,
    out_stride: u32,
    accumulate: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const TT: u32 = 16u;
const TN: u32 = 64u;
const TK: u32 = 32u;
// 256 threads: thread (tt, tn) computes token tt and outputs tn, tn+16, tn+32, tn+48.
const WG: u32 = 256u;

var<workgroup> xs: array<f32, TT * TK>;
var<workgroup> ws: array<f32, TN * (TK + 1u)>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let j0 = wg.x * TN;
    let t0 = wg.y * TT;
    let tt = li / 16u;
    let tn = li % 16u;

    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var k0 = 0u; k0 < p.k; k0 += TK) {
        // x tile: TT×TK floats, two per thread.
        for (var e = li; e < TT * TK; e += WG) {
            let r = e / TK;
            let c = e % TK;
            let t = t0 + r;
            xs[e] = select(0.0, x[t * p.k + k0 + c], t < p.rows);
        }
        // w tile: TN rows × TK/2 words, four per thread.
        for (var e = li; e < TN * (TK / 2u); e += WG) {
            let r = e / (TK / 2u);
            let c = e % (TK / 2u);
            let j = j0 + r;
            var word = 0u;
            if (j < p.n) {
                word = w[(j * p.k + k0) / 2u + c];
            }
            ws[r * (TK + 1u) + 2u * c] = half_lo(word);
            ws[r * (TK + 1u) + 2u * c + 1u] = half_hi(word);
        }
        workgroupBarrier();
        for (var c = 0u; c < TK; c++) {
            let xv = xs[tt * TK + c];
            for (var q = 0u; q < 4u; q++) {
                acc[q] += xv * ws[(tn + 16u * q) * (TK + 1u) + c];
            }
        }
        workgroupBarrier();
    }

    let t = t0 + tt;
    if (t >= p.rows) {
        return;
    }
    for (var q = 0u; q < 4u; q++) {
        let j = j0 + tn + 16u * q;
        if (j < p.n) {
            let o = t * p.out_stride + p.col0 + j;
            if (p.accumulate != 0u) {
                out[o] += acc[q];
            } else {
                out[o] = acc[q];
            }
        }
    }
}
