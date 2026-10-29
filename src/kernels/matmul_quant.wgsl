// out[t][col0 + j] = Σ_i x[t][i] · w[j][i] for t < rows, j < n, quantized
// weights (the type file before this one decodes them): the prefill
// multiply, as matmul_bf16. A workgroup computes TT tokens × TN outputs;
// each step stages a TT×TK tile of x and the TN rows' 32-weight units at
// k0 (TK = 32), decoded to f32, in workgroup memory. With `accumulate`,
// out += result.

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
@group(0) @binding(2) var<storage, read> w: array<vec4u>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const TT: u32 = 16u;
const TN: u32 = 64u;
const TK: u32 = 32u;
const WG: u32 = 256u;

var<workgroup> xs: array<f32, TT * TK>;
var<workgroup> ws: array<f32, TN * (TK + 1u)>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let j0 = wg.x * TN;
    let t0 = wg.y * TT;
    let tt = li / 16u;
    let tn = li % 16u;
    let row_bytes = (p.k / BLOCK_ELEMS) * BLOCK_BYTES;

    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var k0 = 0u; k0 < p.k; k0 += TK) {
        for (var e = li; e < TT * TK; e += WG) {
            let t = t0 + e / TK;
            xs[e] = select(0.0, x[t * p.k + k0 + e % TK], t < p.rows);
        }
        // One thread decodes one row's unit.
        if (li < TN) {
            let j = j0 + li;
            if (j < p.n) {
                let h = head(j * row_bytes, k0 / TK);
                for (var i0 = 0u; i0 < 8u; i0 += 4u) {
                    let q = quad(h, i0);
                    for (var e = 0u; e < 4u; e++) {
                        for (var c = 0u; c < 4u; c++) {
                            ws[li * (TK + 1u) + 4u * (i0 + e) + c] = q[e][c];
                        }
                    }
                }
            } else {
                for (var c = 0u; c < TK; c++) {
                    ws[li * (TK + 1u) + c] = 0.0;
                }
            }
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
