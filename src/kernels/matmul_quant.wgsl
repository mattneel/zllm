// out[t][col0 + j] = Σ_i x[t][i] · w[j][i] for t < rows, j < n, quantized
// weights (the type file before this one decodes them): the prefill
// multiply. With `accumulate`, out += result. Workgroup z sums only
// positions [z·k_slice, (z + 1)·k_slice) and writes at z·slice_stride: a
// split over k (kernels.matmulSlices) that matmul_reduce adds up.
//
// A workgroup computes TM tokens × TN outputs, 4 × 8 per thread. Each step
// of TK = 16 stages x for those positions (16-byte loads) and the TN rows'
// weights, half a 32-weight unit, each thread decoding one row's four words
// to f32 (quad, 16-byte loads); both k-major in workgroup memory. Per
// position a thread then reads one vector of x and two of weights for 32
// multiply-adds, and each decoded weight serves TM tokens. TM is small, as
// a step's tile costs the same however few of its tokens are real, and
// the scheduler's steps on a large model are short.

struct Params {
    rows: u32,
    n: u32,
    k: u32,
    col0: u32,
    out_stride: u32,
    accumulate: u32,
    k_slice: u32,
    slice_stride: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read> w: array<vec4u>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const TM: u32 = 32u;
const TN: u32 = 128u;
const TK: u32 = 16u;
const WG: u32 = 128u;

// xs[kk][m], ws[kk][n]: k-major, 4 tokens or 4 outputs per vector.
var<workgroup> xs: array<vec4f, TK * TM / 4u>;
var<workgroup> ws: array<vec4f, TK * TN / 4u>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let kb = wg.z * p.k_slice;
    let ke = min(p.k, kb + p.k_slice);
    let n0 = wg.x * TN;
    let t0 = wg.y * TM;
    // This thread's tokens t0 + 4·tg + a and outputs n0 + 8·ng + c.
    let tg = li % (TM / 4u);
    let ng = li / (TM / 4u);
    let row_bytes = (p.k / BLOCK_ELEMS) * BLOCK_BYTES;
    // The weight row this thread decodes.
    let wj = n0 + li;
    let k4 = p.k / 4u;

    // acc[2a + h]: token 4·tg + a, outputs 8·ng + 4h ..
    var acc: array<vec4f, 8>;
    for (var k0 = kb; k0 < ke; k0 += TK) {
        for (var e = li; e < TM * TK / 4u; e += WG) {
            let m = e / 4u;
            let q = e % 4u;
            let t = t0 + m;
            var v = vec4f(0.0);
            if (t < p.rows) {
                v = x[t * k4 + k0 / 4u + q];
            }
            for (var c = 0u; c < 4u; c++) {
                xs[((4u * q + c) * TM + m) / 4u][m % 4u] = v[c];
            }
        }
        // Words (k0 % 32)/4 .. +3 of unit k0/32: positions k0 .. k0 + 15.
        let w0 = (k0 % 32u) / 4u;
        if (wj < p.n) {
            let v = quad(head(wj * row_bytes, k0 / 32u), w0);
            for (var q = 0u; q < 4u; q++) {
                for (var c = 0u; c < 4u; c++) {
                    ws[((4u * q + c) * TN + li) / 4u][li % 4u] = v[q][c];
                }
            }
        } else {
            for (var kk = 0u; kk < TK; kk++) {
                ws[(kk * TN + li) / 4u][li % 4u] = 0.0;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < TK; kk++) {
            let x0 = xs[kk * TM / 4u + tg];
            let w0v = ws[kk * TN / 4u + 2u * ng];
            let w1v = ws[kk * TN / 4u + 2u * ng + 1u];
            for (var a = 0u; a < 4u; a++) {
                acc[2u * a] += x0[a] * w0v;
                acc[2u * a + 1u] += x0[a] * w1v;
            }
        }
        workgroupBarrier();
    }

    for (var a = 0u; a < 4u; a++) {
        let t = t0 + 4u * tg + a;
        if (t >= p.rows) {
            break;
        }
        for (var c = 0u; c < 8u; c++) {
            let j = n0 + 8u * ng + c;
            if (j < p.n) {
                let o = wg.z * p.slice_stride + t * p.out_stride + p.col0 + j;
                let v = acc[2u * a + c / 4u][c % 4u];
                if (p.accumulate != 0u) {
                    out[o] += v;
                } else {
                    out[o] = v;
                }
            }
        }
    }
}
