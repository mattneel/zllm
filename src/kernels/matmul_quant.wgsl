// out[t][col0 + j] = Σ_i x[t][i] · w[j][i] for t < rows, j < n, quantized
// weights (the type file before this one decodes them) and activations
// rounded to 8-bit blocks (quantize_q8: x = qx · dx, blocks of ACT_BLOCK):
// the prefill multiply, as llama.cpp's CPU backend computes it. Each weight
// sub-block's dot product is exact in integers (dot4I8Packed), then scaled:
// Σ = dx · (s · Σ q·qx − m · Σ qx). With `accumulate`, out += result.
// Workgroup z sums only positions [z·k_slice, (z + 1)·k_slice) and writes
// at z·slice_stride: a split over k (kernels.matmulSlices) that
// matmul_reduce adds up.
//
// A workgroup computes TM tokens × TN outputs, 4 × 8 per thread, one
// 32-weight unit per step: it stages the TM tokens' 8 words of qx and
// their dx, and each thread decodes one row's unit to 8 words of signed
// bytes and its scales. The next unit is fetched and decoded into
// registers while the current one computes, hiding the load latency. A small TM suits the short steps the scheduler
// gives a large model: a partial tile costs as much as a whole one.

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
@group(0) @binding(1) var<storage, read> qx: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<vec4u>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> dx: array<f32>;

const TM: u32 = 32u;
const TN: u32 = 128u;
const WG: u32 = 128u;

// xq[e][m]: word e of the unit for 4 tokens; wq[e][n]: for 4 rows.
var<workgroup> xq: array<vec4u, 8u * TM / 4u>;
var<workgroup> xd: array<f32, TM>;
var<workgroup> wq: array<vec4u, 8u * TN / 4u>;
// The rows' scales for words 0..3 and 4..7, and mins, 4 rows per vector.
var<workgroup> s0: array<vec4f, TN / 4u>;
var<workgroup> s1: array<vec4f, TN / 4u>;
var<workgroup> mn: array<vec4f, TN / 4u>;

struct Unit {
    lo: vec4u,
    hi: vec4u,
    s: vec3f,
}

// This thread's two words of the TM tokens' qx for the unit at k0: word
// e = li + i·WG is token e/8's word e%8.
fn fetch_x(t0: u32, k0: u32, li: u32) -> vec2u {
    var r = vec2u(0u);
    for (var i = 0u; i < 2u; i++) {
        let e = li + i * WG;
        let t = t0 + e / 8u;
        if (t < p.rows) {
            r[i] = qx[t * (p.k / 4u) + k0 / 4u + e % 8u];
        }
    }
    return r;
}

// dx of token t0 + li (li < TM) for the unit at k0.
fn fetch_dx(t0: u32, k0: u32, li: u32) -> f32 {
    let t = t0 + li;
    if (li >= TM || t >= p.rows) {
        return 0.0;
    }
    return dx[t * (p.k / ACT_BLOCK) + k0 / ACT_BLOCK];
}

// Row j's unit at k0 (zero past n).
fn fetch_w(j: u32, row_bytes: u32, k0: u32) -> Unit {
    var r = Unit(vec4u(0u), vec4u(0u), vec3f(0.0));
    if (j < p.n) {
        let h = head(j * row_bytes, k0 / 32u);
        r = Unit(quad_i8(h, 0u), quad_i8(h, 4u), scales(h));
    }
    return r;
}

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
    let k4 = p.k / 4u;
    let blocks = p.k / ACT_BLOCK;
    // The weight row this thread decodes.
    let wj = n0 + li;

    // acc[2a + h]: token 4·tg + a, outputs 8·ng + 4h ..
    var acc: array<vec4f, 8>;
    // The next unit's x words, dx and decoded weights, fetched into
    // registers while the current unit computes.
    var nx = fetch_x(t0, kb, li);
    var nd = fetch_dx(t0, kb, li);
    var nw = fetch_w(wj, row_bytes, kb);
    for (var k0 = kb; k0 < ke; k0 += 32u) {
        for (var i = 0u; i < 2u; i++) {
            let e = li + i * WG;
            let m = e / 8u;
            let q = e % 8u;
            xq[(q * TM + m) / 4u][m % 4u] = nx[i];
        }
        if (li < TM) {
            xd[li] = nd;
        }
        for (var e = 0u; e < 4u; e++) {
            wq[(e * TN + li) / 4u][li % 4u] = nw.lo[e];
            wq[((4u + e) * TN + li) / 4u][li % 4u] = nw.hi[e];
        }
        s0[li / 4u][li % 4u] = nw.s.x;
        s1[li / 4u][li % 4u] = nw.s.y;
        mn[li / 4u][li % 4u] = nw.s.z;
        workgroupBarrier();
        if (k0 + 32u < ke) {
            nx = fetch_x(t0, k0 + 32u, li);
            nd = fetch_dx(t0, k0 + 32u, li);
            nw = fetch_w(wj, row_bytes, k0 + 32u);
        }

        let d = vec4f(xd[4u * tg], xd[4u * tg + 1u], xd[4u * tg + 2u], xd[4u * tg + 3u]);
        for (var half = 0u; half < 2u; half++) {
            // is[2a + h]: Σ q·qx over this half's 4 words. Initialized
            // explicitly: without an initializer, Dawn on llvmpipe carried
            // the sums over from the previous iteration.
            var is = array<vec4i, 8>();
            for (var e4 = 0u; e4 < 4u; e4++) {
                let e = 4u * half + e4;
                let xv = xq[e * TM / 4u + tg];
                let w0 = wq[e * TN / 4u + 2u * ng];
                let w1 = wq[e * TN / 4u + 2u * ng + 1u];
                for (var a = 0u; a < 4u; a++) {
                    is[2u * a] += vec4i(dot4I8Packed(xv[a], w0.x), dot4I8Packed(xv[a], w0.y), dot4I8Packed(xv[a], w0.z), dot4I8Packed(xv[a], w0.w));
                    is[2u * a + 1u] += vec4i(dot4I8Packed(xv[a], w1.x), dot4I8Packed(xv[a], w1.y), dot4I8Packed(xv[a], w1.z), dot4I8Packed(xv[a], w1.w));
                }
            }
            var sa = s0[2u * ng];
            var sb = s0[2u * ng + 1u];
            if (half == 1u) {
                sa = s1[2u * ng];
                sb = s1[2u * ng + 1u];
            }
            for (var a = 0u; a < 4u; a++) {
                acc[2u * a] += d[a] * (sa * vec4f(is[2u * a]));
                acc[2u * a + 1u] += d[a] * (sb * vec4f(is[2u * a + 1u]));
            }
        }
        if (HAS_MIN) {
            let ma = mn[2u * ng];
            let mb = mn[2u * ng + 1u];
            for (var a = 0u; a < 4u; a++) {
                var sum = 0;
                for (var e = 0u; e < 8u; e++) {
                    sum += dot4I8Packed(xq[e * TM / 4u + tg][a], 0x01010101u);
                }
                let dm = d[a] * f32(sum);
                acc[2u * a] -= dm * ma;
                acc[2u * a + 1u] -= dm * mb;
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
