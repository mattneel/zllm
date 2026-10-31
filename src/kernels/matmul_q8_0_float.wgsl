// out[t][col0 + j] = Σ_i x[t][i] · w[j][i], without rounding x.
// Repacked Q8_0 rows have k signed bytes, k/32 f16 scales, then padding
// to a 16-byte boundary (repack_q8_0.wgsl). k and k_slice are multiples
// of 32. Workgroup z sums [z*k_slice, min(k, (z+1)*k_slice)) and writes
// at z*slice_stride, following matmul_quant/matmul_reduce conventions.
//
// Each workgroup computes 16 tokens × 64 outputs. A 32-position tile of
// f32 activations and decoded f32 weights is shared across the tile.
// Each output is accumulated in increasing k order, regardless of how
// many tokens or output columns occupy the tile.

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
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const TT: u32 = 16u;
const TN: u32 = 64u;
const TK: u32 = 32u;
const WG: u32 = 256u;

var<workgroup> xs: array<f32, TT * TK>;
var<workgroup> ws: array<f32, TN * (TK + 1u)>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let kb = wg.z * p.k_slice;
    let ke = min(p.k, kb + p.k_slice);
    let j0 = wg.x * TN;
    let t0 = wg.y * TT;
    let tt = li / 16u;
    let tn = li % 16u;
    let row_words = ((p.k + p.k / 16u + 15u) & ~15u) / 4u;

    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var k0 = kb; k0 < ke; k0 += TK) {
        // Two f32 activation values per thread; no f16 conversion or Q8 step.
        for (var e = li; e < TT * TK; e += WG) {
            let r = e / TK;
            let c = e % TK;
            let t = t0 + r;
            var v = 0.0;
            if (t < p.rows) {
                v = x[t * p.k + k0 + c];
            }
            xs[e] = v;
        }
        // Two packed words per thread, four signed weights per word.
        // Scales follow all k weight bytes; two scales share a u32.
        for (var e = li; e < TN * (TK / 4u); e += WG) {
            let r = e / (TK / 4u);
            let c = e % (TK / 4u);
            let j = j0 + r;
            var word = 0u;
            var scale = 0.0;
            if (j < p.n) {
                let base = j * row_words;
                word = w[base + k0 / 4u + c];
                let block = k0 / TK;
                let scales = unpack2x16float(w[base + p.k / 4u + block / 2u]);
                scale = scales[block % 2u];
            }
            let o = r * (TK + 1u) + 4u * c;
            ws[o] = scale * f32(bitcast<i32>(word << 24u) >> 24u);
            ws[o + 1u] = scale * f32(bitcast<i32>(word << 16u) >> 24u);
            ws[o + 2u] = scale * f32(bitcast<i32>(word << 8u) >> 24u);
            ws[o + 3u] = scale * f32(bitcast<i32>(word) >> 24u);
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
            let o = wg.z * p.slice_stride + t * p.out_stride + p.col0 + j;
            if (p.accumulate != 0u) {
                out[o] += acc[q];
            } else {
                out[o] = acc[q];
            }
        }
    }
}
