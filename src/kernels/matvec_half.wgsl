// out[t][col0 + j] = Σ_i x[t][i] · w[j][i]: the matrix multiply for steps of
// a few tokens (decode). Same layout as matmul_half: w holds `n` rows of `k`
// half-precision values (BF16 or F16, by the helper file before this one,
// half/<type>.wgsl), packed two per u32, possibly one chunk of a larger matrix.
// k % 8 == 0.
//
// A workgroup computes 8 rows for token t = workgroup_id.y. Each row's dot
// product is split across 32 lanes: lane l reads 16-byte vectors (8 half
// weights) l, l + 32, ... of the row, so the lanes' loads are contiguous,
// then the 32 partial sums are reduced in a fixed tree. The token's x is
// staged in workgroup memory first when it fits (k ≤ MAX_STAGED), so the
// 8 rows share one read of it.
//
// Fused around it, by flag:
//   NORM        x ← rmsnorm(x) · norm_w first (needs staging): every
//               workgroup reduces the row's sum of squares itself
//   ACCUMULATE  out += result (a residual add)
//   SILU_PAIR   out[j] = silu(w[j] · x) · (w[half + j] · x): gate and up
//               rows of one matrix, j < n

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
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read> w: array<vec4u>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> norm_w: array<vec4f>;

const NORM: u32 = 1u;
const ACCUMULATE: u32 = 2u;
const SILU_PAIR: u32 = 4u;

const WG: u32 = 256u;
const LANES: u32 = 32u;
const ROWS: u32 = WG / LANES;
const MAX_STAGED: u32 = 3584u;

var<workgroup> partial: array<f32, WG>;
var<workgroup> partial2: array<f32, WG>;
var<workgroup> xs: array<vec4f, MAX_STAGED / 4u>;

fn dot8(x0: vec4f, x1: vec4f, ww: vec4u) -> f32 {
    return x0.x * half_lo(ww.x) + x0.y * half_hi(ww.x) + x0.z * half_lo(ww.y) + x0.w * half_hi(ww.y) +
        x1.x * half_lo(ww.z) + x1.y * half_hi(ww.z) + x1.z * half_lo(ww.w) + x1.w * half_hi(ww.w);
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let lane = li % LANES;
    let r = li / LANES;
    let j = wg.x * ROWS + r;
    let t = wg.y;
    let units = p.k / 8u;
    let xb = t * (p.k / 4u);
    let staged = p.k <= MAX_STAGED;

    if (staged) {
        var ss = 0.0;
        for (var i = li; i < p.k / 4u; i += WG) {
            let v = x[xb + i];
            xs[i] = v;
            ss += dot(v, v);
        }
        if ((p.flags & NORM) != 0u) {
            partial[li] = ss;
            workgroupBarrier();
            for (var s = WG / 2u; s > 0u; s /= 2u) {
                if (li < s) {
                    partial[li] += partial[li + s];
                }
                workgroupBarrier();
            }
            let scale = 1.0 / sqrt(partial[0] / f32(p.k) + p.eps);
            for (var i = li; i < p.k / 4u; i += WG) {
                xs[i] = xs[i] * scale * norm_w[i];
            }
        }
    }
    workgroupBarrier();

    let pair = (p.flags & SILU_PAIR) != 0u;
    var acc = 0.0;
    var acc2 = 0.0;
    if (j < p.n) {
        let wb = j * units;
        let wb2 = (p.half + j) * units;
        for (var u = lane; u < units; u += LANES) {
            var x0: vec4f;
            var x1: vec4f;
            if (staged) {
                x0 = xs[2u * u];
                x1 = xs[2u * u + 1u];
            } else {
                x0 = x[xb + 2u * u];
                x1 = x[xb + 2u * u + 1u];
            }
            acc += dot8(x0, x1, w[wb + u]);
            if (pair) {
                acc2 += dot8(x0, x1, w[wb2 + u]);
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
