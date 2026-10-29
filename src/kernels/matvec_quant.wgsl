// out[t][col0 + j] = Σ_i x[t][i] · w[j][i] for quantized weights: the
// decode-path multiply, as matvec_bf16. A workgroup computes 8 lane groups
// × R rows for token t = workgroup_id.y. A row's blocks are split into
// (block, part) tasks across the group's 32 lanes, so consecutive lanes
// read consecutive words (quant/<type>_dot.wgsl decodes a part); then the
// lanes' partial sums reduce in a fixed tree.
//
// Fused around it, by flag (as matvec_bf16):
//   NORM        x ← rmsnorm(x) · norm_w first; every workgroup reduces the
//               row's sum of squares itself
//   ACCUMULATE  out += result
//   SILU_PAIR   out[j] = silu(w[j] · x) · (w[half + j] · x), j < n
// x is staged in workgroup memory in tiles of TILE floats (normalized as
// they are staged), so each workgroup reads it once for all its rows: read
// from storage per row, the 27B's 5,120-wide inputs cost ten times the
// weights' own traffic.

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
// Rows per lane group: a staged x tile serves ROWS · R rows.
const R: u32 = 1u;
const ROWS: u32 = WG / LANES;
const TILE: u32 = 2048u;

var<workgroup> partial: array<f32, WG * R>;
var<workgroup> partial2: array<f32, WG * R>;
var<workgroup> xs: array<vec4f, TILE / 4u>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let lane = li % LANES;
    let r = li / LANES;
    // This lane group's rows: j0 + q, q < R.
    let j0 = (wg.x * ROWS + r) * R;
    let t = wg.y;
    let k4 = p.k / 4u;
    let xb = t * k4;
    let norm = (p.flags & NORM) != 0u;

    var scale = 1.0;
    if (norm) {
        var ss = 0.0;
        for (var i = li; i < k4; i += WG) {
            let v = x[xb + i];
            ss += dot(v, v);
        }
        partial[li] = ss;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s /= 2u) {
            if (li < s) {
                partial[li] += partial[li + s];
            }
            workgroupBarrier();
        }
        scale = 1.0 / sqrt(partial[0] / f32(p.k) + p.eps);
    }

    let pair = (p.flags & SILU_PAIR) != 0u;
    let row_bytes = (p.k / BLOCK_ELEMS) * BLOCK_BYTES;
    var acc: array<f32, R>;
    var acc2: array<f32, R>;
    for (var k0 = 0u; k0 < p.k; k0 += TILE) {
        let len4 = min(TILE, p.k - k0) / 4u;
        workgroupBarrier();
        for (var i = li; i < len4; i += WG) {
            let g = k0 / 4u + i;
            var v = x[xb + g];
            if (norm) {
                v = v * scale * norm_w[g];
            }
            xs[i] = v;
        }
        workgroupBarrier();
        // (block, part) tasks: consecutive lanes read consecutive words.
        let tasks = (4u * len4 / BLOCK_ELEMS) * PARTS;
        for (var task = lane; task < tasks; task += LANES) {
            let blk = task / PARTS;
            let part = task % PARTS;
            let gblk = k0 / BLOCK_ELEMS + blk;
            let xb4 = blk * (BLOCK_ELEMS / 4u);
            for (var q = 0u; q < R; q++) {
                let j = j0 + q;
                if (j >= p.n) {
                    break;
                }
                acc[q] += block_dot(j * row_bytes, gblk, xb4, part);
                if (pair) {
                    acc2[q] += block_dot((p.half + j) * row_bytes, gblk, xb4, part);
                }
            }
        }
    }
    for (var q = 0u; q < R; q++) {
        partial[q * WG + li] = acc[q];
        partial2[q * WG + li] = acc2[q];
    }
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s /= 2u) {
        if (lane < s) {
            for (var q = 0u; q < R; q++) {
                partial[q * WG + li] += partial[q * WG + li + s];
                partial2[q * WG + li] += partial2[q * WG + li + s];
            }
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        for (var q = 0u; q < R; q++) {
            let j = j0 + q;
            if (j >= p.n) {
                break;
            }
            let o = t * p.out_stride + p.col0 + j;
            var v = partial[q * WG + li];
            if (pair) {
                v = v / (1.0 + exp(-v)) * partial2[q * WG + li];
            }
            if ((p.flags & ACCUMULATE) != 0u) {
                v += out[o];
            }
            out[o] = v;
        }
    }
}
