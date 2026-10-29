// out[t][col0 + j] = Σ_i x[t][i] · w[j][i]: the matrix multiply for steps of
// a few tokens (decode). Same layout as matmul_bf16: w holds `n` rows of `k`
// bf16 values, packed two per u32, possibly one chunk of a larger matrix.
// k % 8 == 0.
//
// A workgroup computes 8 rows for token t = workgroup_id.y. Each row's dot
// product is split across 32 lanes: lane l reads 16-byte vectors (8 bf16
// weights) l, l + 32, ... of the row, so the lanes' loads are contiguous,
// then the 32 partial sums are reduced in a fixed tree.

struct Params {
    n: u32,
    k: u32,
    col0: u32,
    out_stride: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read> w: array<vec4u>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const WG: u32 = 256u;
const LANES: u32 = 32u;
const ROWS: u32 = WG / LANES;

var<workgroup> partial: array<f32, WG>;

fn lo(word: u32) -> f32 {
    return bitcast<f32>(word << 16u);
}

fn hi(word: u32) -> f32 {
    return bitcast<f32>(word & 0xffff0000u);
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let lane = li % LANES;
    let r = li / LANES;
    let j = wg.x * ROWS + r;
    let t = wg.y;
    let units = p.k / 8u;

    var acc = 0.0;
    if (j < p.n) {
        let wb = j * units;
        let xb = t * (p.k / 4u);
        for (var u = lane; u < units; u += LANES) {
            let ww = w[wb + u];
            let x0 = x[xb + 2u * u];
            let x1 = x[xb + 2u * u + 1u];
            acc += x0.x * lo(ww.x) + x0.y * hi(ww.x) + x0.z * lo(ww.y) + x0.w * hi(ww.y);
            acc += x1.x * lo(ww.z) + x1.y * hi(ww.z) + x1.z * lo(ww.w) + x1.w * hi(ww.w);
        }
    }
    partial[li] = acc;
    workgroupBarrier();
    for (var s = LANES / 2u; s > 0u; s /= 2u) {
        if (lane < s) {
            partial[li] += partial[li + s];
        }
        workgroupBarrier();
    }
    if (lane == 0u && j < p.n) {
        out[t * p.out_stride + p.col0 + j] = partial[li];
    }
}
