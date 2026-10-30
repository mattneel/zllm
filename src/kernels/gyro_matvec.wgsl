// Routed-expert matrix-vector products on Gyro TQ2_T weights (lifted from
// gyro-dx's moe_gyro): out[slot][z][row] = W_z,e[row] · x, where e is the
// expert routed to `slot` (ids[slot], written by moe_route on the GPU, so
// no readback) and W_z is projection z of a [n_expert × rows × cols]
// tensor: z = 0 reads `wa`, z = 1 `wb` (gate and up share one dispatch).
//
// A TQ2_T block is 128 weights: a binary16 scale and 32 stream bytes. Step
// s (0..31) has the 16-bit state (byte s − 1 mod 32) << 8 | byte s; its low
// 15 bits, times 0x9e3779b1, give two indices into the 2,048-pair codebook:
// weights 4s..4s+3. A lane group of 32 lanes takes one row, a lane per step,
// over the row's blocks; the lanes' sums then reduce in a fixed tree. x is
// staged in workgroup memory once for the workgroup's 8 rows.
//
// Grid: x = row groups of 8, y = slots (token × n_used + k), z = projections.

struct Params {
    // Output features per expert, and inputs (a multiple of 128).
    rows: u32,
    cols: u32,
    n_used: u32,
    // 1: slot s reads x row s; 0: x row s / n_used (its token).
    x_per_slot: u32,
    x_stride: u32,
    // Floats between slots' outputs, and between the two projections'.
    out_stride: u32,
    out_z: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> wa: array<u32>;
@group(0) @binding(2) var<storage, read> wb: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> ids: array<u32>;
// The codebook: 2,048 pairs of binary16, one pair per word.
@group(0) @binding(6) var<storage, read> book: array<u32>;

const WG: u32 = 256u;
const LANES: u32 = 32u;
const ROWS: u32 = WG / LANES;
const MAX_COLS: u32 = 2560u;
const BLOCK_BYTES: u32 = 34u;

var<workgroup> xs: array<f32, MAX_COLS>;
var<workgroup> partial: array<f32, WG>;

fn wbyte(z: u32, i: u32) -> u32 {
    var word = 0u;
    if (z == 0u) {
        word = wa[i >> 2u];
    } else {
        word = wb[i >> 2u];
    }
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}

fn whalf(z: u32, i: u32) -> f32 {
    return unpack2x16float(wbyte(z, i) | (wbyte(z, i + 1u) << 8u)).x;
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let slot = wg.y;
    let z = wg.z;
    let lane = li % LANES;
    let j = wg.x * ROWS + li / LANES;

    var xrow = slot;
    if (p.x_per_slot == 0u) {
        xrow = slot / p.n_used;
    }
    for (var i = li; i < p.cols; i += WG) {
        xs[i] = x[xrow * p.x_stride + i];
    }
    workgroupBarrier();

    var acc = 0.0;
    if (j < p.rows) {
        let blocks = p.cols / 128u;
        let row_bytes = blocks * BLOCK_BYTES;
        let row0 = (ids[slot] * p.rows + j) * row_bytes;
        let prev_lane = (lane + 31u) & 31u;
        for (var b = 0u; b < blocks; b++) {
            let bo = row0 + b * BLOCK_BYTES;
            let state = (wbyte(z, bo + 2u + prev_lane) << 8u) | wbyte(z, bo + 2u + lane);
            let h = (state & 0x7fffu) * 0x9e3779b1u;
            let pa = unpack2x16float(book[h >> 21u]);
            let pb = unpack2x16float(book[(h >> 10u) & 2047u]);
            let xi = b * 128u + lane * 4u;
            let part = pa.x * xs[xi] + pa.y * xs[xi + 1u] + pb.x * xs[xi + 2u] + pb.y * xs[xi + 3u];
            acc += whalf(z, bo) * part;
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
    if (lane == 0u && j < p.rows) {
        out[slot * p.out_stride + z * p.out_z + j] = partial[li];
    }
}
