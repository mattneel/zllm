// Grouped TQ2_T expert matrix multiply. Grid: ceil(rows / 32), expert,
// projection (0=wa, 1=wb). moe_group supplies stable original output slots.
// A four-token tile shares each decoded block, with independent scalar
// accumulators and the same operation order / 32-lane tree as gyro_matvec.
struct Params {
    rows: u32,
    cols: u32,
    n_used: u32,
    x_per_slot: u32,
    x_stride: u32,
    out_stride: u32,
    out_z: u32,
    max_slots: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> wa: array<u32>;
@group(0) @binding(2) var<storage, read> wb: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> count: array<u32>;
@group(0) @binding(6) var<storage, read> slots: array<u32>;
@group(0) @binding(7) var<storage, read> book: array<u32>;

const WG: u32 = 256u;
const LANES: u32 = 32u;
const RPG: u32 = 4u;
const BLOCK_BYTES: u32 = 34u;
var<workgroup> xs: array<f32, 512>;
var<workgroup> partial: array<vec4f, WG>;
var<workgroup> group_count: u32;

fn wbyte(z: u32, i: u32) -> u32 {
    var word = 0u;
    if (z == 0u) { word = wa[i >> 2u]; } else { word = wb[i >> 2u]; }
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}
fn whalf(z: u32, i: u32) -> f32 {
    var word = 0u;
    if (z == 0u) { word = wa[i >> 2u]; } else { word = wb[i >> 2u]; }
    return unpack2x16float(word >> ((i & 2u) * 8u)).x;
}
fn product(pa: vec2f, pb: vec2f, xi: u32) -> f32 {
    return pa.x * xs[xi] + pa.y * xs[xi + 1u] + pb.x * xs[xi + 2u] + pb.y * xs[xi + 3u];
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let lane = li % LANES;
    if (li == 0u) { group_count = count[wg.y]; }
    let n = workgroupUniformLoad(&group_count);
    let blocks = p.cols / 128u;
    let row_bytes = blocks * BLOCK_BYTES;
    let prev_lane = (lane + 31u) & 31u;
    for (var tile = 0u; tile < n; tile += 4u) {
        for (var r = 0u; r < RPG; r++) {
            let j = (wg.x * (WG / LANES) + li / LANES) * RPG + r;
            var a0 = 0.0;
            var a1 = 0.0;
            var a2 = 0.0;
            var a3 = 0.0;
            for (var b = 0u; b < blocks; b++) {
                for (var i = li; i < 512u; i += WG) {
                    let t = i / 128u;
                    var v = 0.0;
                    if (tile + t < n) {
                        let slot = slots[wg.y * p.max_slots + tile + t];
                        var xrow = slot;
                        if (p.x_per_slot == 0u) { xrow = slot / p.n_used; }
                        v = x[xrow * p.x_stride + b * 128u + i % 128u];
                    }
                    xs[i] = v;
                }
                workgroupBarrier();
                if (j < p.rows) {
                    let bo = (wg.y * p.rows + j) * row_bytes + b * BLOCK_BYTES;
                    let state = (wbyte(wg.z, bo + 2u + prev_lane) << 8u) | wbyte(wg.z, bo + 2u + lane);
                    let h = (state & 0x7fffu) * 0x9e3779b1u;
                    let pa = unpack2x16float(book[h >> 21u]);
                    let pb = unpack2x16float(book[(h >> 10u) & 2047u]);
                    let scale = whalf(wg.z, bo);
                    let xi = lane * 4u;
                    a0 += scale * product(pa, pb, xi);
                    a1 += scale * product(pa, pb, 128u + xi);
                    a2 += scale * product(pa, pb, 256u + xi);
                    a3 += scale * product(pa, pb, 384u + xi);
                }
                workgroupBarrier();
            }
            partial[li] = vec4f(a0, a1, a2, a3);
            workgroupBarrier();
            for (var s = LANES / 2u; s > 0u; s /= 2u) {
                if (lane < s) { partial[li] += partial[li + s]; }
                workgroupBarrier();
            }
            if (lane == 0u && j < p.rows) {
                for (var t = 0u; t < 4u; t++) {
                    if (tile + t < n) {
                        let slot = slots[wg.y * p.max_slots + tile + t];
                        out[slot * p.out_stride + wg.z * p.out_z + j] = partial[li][t];
                    }
                }
            }
            workgroupBarrier();
        }
    }
}
