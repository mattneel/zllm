// y = H₁₂₈ (signs · x) for each 128-value block of each row: the basis the
// Gyro expert weights are stored in (lifted from gyro-dx's rotate128). With
// `silu_pair`, x is first silu(gate) · up from a row of [gate | up] (the
// expert activation, before the down projection).
//
// Grid: x = blocks of the row, y = rows. The butterflies run in workgroup
// memory, 64 pairs a stage; then the normalization 1/√128.

struct Params {
    // Values a row has (a multiple of 128).
    width: u32,
    // Floats between rows of x, and of y.
    x_stride: u32,
    y_stride: u32,
    // 1: x rows are [gate (width) | up (width)].
    silu_pair: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> signs: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

var<workgroup> buf: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let col0 = wg.x * 128u;
    let xb = wg.y * p.x_stride;
    for (var i = li; i < 128u; i += 64u) {
        let c = col0 + i;
        var v = x[xb + c];
        if (p.silu_pair != 0u) {
            v = v / (1.0 + exp(-v)) * x[xb + p.width + c];
        }
        buf[i] = v * signs[c];
    }
    workgroupBarrier();
    for (var width = 1u; width < 128u; width *= 2u) {
        let lo = (li / width) * (2u * width) + (li % width);
        let a = buf[lo];
        let b = buf[lo + width];
        workgroupBarrier();
        buf[lo] = a + b;
        buf[lo + width] = a - b;
        workgroupBarrier();
    }
    for (var i = li; i < 128u; i += 64u) {
        y[wg.y * p.y_stride + col0 + i] = buf[i] * 0.08838834764831845;
    }
}
