// out[t][col0 + j] (+)= Σ_s parts[s·slice_stride + t·n + j]: adds up the
// partial sums of a quantized prefill multiply split over k
// (kernels.matmulSlices). One thread per (t, j).

struct Params {
    rows: u32,
    n: u32,
    slices: u32,
    col0: u32,
    out_stride: u32,
    accumulate: u32,
    slice_stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> parts: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) ng: vec3u) {
    let i = id.x + id.y * ng.x * 256u;
    if (i >= p.rows * p.n) {
        return;
    }
    let t = i / p.n;
    let j = i % p.n;
    var sum = 0.0;
    for (var s = 0u; s < p.slices; s++) {
        sum += parts[s * p.slice_stride + i];
    }
    let o = t * p.out_stride + p.col0 + j;
    if (p.accumulate != 0u) {
        out[o] += sum;
    } else {
        out[o] = sum;
    }
}
