// act[t][j] = silu(gu[t][j]) · gu[t][width + j]: the SwiGLU activation over
// the fused gate|up projection, whose rows hold the gate half then the up
// half. One thread per output element; count = rows · width.

struct Params {
    count: u32,
    width: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> gu: array<f32>;
@group(0) @binding(2) var<storage, read_write> act: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) n: vec3u) {
    let i = id.x + id.y * n.x * 256u;
    if (i >= p.count) {
        return;
    }
    let t = i / p.width;
    let j = i % p.width;
    let g = gu[t * 2u * p.width + j];
    act[i] = g / (1.0 + exp(-g)) * gu[t * 2u * p.width + p.width + j];
}
