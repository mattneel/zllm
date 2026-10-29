// gate[i] = silu(gate[i]) · up[i]: the SwiGLU activation, in place.

struct Params {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) n: vec3u) {
    let i = id.x + id.y * n.x * 256u;
    if (i >= p.count) {
        return;
    }
    let g = gate[i];
    gate[i] = g / (1.0 + exp(-g)) * up[i];
}
