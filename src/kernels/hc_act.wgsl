// lo[i] = silu(lo[i] / hc), in place: the hyper-connection mixer's low-rank
// activation, between its down and up projections.

struct Params {
    count: u32,
    hc: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> lo: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= p.count) {
        return;
    }
    let v = lo[i] / f32(p.hc);
    lo[i] = v / (1.0 + exp(-v));
}
