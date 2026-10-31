// Stable original routed slots per expert. Capacity affects storage only;
// step.m is the active token count. Grid: ceil(n_expert / 64), 1, 1.
struct Params {
    n_used: u32,
    n_expert: u32,
    max_slots: u32,
    _pad: u32,
}
struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> count: array<u32>;
@group(0) @binding(3) var<storage, read_write> slots: array<u32>;
@group(0) @binding(4) var<uniform> step: Step;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let expert = gid.x;
    if (expert >= p.n_expert) { return; }
    var n = 0u;
    for (var slot = 0u; slot < step.m * p.n_used; slot++) {
        if (ids[slot] == expert) {
            slots[expert * p.max_slots + n] = slot;
            n++;
        }
    }
    count[expert] = n;
}
