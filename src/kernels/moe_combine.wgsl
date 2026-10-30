// The expert block's output (lifted from gyro-dx's moe_combine):
// out[t][i] = Σ_k weights[t][k] · down[t·n_used + k][i]
//           + sigmoid(shared_logit[t]) · shared[t][i].
// Grid: x = columns in groups of 256, y = tokens.

struct Params {
    n_used: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> down: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> shared_out: array<f32>;
@group(0) @binding(4) var<storage, read> shared_logit: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    let i = wg.x * 256u + li;
    if (i >= p.dim) {
        return;
    }
    var acc = 0.0;
    for (var k = 0u; k < p.n_used; k++) {
        acc += weights[t * p.n_used + k] * down[(t * p.n_used + k) * p.dim + i];
    }
    let g = 1.0 / (1.0 + exp(-shared_logit[t]));
    out[t * p.dim + i] = acc + g * shared_out[t * p.dim + i];
}
