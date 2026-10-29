// DeltaNet gates, in place, for every (token, value head):
//   beta ← sigmoid(beta)
//   g    ← softplus(g + dt_bias[h]) · a[h]      (a holds −exp(A_log))
// softplus is linear above 20, as in ggml.

struct Params {
    count: u32,
    heads: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> beta: array<f32>;
@group(0) @binding(2) var<storage, read_write> g: array<f32>;
@group(0) @binding(3) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read> a: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= p.count) {
        return;
    }
    let h = i % p.heads;
    beta[i] = 1.0 / (1.0 + exp(-beta[i]));
    let x = g[i] + dt_bias[h];
    let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    g[i] = sp * a[h];
}
