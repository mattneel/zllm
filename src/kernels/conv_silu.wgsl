// out[t][c] = silu(Σ_tap w[c][tap] · x[t − (K−1−tap)][c]): the DeltaNet
// depthwise causal convolution, with zero history before token 0.
// One thread per (token, channel).

struct Params {
    tokens: u32,
    channels: u32,
    taps: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) n: vec3u) {
    let i = id.x + id.y * n.x * 256u;
    if (i >= p.tokens * p.channels) {
        return;
    }
    let t = i / p.channels;
    let c = i % p.channels;
    var sum = 0.0;
    for (var tap = 0u; tap < p.taps; tap++) {
        let back = p.taps - 1u - tap;
        if (t >= back) {
            sum += w[c * p.taps + tap] * x[(t - back) * p.channels + c];
        }
    }
    out[i] = sum / (1.0 + exp(-sum));
}
