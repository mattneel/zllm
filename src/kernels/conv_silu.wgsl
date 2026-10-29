// out[t][c] = silu(Σ_tap w[c][tap] · x[t − (K−1−tap)][c]) for the step's m
// tokens: the DeltaNet depthwise causal convolution. Inputs before the step
// come from hist, the last K − 1 raw inputs of the sequence so far (oldest
// first; zero for a fresh sequence), which the kernel then advances. One
// thread per channel walks the tokens in order with a sliding window.

struct Params {
    channels: u32,
    taps: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> hist: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> step: Step;

const MAX_TAPS: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let c = id.x;
    if (c >= p.channels) {
        return;
    }
    let back = p.taps - 1u;
    var win: array<f32, MAX_TAPS>;
    for (var i = 0u; i < back; i++) {
        win[i] = hist[i * p.channels + c];
    }
    for (var t = 0u; t < step.m; t++) {
        win[back] = x[t * p.channels + c];
        var sum = 0.0;
        for (var tap = 0u; tap < p.taps; tap++) {
            sum += w[c * p.taps + tap] * win[tap];
        }
        out[t * p.channels + c] = sum / (1.0 + exp(-sum));
        for (var i = 0u; i < back; i++) {
            win[i] = win[i + 1u];
        }
    }
    for (var i = 0u; i < back; i++) {
        hist[i * p.channels + c] = win[i];
    }
}
