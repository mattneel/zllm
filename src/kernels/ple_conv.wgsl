// The PLE block's causal dilated convolution and residual update, per
// channel over the step's tokens in order:
//
//   hidden[t][ch] += gated[t][ch] + silu(Σ_k w[ch][k] · x[t − (K−1−k)·dil][ch])
//
// where x is `normed` for positions in the step and the history (the last
// (K − 1) · dil rows of earlier steps, oldest first) before it. The history
// then advances to the step's end: slot i takes the row (K − 1)·dil − i
// before it, read before it is overwritten (ascending i reads higher slots).

struct Params {
    channels: u32,
    taps: u32,
    dilation: u32,
    _pad: u32,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    keep: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> gated: array<f32>;
@group(0) @binding(2) var<storage, read> normed: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> hist: array<f32>;
@group(0) @binding(5) var<storage, read_write> hidden: array<f32>;
@group(0) @binding(6) var<uniform> step: Step;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let ch = id.x;
    if (ch >= p.channels) {
        return;
    }
    let n = p.channels;
    let rows = (p.taps - 1u) * p.dilation;
    for (var t = 0u; t < step.m; t++) {
        var acc = 0.0;
        for (var k = 0u; k < p.taps; k++) {
            let back = (p.taps - 1u - k) * p.dilation;
            var v = 0.0;
            if (t >= back) {
                v = normed[(t - back) * n + ch];
            } else {
                v = hist[(rows - back + t) * n + ch];
            }
            acc += w[ch * p.taps + k] * v;
        }
        hidden[t * n + ch] += gated[t * n + ch] + acc / (1.0 + exp(-acc));
    }
    for (var i = 0u; i < rows; i++) {
        let j = step.m + i;
        if (j < rows) {
            hist[i * n + ch] = hist[j * n + ch];
        } else {
            hist[i * n + ch] = normed[(j - rows) * n + ch];
        }
    }
}
