// The DeltaNet front end for the step's m tokens, from the fused input
// projection (row t holds qkv at 0, then z, then β at beta_off and α at
// alpha_off; `row` floats per token). Workgroup (c, t):
//
//   c < groups   channels [c·head, (c+1)·head) of token t: the depthwise
//                causal conv (taps K) and SiLU,
//                  out[t][ch] = silu(Σ_tap w[ch][tap] · x[t − (K−1−tap)][ch])
//                then, for the q and k channels (ch < key_channels), the L2
//                norm over the group (one head)
//   c = groups   the gates of token t for every value head j:
//                  β = sigmoid(β), g = softplus(α + dt_bias[j]) · a[j]
//
// Inputs before the step come from hist, the last K − 1 raw inputs of the
// sequence (oldest first; zero for a fresh sequence). Tokens are independent
// here; delta_rule, which runs next and does not read hist, advances it.

struct Params {
    channels: u32,
    taps: u32,
    key_channels: u32,
    head: u32,
    row: u32,
    v_heads: u32,
    beta_off: u32,
    alpha_off: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> proj: array<f32>;
@group(0) @binding(2) var<storage, read> conv_w: array<f32>;
@group(0) @binding(3) var<storage, read> hist: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(6) var<storage, read> a: array<f32>;
@group(0) @binding(7) var<storage, read_write> g: array<f32>;
@group(0) @binding(8) var<storage, read_write> beta: array<f32>;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let i = lid3.x;
    let c = wg.x;
    let t = wg.y;
    let groups = p.channels / p.head;

    if (c == groups) {
        if (i < p.v_heads) {
            let k = t * p.v_heads + i;
            beta[k] = 1.0 / (1.0 + exp(-proj[t * p.row + p.beta_off + i]));
            let x = proj[t * p.row + p.alpha_off + i] + dt_bias[i];
            let sp = select(log(1.0 + exp(x)), x, x > 20.0);
            g[k] = sp * a[i];
        }
        return;
    }

    let ch = c * p.head + i;
    let on = i < p.head;
    var y = 0.0;
    if (on) {
        var sum = 0.0;
        for (var tap = 0u; tap < p.taps; tap++) {
            let back = p.taps - 1u - tap;
            var xv: f32;
            if (t >= back) {
                xv = proj[(t - back) * p.row + ch];
            } else {
                // Position t − back is in the history: slot taps − 1 − (back − t).
                xv = hist[(p.taps - 1u - (back - t)) * p.channels + ch];
            }
            sum += conv_w[ch * p.taps + tap] * xv;
        }
        y = sum / (1.0 + exp(-sum));
    }

    if (c * p.head < p.key_channels) {
        partial[i] = y * y;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s /= 2u) {
            if (i < s) {
                partial[i] += partial[i + s];
            }
            workgroupBarrier();
        }
        y = y / sqrt(partial[0] + p.eps);
    }
    if (on) {
        out[t * p.channels + ch] = y;
    }
}
