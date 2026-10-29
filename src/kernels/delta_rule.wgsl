// Gated delta rule over the step's m tokens, then the SiLU(z)-gated RMSNorm.
// One workgroup per value head j; thread b owns value column b of the state
// S (dim × dim, S[a][b] at state[j·dim² + a·dim + b]), so the recurrence
// needs no cross-thread reduction. Per token t, with q, k, v the head's
// slices of conv row t (q, k already L2-normalized; value head j reads key
// head j mod k_heads):
//
//   S ← e^g · S
//   δ = β · (v − Sᵀk)
//   S ← S + k δᵀ
//   o = Sᵀq / sqrt(dim)
//   out = rmsnorm(o) · w · silu(z)
//
// The state persists in its buffer across dispatches.

struct Params {
    k_heads: u32,
    v_heads: u32,
    dim: u32,
    conv_stride: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> conv: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read> z: array<f32>;
@group(0) @binding(5) var<storage, read> norm_w: array<f32>;
@group(0) @binding(6) var<storage, read_write> state: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> step: Step;

const MAX_DIM: u32 = 256u;
var<workgroup> qs: array<f32, MAX_DIM>;
var<workgroup> ks: array<f32, MAX_DIM>;
var<workgroup> partial: array<f32, MAX_DIM>;

@compute @workgroup_size(MAX_DIM)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let j = wg.x;
    let b = lid.x;
    let d = p.dim;
    let in_head = b < d;
    let key_dim = p.k_heads * d;
    let kh = j % p.k_heads;
    let s0 = j * d * d;
    let inner = p.v_heads * d;
    let scale = 1.0 / sqrt(f32(d));

    for (var t = 0u; t < step.m; t++) {
        let row = t * p.conv_stride;
        if (in_head) {
            qs[b] = conv[row + kh * d + b];
            ks[b] = conv[row + key_dim + kh * d + b];
        }
        workgroupBarrier();

        var o = 0.0;
        if (in_head) {
            let decay = exp(g[t * p.v_heads + j]);
            var stk = 0.0;
            for (var a = 0u; a < d; a++) {
                let s = state[s0 + a * d + b] * decay;
                state[s0 + a * d + b] = s;
                stk += s * ks[a];
            }
            let v = conv[row + 2u * key_dim + j * d + b];
            let delta = (v - stk) * beta[t * p.v_heads + j];
            for (var a = 0u; a < d; a++) {
                let s = state[s0 + a * d + b] + ks[a] * delta;
                state[s0 + a * d + b] = s;
                o += s * qs[a];
            }
            o *= scale;
        }

        // Gated RMSNorm over the head's d outputs.
        partial[b] = select(0.0, o * o, in_head);
        workgroupBarrier();
        for (var s = MAX_DIM / 2u; s > 0u; s /= 2u) {
            if (b < s) {
                partial[b] += partial[b + s];
            }
            workgroupBarrier();
        }
        if (in_head) {
            let zz = z[t * inner + j * d + b];
            let r = 1.0 / sqrt(partial[0] / f32(d) + p.eps);
            out[t * inner + j * d + b] = o * r * norm_w[b] * (zz / (1.0 + exp(-zz)));
        }
        workgroupBarrier();
    }
}
