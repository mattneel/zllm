// Decode attention, part 2: merges attn_split's blocks for one (query head
// h, token t) and applies the sigmoid output gate:
//
//   M = max_b m_b,  L = Σ_b l_b · e^(m_b − M),  o = Σ_b o_b · e^(m_b − M) / L
//   out = o · sigmoid(gate)
//
// One thread per output dimension.

struct Params {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    f16: u32,
    scale: f32,
    max_blocks: u32,
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
@group(0) @binding(1) var<storage, read> qg: array<f32>;
@group(0) @binding(2) var<storage, read> part: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> step: Step;

const WG: u32 = 256u;
const BLOCK: u32 = 64u;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let d = lid3.x;
    let h = wg.x;
    let t = wg.y;
    let hd = p.head_dim;
    if (d >= hd) {
        return;
    }
    let keys = step.pos0 + t + 1u;
    let blocks = (keys + BLOCK - 1u) / BLOCK;
    let base0 = (t * p.heads + h) * p.max_blocks * (hd + 2u);

    var mx = -0x1.fffffep+127f;
    for (var b = 0u; b < blocks; b++) {
        mx = max(mx, part[base0 + b * (hd + 2u)]);
    }
    var l = 0.0;
    var o = 0.0;
    for (var b = 0u; b < blocks; b++) {
        let base = base0 + b * (hd + 2u);
        let w = exp(part[base] - mx);
        l += part[base + 1u] * w;
        o += part[base + 2u + d] * w;
    }
    let gate = qg[t * 2u * p.heads * hd + 2u * h * hd + hd + d];
    out[(t * p.heads + h) * hd + d] = o / l / (1.0 + exp(-gate));
}
