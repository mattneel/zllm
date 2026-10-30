// In place, per value head j of token t: o ← rmsnorm(o) · w · silu(z), the
// DeltaNet output gate after delta_rule; with `sigmoid`, · sigmoid(z) instead
// (Qwen4Exp). z is read from the fused input projection: row t at t · z_row,
// offset z_off. One workgroup per (head, token).

struct Params {
    v_heads: u32,
    dim: u32,
    eps: f32,
    z_row: u32,
    z_off: u32,
    sigmoid: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<storage, read> z: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let b = lid3.x;
    let base = (wg.y * p.v_heads + wg.x) * p.dim;
    let in_head = b < p.dim;
    var v = 0.0;
    if (in_head) {
        v = o[base + b];
    }
    partial[b] = v * v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (b < s) {
            partial[b] += partial[b + s];
        }
        workgroupBarrier();
    }
    if (in_head) {
        let zz = z[wg.y * p.z_row + p.z_off + wg.x * p.dim + b];
        let r = 1.0 / sqrt(partial[0] / f32(p.dim) + p.eps);
        let gate = 1.0 / (1.0 + exp(-zz));
        o[base + b] = v * r * w[b] * select(zz * gate, gate, p.sigmoid != 0u);
    }
}
