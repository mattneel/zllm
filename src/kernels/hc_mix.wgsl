// The hyper-connection mix of a token's streams into the block's input:
// out[t][i] = Σ_c xn[t][c][i] · sigmoid(gate[t][c][i]) / hc, where xn is the
// streams after their norms and gate the mixer's up projection.
// Grid: x = columns in groups of 256, y = tokens.

struct Params {
    dim: u32,
    hc: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> xn: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    let i = wg.x * 256u + li;
    if (i >= p.dim) {
        return;
    }
    let base = t * p.hc * p.dim + i;
    var acc = 0.0;
    for (var c = 0u; c < p.hc; c++) {
        let g = gate[base + c * p.dim];
        acc += xn[base + c * p.dim] / (1.0 + exp(-g));
    }
    out[t * p.dim + i] = acc / f32(p.hc);
}
