// Adds a block's output back into each residual stream:
// res[t][c][i] += block[t][i] · 2 · sigmoid(inject[t][c] / hc).
// Grid: x = columns in groups of 256, y = tokens.

struct Params {
    dim: u32,
    hc: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> block: array<f32>;
@group(0) @binding(2) var<storage, read> inject: array<f32>;
@group(0) @binding(3) var<storage, read_write> res: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let t = wg.y;
    let i = wg.x * 256u + li;
    if (i >= p.dim) {
        return;
    }
    let b = block[t * p.dim + i];
    for (var c = 0u; c < p.hc; c++) {
        let w = 2.0 / (1.0 + exp(-inject[t * p.hc + c] / f32(p.hc)));
        res[(t * p.hc + c) * p.dim + i] += b * w;
    }
}
