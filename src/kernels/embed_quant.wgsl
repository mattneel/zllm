// out[t] = table[tokens[t] - row0] decoded to f32 (the type file before
// this one decodes it), for tokens whose row lies in this table chunk
// [row0, row0 + chunk_rows). One workgroup per token; a thread per unit.

struct Params {
    dim: u32,
    row0: u32,
    chunk_rows: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let t = wg.x;
    let tok = tokens[t];
    if (tok < p.row0 || tok >= p.row0 + p.chunk_rows) {
        return;
    }
    let row_bytes = (p.dim / BLOCK_ELEMS) * BLOCK_BYTES;
    for (var u = lid.x; u < p.dim / 32u; u += 256u) {
        let wv = unit((tok - p.row0) * row_bytes, u);
        for (var l = 0u; l < 32u; l++) {
            out[t * p.dim + 32u * u + l] = wv[l];
        }
    }
}
