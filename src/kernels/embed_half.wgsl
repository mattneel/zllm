// out[t] = table[tokens[t] - row0] widened to f32, for tokens whose row lies
// in this table chunk [row0, row0 + chunk_rows). One workgroup per token;
// tokens outside the chunk are left for another chunk's dispatch.

struct Params {
    dim: u32,
    row0: u32,
    chunk_rows: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let t = wg.x;
    let tok = tokens[t];
    if (tok < p.row0 || tok >= p.row0 + p.chunk_rows) {
        return;
    }
    let base = (tok - p.row0) * p.dim;
    for (var i = lid.x; i < p.dim; i += 256u) {
        let word = table[(base + i) / 2u];
        out[t * p.dim + i] = select(half_lo(word), half_hi(word), (i & 1u) == 1u);
    }
}
