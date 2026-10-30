// out[t] = table[tokens[t] - row0] decoded to f32 (the type file before
// this one decodes it), for tokens whose row lies in this table chunk
// [row0, row0 + chunk_rows). One workgroup per token; a thread per unit.
// Rows are `stride` bytes apart, or packed when it is 0 (rows read from a
// file in pieces the GPU can take, multiples of 4 bytes, sit further apart).

struct Params {
    dim: u32,
    row0: u32,
    chunk_rows: u32,
    stride: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> w: array<vec4u>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let t = wg.x;
    let tok = tokens[t];
    if (tok < p.row0 || tok >= p.row0 + p.chunk_rows) {
        return;
    }
    let row_bytes = select((p.dim / BLOCK_ELEMS) * BLOCK_BYTES, p.stride, p.stride != 0u);
    for (var u = lid.x; u < p.dim / 32u; u += 256u) {
        let h = head((tok - p.row0) * row_bytes, u);
        for (var i0 = 0u; i0 < 8u; i0 += 4u) {
            let q = quad(h, i0);
            for (var e = 0u; e < 4u; e++) {
                let o = t * p.dim + 32u * u + 4u * (i0 + e);
                out[o] = q[e].x;
                out[o + 1u] = q[e].y;
                out[o + 2u] = q[e].z;
                out[o + 3u] = q[e].w;
            }
        }
    }
}
