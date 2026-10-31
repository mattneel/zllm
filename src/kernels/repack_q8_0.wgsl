// Lays out `rows` rows of k Q8_0 weights, as they are in the GGUF file
// (34-byte blocks: d f16, qs[32] i8), as split rows (quant/q8_0.wgsl): the
// row's k values as i8, then its k/32 scales as f16, then zero padding to a
// multiple of 16 bytes. One thread per destination word, the grid folded into
// y past 65,535 workgroups.

struct Params {
    rows: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn src_byte(i: u32) -> u32 {
    return (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

fn src_half(i: u32) -> u32 {
    return src_byte(i) | (src_byte(i + 1u) << 8u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) n: vec3u) {
    let stride = (p.k + p.k / 16u + 15u) & ~15u;
    let i = id.x + id.y * n.x * 256u;
    if (i >= p.rows * (stride / 4u)) {
        return;
    }
    let r = i / (stride / 4u);
    let o = 4u * (i % (stride / 4u));
    let raw = r * (p.k / 32u) * 34u;
    let blocks = p.k / 32u;
    var word = 0u;
    if (o < p.k) {
        // Four values of block o / 32.
        let b = raw + (o / 32u) * 34u + 2u + o % 32u;
        word = src_byte(b) | (src_byte(b + 1u) << 8u) | (src_byte(b + 2u) << 16u) | (src_byte(b + 3u) << 24u);
    } else if (o < p.k + p.k / 16u) {
        // The scales of blocks s and s + 1.
        let s = (o - p.k) / 2u;
        word = src_half(raw + s * 34u);
        if (s + 1u < blocks) {
            word |= src_half(raw + (s + 1u) * 34u) << 16u;
        }
    }
    dst[i] = word;
}
