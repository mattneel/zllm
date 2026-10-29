// Copies `blocks` quantized blocks of src_bytes each, packed back to back
// as in the GGUF file, into blocks of dst_bytes each (dst_bytes ≥ src_bytes,
// a multiple of 16), so every block starts on a 16-byte boundary and the
// decode kernels can read it with 16-byte loads. The padding is zero. One
// thread per destination word; src_bytes is even, so a source word takes
// at most two loads.

struct Params {
    blocks: u32,
    src_bytes: u32,
    dst_bytes: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) n: vec3u) {
    let i = id.x + id.y * n.x * 256u;
    let words = p.dst_bytes / 4u;
    if (i >= p.blocks * words) {
        return;
    }
    let blk = i / words;
    let off = 4u * (i % words);
    if (off >= p.src_bytes) {
        dst[i] = 0u;
        return;
    }
    let s = blk * p.src_bytes + off;
    let q = s >> 2u;
    var v = src[q];
    if ((s & 2u) != 0u) {
        v = (v >> 16u) | (src[q + 1u] << 16u);
    }
    // The block's last word may run past its end (110 is not a multiple of
    // 4): keep only its own bytes.
    let own = min(4u, p.src_bytes - off);
    if (own < 4u) {
        v &= (1u << (8u * own)) - 1u;
    }
    dst[i] = v;
}
