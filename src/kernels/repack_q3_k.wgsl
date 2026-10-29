// Repacks Q3_K blocks (110 bytes: hmask[32], qs[64], scales[12], d) into 8
// self-contained 16-byte units of 32 weights each, so that a unit decodes
// from one 16-byte load (quant/q3_k.wgsl). Unit s of a block (half n = s/4,
// bit pair j = s%4) holds weights w_l = d·(scale − 32)·(q_l + 4·h_l − 4) for
// l = 0..31, as four words and the scales:
//
// - L0, L1: byte b of L0 holds q for positions b, 4 + b, 8 + b, 12 + b in
//   its bit pairs 0, 2, 4, 6 (L1: positions 16 + b, ...), so word e of the
//   unit is (L >> 2(e % 4)) & 0x03030303;
// - H: bit e of byte b is h for position 4e + b;
// - then scale − 32 for positions 0..15 and 16..31 as signed bytes, and d.
//
// One thread per unit; `blocks` blocks packed back to back in `src`.

struct Params {
    blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4u>;

fn byte_at(i: u32) -> u32 {
    return (src[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

// Scale i (0..16): low 4 bits from bytes 0..8, high 2 bits from 8..12.
fn scale3(sb: u32, i: u32) -> u32 {
    let k = i % 4u;
    let g = i / 4u;
    let low = (byte_at(sb + 4u * (g % 2u) + k) >> (4u * (g / 2u))) & 0xfu;
    let high = (byte_at(sb + 8u + k) >> (2u * g)) & 3u;
    return low | (high << 4u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) n: vec3u) {
    let unit = id.x + id.y * n.x * 256u;
    if (unit >= p.blocks * 8u) {
        return;
    }
    let b = (unit / 8u) * 110u;
    let s = unit % 8u;
    let half = s / 4u;
    let j = s % 4u;
    var l0 = 0u;
    var l1 = 0u;
    var hw = 0u;
    for (var l = 0u; l < 32u; l++) {
        let q = (byte_at(b + 32u + 32u * half + l) >> (2u * j)) & 3u;
        let h = (byte_at(b + l) >> (4u * half + j)) & 1u;
        let e = l / 4u;
        let byte = l % 4u;
        if (e < 4u) {
            l0 |= q << (8u * byte + 2u * e);
        } else {
            l1 |= q << (8u * byte + 2u * (e - 4u));
        }
        hw |= h << (8u * byte + e);
    }
    // scale − 32 is in [−32, 31]: as a signed byte, (scale − 32) & 0xff.
    let sc0 = (scale3(b + 96u, 8u * half + 2u * j) + 224u) & 0xffu;
    let sc1 = (scale3(b + 96u, 8u * half + 2u * j + 1u) + 224u) & 0xffu;
    let d = byte_at(b + 108u) | (byte_at(b + 109u) << 8u);
    dst[unit] = vec4u(l0, l1, hw, sc0 | (sc1 << 8u) | (d << 16u));
}
