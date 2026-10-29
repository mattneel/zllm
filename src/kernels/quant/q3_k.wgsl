// Q3_K: blocks of 256 = hmask[32] (third bits), qs[64] (2 bits each),
// scales[12] (16 6-bit scales), d (f16). Unit s: half n = s/4, shift
// j = s%4 within it; weight l takes bits 2j of qs[32n + l], minus 4 unless
// bit 4n + j of hmask[l] is set, times d·(scale − 32) for scale
// 8n + 2j + l/16.
const BLOCK_ELEMS: u32 = 256u;
// 110 bytes in the file, padded to 112 on the GPU (kernels.blockStride), so
// hmask, qs and scales | d each start on a 16-byte boundary.
const BLOCK_BYTES: u32 = 112u;

// Scale i (0..16) of the 12 scale bytes held as three words: low 4 bits
// from bytes 0..8, high 2 bits from 8..12.
fn scale3(sw: vec3u, i: u32) -> i32 {
    let k = i % 4u;
    let g = i / 4u;
    let low = (byte12(sw, 4u * (g % 2u) + k) >> (4u * (g / 2u))) & 0xfu;
    let high = (byte12(sw, 8u + k) >> (2u * g)) & 3u;
    return i32(low | (high << 4u)) - 32;
}

struct Head {
    qs: u32,
    hm: u32,
    shift: u32,
    bit: u32,
    // d·(scale − 32) for the unit's two halves of 16.
    dl0: f32,
    dl1: f32,
}

fn head(row_byte: u32, u: u32) -> Head {
    let b = row_byte + (u / 8u) * BLOCK_BYTES;
    let s = u % 8u;
    let n = s / 4u;
    let j = s % 4u;
    let sv = w[(b + 96u) >> 4u];
    let d_all = unpack2x16float(sv.w).x;
    return Head(b + 32u + 32u * n, b, 2u * j, 4u * n + j,
                d_all * f32(scale3(sv.xyz, 8u * n + 2u * j)), d_all * f32(scale3(sv.xyz, 8u * n + 2u * j + 1u)));
}

fn quad(h: Head, i0: u32) -> array<vec4f, 4> {
    let qv = w[(h.qs + 4u * i0) >> 4u];
    let hv = w[(h.hm + 4u * i0) >> 4u];
    let dl = select(h.dl0, h.dl1, i0 >= 4u);
    var r: array<vec4f, 4>;
    for (var e = 0u; e < 4u; e++) {
        // q + 4·hbit for four positions at once, one byte each.
        let c = ((qv[e] >> h.shift) & 0x03030303u) | (((hv[e] >> h.bit) & 0x01010101u) << 2u);
        r[e] = dl * (vec4f(unpack4xU8(c)) - 4.0);
    }
    return r;
}
