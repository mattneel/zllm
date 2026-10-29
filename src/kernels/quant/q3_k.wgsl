// Q3_K: blocks of 256 = hmask[32] (third bits), qs[64] (2 bits each),
// scales[12] (16 6-bit scales), d (f16). Unit s: half n = s/4, shift
// j = s%4 within it; weight l takes bits 2j of qs[32n + l], minus 4 unless
// bit 4n + j of hmask[l] is set, times d·(scale − 32) for scale
// 8n + 2j + l/16.
//
// On the GPU the blocks are repacked at load (repack_q3_k.wgsl, 128 bytes
// each): every unit is one 16-byte vector holding its 2-bit values (two
// words), its high bits (one word), its two scales − 32 as signed bytes and
// d, so a unit decodes from a single load.
const BLOCK_ELEMS: u32 = 256u;
const BLOCK_BYTES: u32 = 128u;

const ACT_BLOCK: u32 = 256u;
const HAS_MIN: bool = false;

struct Head {
    lo: vec2u,
    hi: u32,
    // d·(scale − 32) for the unit's two halves of 16.
    dl0: f32,
    dl1: f32,
}

fn head(row_byte: u32, u: u32) -> Head {
    let v = w[(row_byte + 16u * u) >> 4u];
    let d = unpack2x16float(v.w >> 16u).x;
    let sc0 = (i32(v.w) << 24u) >> 24u;
    let sc1 = (i32(v.w) << 16u) >> 24u;
    return Head(v.xy, v.z, d * f32(sc0), d * f32(sc1));
}

fn scales(h: Head) -> vec3f {
    return vec3f(h.dl0, h.dl1, 0.0);
}

fn quad_i8(h: Head, i0: u32) -> vec4u {
    let l = h.lo[i0 / 4u];
    var r: vec4u;
    for (var e = 0u; e < 4u; e++) {
        // q + 4·hbit for four positions at once, one byte each (0..7), then
        // minus 4 per byte without borrows.
        let c = ((l >> (2u * e)) & 0x03030303u) | (((h.hi >> (i0 + e)) & 0x01010101u) << 2u);
        r[e] = (c + 0x7c7c7c7cu) ^ 0x80808080u;
    }
    return r;
}
