// Q6_K: blocks of 256 = ql[128] (low 4 bits), qh[64] (high 2 bits),
// scales[16] (i8, one per 16), d (f16). Unit s: half n = s/4 of the block,
// quarter c = s%4 of that half; y = d·sc·(q − 32).
const BLOCK_ELEMS: u32 = 256u;
// 210 bytes in the file, padded to 224 on the GPU (kernels.blockStride), so
// ql, qh, scales and d each start on a 16-byte boundary.
const BLOCK_BYTES: u32 = 224u;

struct Head {
    ql: u32,
    qh: u32,
    low_shift: u32,
    high_shift: u32,
    // d·sc for the unit's two halves of 16.
    ds0: f32,
    ds1: f32,
}

// Signed byte j (0..16) of 16 bytes.
fn sbyte16(s: vec4u, j: u32) -> i32 {
    return (i32(s[j / 4u] >> (8u * (j % 4u))) << 24u) >> 24u;
}

fn head(row_byte: u32, u: u32) -> Head {
    let b = row_byte + (u / 8u) * BLOCK_BYTES;
    let s = u % 8u;
    let n = s / 4u;
    let c = s % 4u;
    let d = unpack2x16float(w[(b + 208u) >> 4u].x).x;
    let sv = w[(b + 192u) >> 4u];
    let si = 8u * n + 2u * c;
    return Head(b + 64u * n + 32u * (c % 2u), b + 128u + 32u * n, 4u * (c / 2u), 2u * c, d * f32(sbyte16(sv, si)), d * f32(sbyte16(sv, si + 1u)));
}

fn quad(h: Head, i0: u32) -> array<vec4f, 4> {
    let lv = w[(h.ql + 4u * i0) >> 4u];
    let hv = w[(h.qh + 4u * i0) >> 4u];
    let ds = select(h.ds0, h.ds1, i0 >= 4u);
    var r: array<vec4f, 4>;
    for (var e = 0u; e < 4u; e++) {
        let q = ((lv[e] >> h.low_shift) & 0x0f0f0f0fu) | (((hv[e] >> h.high_shift) & 0x03030303u) << 4u);
        r[e] = ds * (vec4f(unpack4xU8(q)) - 32.0);
    }
    return r;
}
