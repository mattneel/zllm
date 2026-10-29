// Shared by every quantized kernel: reads from the weight buffer `w`
// (array<vec4u>). Blocks are BLOCK_BYTES apart: the GGUF block size, padded
// to a multiple of 16 where needed (kernels.blockStride), so the decoders
// read 16-byte vectors. Every block size is even, so a 32-bit read takes at
// most two loads. The type file that follows defines BLOCK_ELEMS,
// BLOCK_BYTES, and for unit u (32 weights) of the row at byte row_byte:
// head(row_byte, u), its offsets and scales; quad_i8(h, i0) for i0 = 0 or 4,
// weights 4i0..4i0+15 of the unit as signed 8-bit integers q, four per
// word; and scales(h) = (s0, s1, m): weight = s·q − m, with s0 for words
// 0..3 and s1 for 4..7, as ggml's dequantize_row_* computes it. It also
// defines ACT_BLOCK, the activation block the prefill multiply rounds to
// (llama.cpp's vec_dot_type: Q8_K or Q8_0), and HAS_MIN (m ≠ 0). The decode
// multiply also takes the type's block_dot (<type>_dot.wgsl).

// Word q (4-byte index) of the buffer.
fn wword(q: u32) -> u32 {
    return w[q >> 2u][q & 3u];
}

fn wbyte(i: u32) -> u32 {
    return (wword(i >> 2u) >> ((i & 3u) * 8u)) & 0xffu;
}

fn whalf(i: u32) -> f32 {
    return unpack2x16float(wbyte(i) | (wbyte(i + 1u) << 8u)).x;
}

// The 4 bytes at even offset i, little-endian.
fn w32(i: u32) -> u32 {
    let q = i >> 2u;
    if ((i & 2u) == 0u) {
        return wword(q);
    }
    return (wword(q) >> 16u) | (wword(q + 1u) << 16u);
}

// Weights 4i0..4i0+15 of a unit as f32, value for value as ggml.
fn quad(h: Head, i0: u32) -> array<vec4f, 4> {
    let q = quad_i8(h, i0);
    let s = scales(h);
    let si = select(s.x, s.y, i0 >= 4u);
    var r: array<vec4f, 4>;
    for (var e = 0u; e < 4u; e++) {
        r[e] = si * vec4f(unpack4xI8(q[e])) - s.z;
    }
    return r;
}

// Byte j (0..12) of 12 bytes held as three words.
fn byte12(s: vec3u, j: u32) -> u32 {
    return (s[j / 4u] >> (8u * (j % 4u))) & 0xffu;
}

