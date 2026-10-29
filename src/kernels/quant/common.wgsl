// Shared by every quantized kernel: reads from the packed weight buffer `w`
// (array<u32>), whose blocks sit at any byte alignment (Q3_K and Q6_K
// blocks are 110 and 210 bytes). The type file that follows defines
// BLOCK_ELEMS, BLOCK_BYTES and unit(row_byte, u), the 32 weights of unit u
// of the row starting at byte row_byte, decoded as ggml's
// dequantize_row_* does.

fn wbyte(i: u32) -> u32 {
    return (w[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

fn whalf(i: u32) -> f32 {
    return unpack2x16float(wbyte(i) | (wbyte(i + 1u) << 8u)).x;
}

fn wi8(i: u32) -> i32 {
    return (i32(wbyte(i)) << 24u) >> 24u;
}

