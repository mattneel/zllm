// Q3_K for the decode multiply: 8 threads per (repacked) block, one unit
// each: a single 16-byte load, 32 weights. Each part starts at its own
// word, so the 8 parts read x vectors in different banks of workgroup
// memory (part p's x vectors are 8 apart, all in one bank group).
const PARTS: u32 = 8u;

fn block_dot(row_byte: u32, blk: u32, xb4: u32, p: u32) -> f32 {
    let h = head(row_byte + blk * BLOCK_BYTES, p);
    var acc = 0.0;
    for (var k = 0u; k < 8u; k++) {
        let e = (k + p) % 8u;
        // q + 4·hbit (0..7) for four positions at once.
        let c = ((h.lo[e / 4u] >> (2u * (e % 4u))) & 0x03030303u) | (((h.hi >> e) & 0x01010101u) << 2u);
        let dl = select(h.dl0, h.dl1, e >= 4u);
        acc += dot(dl * (vec4f(unpack4xU8(c)) - 4.0), xs[xb4 + 8u * p + e]);
    }
    return acc;
}
