// Widen the token's half-weight embedding row in the matching table chunk
// (kernels/embed_half.wgsl); half_lo/half_hi come from half/*.hlsli.
[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    const uint dim = param(0), row0 = param(1), chunk_rows = param(2);
    const uint t = wg.x, tok = ldu(1, t);
    if (tok < row0 || tok >= row0 + chunk_rows) return;
    const uint row_base = (tok - row0) * dim;
    for (uint i = lid.x; i < dim; i += 256u) {
        const uint word = ldw(2, (row_base + i) / 2u);
        stf(3, t * dim + i, (i & 1u) == 1u ? half_hi(word) : half_lo(word));
    }
}
