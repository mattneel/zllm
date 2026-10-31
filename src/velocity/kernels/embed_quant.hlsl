// Decodes chunked embedding-table rows from embed_quant.wgsl.
// quad from quant/common.wgsl is defined here, after the type's Head.
struct Quad { float4 v[4]; };
Quad quad(Head h, uint i0) {
    uint4 q = quad_i8(h, i0);
    float3 s = scales(h);
    float si = i0 >= 4u ? s.y : s.x;
    Quad r;
    for (uint e = 0u; e < 4u; e++) {
        precise float4 v = si * float4(unpack_i8(q[e])) - s.z;
        r.v[e] = v;
    }
    return r;
}
[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    uint t = wg.x, tok = ldu(1, t);
    uint dim = param(0), row0 = param(1);
    if (tok < row0 || tok >= row0 + param(2)) return;
    row_k = param(3) != 0u ? 0u : dim;
    uint row_bytes = param(3) != 0u ? param(3) : row_stride(dim);
    for (uint u = lid.x; u < dim / 32u; u += 256u) {
        Head h = head((tok - row0) * row_bytes, u);
        for (uint i0 = 0u; i0 < 8u; i0 += 4u) {
            Quad q = quad(h, i0);
            for (uint e = 0u; e < 4u; e++) {
                uint o = t * dim + 32u * u + 4u * (i0 + e);
                rw(3).Store4(base(3) + 4u * o, asuint(q.v[e]));
            }
        }
    }
}
