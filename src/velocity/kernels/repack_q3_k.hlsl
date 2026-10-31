// Repacks file Q3_K blocks to eight independent 16-byte units, from
// repack_q3_k.wgsl; all bit positions and signed scale bytes are preserved.
uint byte_at(uint i) { return (ldw(1, i >> 2u) >> ((i & 3u) * 8u)) & 0xffu; }
uint scale3(uint sb, uint i) {
    uint k = i % 4u, g = i / 4u;
    uint low = (byte_at(sb + 4u * (g % 2u) + k) >> (4u * (g / 2u))) & 0xfu;
    uint high = (byte_at(sb + 8u + k) >> (2u * g)) & 3u;
    return low | (high << 4u);
}
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint unit = id.x + id.y * num_groups().x * 256u;
    if (unit >= param(0) * 8u) return;
    uint b = (unit / 8u) * 110u, s = unit % 8u;
    uint h = s / 4u, j = s % 4u;
    uint l0 = 0u, l1 = 0u, hw = 0u;
    for (uint l = 0u; l < 32u; l++) {
        uint q = (byte_at(b + 32u + 32u * h + l) >> (2u * j)) & 3u;
        uint high = (byte_at(b + l) >> (4u * h + j)) & 1u;
        uint e = l / 4u, by = l % 4u;
        if (e < 4u) l0 |= q << (8u * by + 2u * e);
        else l1 |= q << (8u * by + 2u * (e - 4u));
        hw |= high << (8u * by + e);
    }
    uint sc0 = (scale3(b + 96u, 8u * h + 2u * j) + 224u) & 0xffu;
    uint sc1 = (scale3(b + 96u, 8u * h + 2u * j + 1u) + 224u) & 0xffu;
    uint d = byte_at(b + 108u) | (byte_at(b + 109u) << 8u);
    rw(2).Store4(base(2) + 16u * unit, uint4(l0, l1, hw, sc0 | (sc1 << 8u) | (d << 16u)));
}
