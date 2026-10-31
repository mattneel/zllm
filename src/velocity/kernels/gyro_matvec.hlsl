// Routed Gyro TQ2_T expert products, with two projections and a half-pair
// codebook; eight rows per group (kernels/gyro_matvec.wgsl).

#define WG 256
#define LANES 32
#define ROWS (WG / LANES)
#define MAX_COLS 2560
#define BLOCK_BYTES 34

groupshared float xs[MAX_COLS];
groupshared float partial[WG];

uint wbyte(uint z, uint i) {
    uint word = 0u;
    if (z == 0u) word = ldw(1, i >> 2u);
    else word = ldw(2, i >> 2u);
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}

float whalf(uint z, uint i) {
    return f16tof32(wbyte(z, i) | (wbyte(z, i + 1u) << 8u));
}

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint rows = param(0);
    const uint cols = param(1);
    const uint slot = wg.y;
    const uint z = wg.z;
    const uint lane = li % LANES;
    const uint j = wg.x * ROWS + li / LANES;
    uint xrow = slot;
    if (param(3) == 0u) xrow = slot / param(2);
    for (uint i = li; i < cols; i += WG) {
        xs[i] = ldf(3, xrow * param(4) + i);
    }
    GroupMemoryBarrierWithGroupSync();

    precise float acc = 0.0f;
    if (j < rows) {
        const uint blocks = cols / 128u;
        const uint row_bytes = blocks * BLOCK_BYTES;
        const uint row0 = (ldu(5, slot) * rows + j) * row_bytes;
        const uint prev_lane = (lane + 31u) & 31u;
        for (uint b = 0u; b < blocks; b++) {
            const uint bo = row0 + b * BLOCK_BYTES;
            const uint state = (wbyte(z, bo + 2u + prev_lane) << 8u) | wbyte(z, bo + 2u + lane);
            const uint h = (state & 0x7fffu) * 0x9e3779b1u;
            const uint wa = ldw(6, h >> 21u);
            const uint wb = ldw(6, (h >> 10u) & 2047u);
            const float2 pa = float2(f16tof32(wa), f16tof32(wa >> 16u));
            const float2 pb = float2(f16tof32(wb), f16tof32(wb >> 16u));
            const uint xi = b * 128u + lane * 4u;
            precise float part = pa.x * xs[xi] + pa.y * xs[xi + 1u] + pb.x * xs[xi + 2u] + pb.y * xs[xi + 3u];
            acc += whalf(z, bo) * part;
        }
    }
    partial[li] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = LANES / 2u; s > 0u; s /= 2u) {
        if (lane < s) partial[li] += partial[li + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u && j < rows) {
        stf(4, slot * param(5) + z * param(6) + j, partial[li]);
    }
}
