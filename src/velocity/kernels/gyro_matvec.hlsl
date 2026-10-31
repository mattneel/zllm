// Routed Gyro TQ2_T expert products (kernels/gyro_matvec.wgsl): gate and up
// in one dispatch, 32 rows per group, four per wave. The codebook is staged
// in groupshared memory, each lane's previous stream byte comes from its
// neighbour (WaveReadLaneAt), and a row's sum is a wave sum. Waves are 32.

#define WG 256
#define LANES 32
#define RPG 4
#define MAX_COLS 2560
#define BLOCK_BYTES 34

groupshared float xs[MAX_COLS];
groupshared uint book[2048];

uint wbyte(uint z, uint i) {
    uint word = 0u;
    if (z == 0u) word = ldw(1, i >> 2u);
    else word = ldw(2, i >> 2u);
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}

float whalf(uint z, uint i) {
    return f16tof32(wbyte(z, i) | (wbyte(z, i + 1u) << 8u));
}

[WaveSize(32)]
[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint rows = param(0);
    const uint cols = param(1);
    const uint slot = wg.y;
    const uint z = wg.z;
    const uint lane = WaveGetLaneIndex();
    uint xrow = slot;
    if (param(3) == 0u) xrow = slot / param(2);
    for (uint i = li; i < cols; i += WG) xs[i] = ldf(3, xrow * param(4) + i);
    for (uint k = li; k < 2048u; k += WG) book[k] = ldw(6, k);
    GroupMemoryBarrierWithGroupSync();

    const uint blocks = cols / 128u;
    const uint row_bytes = blocks * BLOCK_BYTES;
    const uint expert = ldu(5, slot);
    for (uint r = 0u; r < RPG; r++) {
        const uint j = (wg.x * (WG / LANES) + li / LANES) * RPG + r;
        if (j >= rows) break;
        const uint row0 = (expert * rows + j) * row_bytes;
        precise float acc = 0.0f;
        for (uint b = 0u; b < blocks; b++) {
            const uint bo = row0 + b * BLOCK_BYTES;
            const uint mine = wbyte(z, bo + 2u + lane);
            const uint prev = WaveReadLaneAt(mine, (lane + 31u) & 31u);
            const uint h = (((prev << 8u) | mine) & 0x7fffu) * 0x9e3779b1u;
            const uint wa = book[h >> 21u];
            const uint wb = book[(h >> 10u) & 2047u];
            const uint xi = b * 128u + lane * 4u;
            precise float part = f16tof32(wa) * xs[xi] + f16tof32(wa >> 16u) * xs[xi + 1u] + f16tof32(wb) * xs[xi + 2u] + f16tof32(wb >> 16u) * xs[xi + 3u];
            acc += whalf(z, bo) * part;
        }
        const float total = WaveActiveSum(acc);
        if (lane == 0u) stf(4, slot * param(5) + z * param(6) + j, total);
    }
}
