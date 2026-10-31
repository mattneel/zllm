// Grouped TQ2_T expert matrix multiply; gyro_matmul.wgsl's exact contract.
// Four tokens share decoded weights; scalar arithmetic and explicit 32-lane
// reduction do not depend on tile occupancy or hardware wave reduction.
#define WG 256
#define LANES 32
#define RPG 4
#define BLOCK_BYTES 34

groupshared float xs[512];
groupshared float4 partial[WG];
groupshared uint book[2048];

uint wbyte(uint z, uint i) {
    const uint word = z == 0u ? ldw(1, i >> 2u) : ldw(2, i >> 2u);
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}
float whalf(uint z, uint i) {
    return f16tof32((z == 0u ? ldw(1, i >> 2u) : ldw(2, i >> 2u)) >> ((i & 2u) * 8u));
}
float product(float2 pa, float2 pb, uint xi) {
    precise float part = pa.x * xs[xi] + pa.y * xs[xi + 1u] + pb.x * xs[xi + 2u] + pb.y * xs[xi + 3u];
    return part;
}

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint n = ldu(5, wg.y);
    const uint lane = li % LANES;
    const uint rows = param(0);
    const uint blocks = param(1) / 128u;
    const uint row_bytes = blocks * BLOCK_BYTES;
    const uint prev_lane = (lane + 31u) & 31u;
    for (uint k = li; k < 2048u; k += WG) book[k] = ldw(7, k);
    GroupMemoryBarrierWithGroupSync();
    for (uint tile = 0u; tile < n; tile += 4u) {
        for (uint r = 0u; r < RPG; r++) {
            const uint j = (wg.x * (WG / LANES) + li / LANES) * RPG + r;
            precise float a0 = 0.0f;
            precise float a1 = 0.0f;
            precise float a2 = 0.0f;
            precise float a3 = 0.0f;
            for (uint b = 0u; b < blocks; b++) {
                for (uint i = li; i < 512u; i += WG) {
                    const uint t = i / 128u;
                    float v = 0.0f;
                    if (tile + t < n) {
                        const uint slot = ldu(6, wg.y * param(7) + tile + t);
                        const uint xrow = param(3) == 0u ? slot / param(2) : slot;
                        v = ldf(3, xrow * param(4) + b * 128u + i % 128u);
                    }
                    xs[i] = v;
                }
                GroupMemoryBarrierWithGroupSync();
                if (j < rows) {
                    const uint bo = (wg.y * rows + j) * row_bytes + b * BLOCK_BYTES;
                    const uint state = (wbyte(wg.z, bo + 2u + prev_lane) << 8u) | wbyte(wg.z, bo + 2u + lane);
                    const uint h = (state & 0x7fffu) * 0x9e3779b1u;
                    const uint wa = book[h >> 21u];
                    const uint wb = book[(h >> 10u) & 2047u];
                    const float2 pa = float2(f16tof32(wa), f16tof32(wa >> 16u));
                    const float2 pb = float2(f16tof32(wb), f16tof32(wb >> 16u));
                    const float scale = whalf(wg.z, bo);
                    const uint xi = lane * 4u;
                    a0 += scale * product(pa, pb, xi);
                    a1 += scale * product(pa, pb, 128u + xi);
                    a2 += scale * product(pa, pb, 256u + xi);
                    a3 += scale * product(pa, pb, 384u + xi);
                }
                GroupMemoryBarrierWithGroupSync();
            }
            partial[li] = float4(a0, a1, a2, a3);
            GroupMemoryBarrierWithGroupSync();
            for (uint s = LANES / 2u; s > 0u; s /= 2u) {
                if (lane < s) {
                    precise float4 sum = partial[li] + partial[li + s];
                    partial[li] = sum;
                }
                GroupMemoryBarrierWithGroupSync();
            }
            if (lane == 0u && j < rows) {
                for (uint t = 0u; t < 4u; t++) {
                    if (tile + t < n) {
                        const uint slot = ldu(6, wg.y * param(7) + tile + t);
                        stf(4, slot * param(5) + wg.z * param(6) + j, partial[li][t]);
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
