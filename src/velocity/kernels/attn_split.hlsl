// Block-local attention softmax and weighted values for each KV head, block,
// and query row (kernels/attn_split.wgsl). MAX_GROUP and included are prefixed.
#define WG 256
#define BLOCK 64
#define SPLIT (WG / BLOCK)
#define MAX_HEAD_DIM 256

groupshared precise float qs[MAX_GROUP * MAX_HEAD_DIM];
groupshared precise float red[WG];
groupshared precise float sc[MAX_GROUP * BLOCK];

float2 k_pair(uint i) {
    if (param(3) != 0u) {
        const uint bits = ldu(2, i);
        return float2(f16tof32(bits & 0xffffu), f16tof32(bits >> 16u));
    }
    return float2(ldf(2, 2u * i), ldf(2, 2u * i + 1u));
}

float v_at(uint i) {
    if (param(3) != 0u) {
        const uint bits = ldu(3, i / 2u);
        const float2 pair = float2(f16tof32(bits & 0xffffu), f16tof32(bits >> 16u));
        return (i & 1u) == 1u ? pair.y : pair.x;
    }
    return ldf(3, i);
}

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint lid = lid3.x;
    const uint g = wg.x;
    const uint b = wg.y;
    const uint t = wg.z;
    const uint row = param(7) + t;
    const uint hd = param(2);
    const uint group = param(0) / param(1);
    const uint keys = ldu(5, 1) + row + 1u;
    const uint k0 = b * BLOCK;
    const uint kv_stride = param(1) * hd;
    const uint q_stride = param(6);
    const uint q_head = param(8) != 0u ? 2u * hd : hd;

    for (uint i = lid; i < group * hd; i += WG) {
        const uint h = g * group + i / hd;
        qs[i] = ldf(1, row * q_stride + h * q_head + i % hd);
    }
    GroupMemoryBarrierWithGroupSync();

    const uint j = lid / SPLIT;
    const uint s = lid % SPLIT;
    const uint key = k0 + j;
    const bool live = key < keys && included(row, key);
    const uint span = hd / SPLIT;
    precise float dots[MAX_GROUP];
    for (uint h = 0u; h < MAX_GROUP; h++) dots[h] = 0.0f;
    if (live) {
        const uint kb = (key * kv_stride + g * hd + s * span) / 2u;
        for (uint d = 0u; d < span / 2u; d++) {
            const float2 kk = k_pair(kb + d);
            const uint e = s * span + 2u * d;
            for (uint h = 0u; h < group; h++) {
                dots[h] += qs[h * hd + e] * kk.x + qs[h * hd + e + 1u] * kk.y;
            }
        }
    }
    for (uint h = 0u; h < group; h++) {
        red[lid] = dots[h];
        GroupMemoryBarrierWithGroupSync();
        if (s == 0u) {
            precise float total = red[lid] + red[lid + 1u] + red[lid + 2u] + red[lid + 3u];
            sc[h * BLOCK + j] = live ? total * paramf(4) : -asfloat(0x7f7fffffu);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid < group) {
        const uint n = min(BLOCK, keys - min(keys, k0));
        precise float mx = -asfloat(0x7f7fffffu);
        for (uint i = 0u; i < n; i++) mx = max(mx, sc[lid * BLOCK + i]);
        precise float sum = 0.0f;
        for (uint i = 0u; i < n; i++) {
            precise float score = sc[lid * BLOCK + i];
            precise float e = score == -asfloat(0x7f7fffffu) ? 0.0f : exp(score - mx);
            sc[lid * BLOCK + i] = e;
            sum += e;
        }
        for (uint i = n; i < BLOCK; i++) sc[lid * BLOCK + i] = 0.0f;
        red[2u * lid] = mx;
        red[2u * lid + 1u] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    const uint n = min(BLOCK, keys - min(keys, k0));
    for (uint d = lid; d < hd; d += WG) {
        precise float acc[MAX_GROUP];
        for (uint h = 0u; h < MAX_GROUP; h++) acc[h] = 0.0f;
        for (uint i = 0u; i < n; i++) {
            const float v = v_at((k0 + i) * kv_stride + g * hd + d);
            for (uint h = 0u; h < group; h++) acc[h] += sc[h * BLOCK + i] * v;
        }
        for (uint h = 0u; h < group; h++) {
            const uint head = g * group + h;
            const uint offset = ((t * param(0) + head) * param(5) + b) * (hd + 2u);
            stf(4, offset + 2u + d, acc[h]);
            if (d == 0u) {
                stf(4, offset, red[2u * h]);
                stf(4, offset + 1u, red[2u * h + 1u]);
            }
        }
    }
}
