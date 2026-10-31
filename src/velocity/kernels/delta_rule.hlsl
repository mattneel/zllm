// Gated DeltaNet state recurrence and output, with kept-row snapshots and
// causal convolution history advancement (kernels/delta_rule.wgsl).

#define WG 256
#define COLS 32
#define R (WG / COLS)
#define MAX_DIM 256
#define MAX_ROWS (MAX_DIM / R)
groupshared float qs[MAX_DIM];
groupshared float ks[MAX_DIM];
groupshared float red[WG];
groupshared float delta[COLS];

void advance_history(uint id, uint threads) {
    const uint conv_stride = param(3);
    const uint back = param(4) - 1u;
    const uint row = param(5);
    const uint kept_hist = param(6);
    const uint m = ldu(6, 0);
    const uint keep = ldu(6, 3);
    for (uint c = id; c < conv_stride; c += threads) {
        if (keep != 0u) {
            for (uint i = 0u; i < back; i++) {
                const uint j = keep + i;
                // Copy words, preserving the history's exact float bits.
                const uint v = j < back ? ldu(8, j * conv_stride + c)
                    : ldu(7, (j - back) * row + c);
                stu(9, kept_hist + i * conv_stride + c, v);
            }
        }
        for (uint i = 0u; i < back; i++) {
            const uint j = m + i;
            const uint v = j < back ? ldu(8, j * conv_stride + c)
                : ldu(7, (j - back) * row + c);
            stu(8, i * conv_stride + c, v);
        }
    }
}

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint k_heads = param(0);
    const uint v_heads = param(1);
    const uint d = param(2);
    const uint conv_stride = param(3);
    const uint m = ldu(6, 0);
    const uint keep = ldu(6, 3);
    const uint lid = lid3.x;
    const uint3 nwg = num_groups();
    advance_history((wg.y * nwg.x + wg.x) * WG + lid, nwg.x * nwg.y * WG);
    const uint j = wg.x;
    const uint col = lid % COLS;
    const uint r = lid / COLS;
    const uint b = wg.y * COLS + col;
    const bool in_head = b < d;
    const uint rows = d / R;
    const uint a0 = r * rows;
    const uint key_dim = k_heads * d;
    const uint kh = j % k_heads;
    const uint s0 = j * d * d;
    precise float scale = 1.0f / sqrt(float(d));
    precise float s[MAX_ROWS];
    for (uint i = 0u; i < rows; i++) {
        s[i] = in_head ? ldf(4, s0 + (a0 + i) * d + b) : 0.0f;
    }
    for (uint t = 0u; t < m; t++) {
        const uint row = t * conv_stride;
        for (uint i = lid; i < d; i += WG) {
            qs[i] = ldf(1, row + kh * d + i);
            ks[i] = ldf(1, row + key_dim + kh * d + i);
        }
        GroupMemoryBarrierWithGroupSync();
        precise float decay = exp(ldf(2, t * v_heads + j));
        precise float stk = 0.0f;
        for (uint i = 0u; i < rows; i++) {
            s[i] *= decay;
            stk += s[i] * ks[a0 + i];
        }
        red[lid] = stk;
        GroupMemoryBarrierWithGroupSync();
        if (r == 0u) {
            precise float total = 0.0f;
            for (uint q = 0u; q < R; q++) total += red[q * COLS + col];
            const float v = in_head ? ldf(1, row + 2u * key_dim + j * d + b) : 0.0f;
            precise float dl = (v - total) * ldf(3, t * v_heads + j);
            delta[col] = dl;
        }
        GroupMemoryBarrierWithGroupSync();
        const float dl = delta[col];
        precise float o = 0.0f;
        for (uint i = 0u; i < rows; i++) {
            s[i] += ks[a0 + i] * dl;
            o += s[i] * qs[a0 + i];
        }
        red[lid] = o;
        GroupMemoryBarrierWithGroupSync();
        if (r == 0u && in_head) {
            precise float total = 0.0f;
            for (uint q = 0u; q < R; q++) total += red[q * COLS + col];
            stf(5, t * v_heads * d + j * d + b, total * scale);
        }
        if (t + 1u == keep && in_head) {
            for (uint i = 0u; i < rows; i++) stf(9, s0 + (a0 + i) * d + b, s[i]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (in_head) {
        for (uint i = 0u; i < rows; i++) stf(4, s0 + (a0 + i) * d + b, s[i]);
    }
}
