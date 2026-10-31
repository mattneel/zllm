// Decode half-weight dot products, eight rows per group, with optional
// RMS normalization, residual add, and SiLU gate/up pairing
// (kernels/matvec_half.wgsl); half_lo/half_hi come from half/*.hlsli.
static const uint NORM = 1u;
static const uint ACCUMULATE = 2u;
static const uint SILU_PAIR = 4u;
static const uint WG = 256u;
static const uint LANES = 32u;
static const uint ROWS = WG / LANES;
static const uint MAX_STAGED = 3584u;
groupshared float partial[WG];
groupshared float partial2[WG];
groupshared float4 xs[MAX_STAGED / 4u];

float dot8(float4 x0, float4 x1, uint4 ww) {
    precise float v = x0.x * half_lo(ww.x) + x0.y * half_hi(ww.x) +
        x0.z * half_lo(ww.y) + x0.w * half_hi(ww.y) +
        x1.x * half_lo(ww.z) + x1.y * half_hi(ww.z) +
        x1.z * half_lo(ww.w) + x1.w * half_hi(ww.w);
    return v;
}

[numthreads(256, 1, 1)]
void main(uint3 wg : SV_GroupID, uint li : SV_GroupIndex) {
    const uint n = param(0), k = param(1), col0 = param(2);
    const uint out_stride = param(3), flags = param(5), half_rows = param(6);
    const uint lane = li % LANES, r = li / LANES;
    const uint j = wg.x * ROWS + r, t = wg.y;
    const uint units = k / 8u, xb = t * (k / 4u);
    const bool staged = k <= MAX_STAGED;
    if (staged) {
        precise float ss = 0.0f;
        for (uint i = li; i < k / 4u; i += WG) {
            const float4 v = asfloat(rw(1).Load4(base(1) + 16u * (xb + i)));
            xs[i] = v;
            ss += dot(v, v);
        }
        if ((flags & NORM) != 0u) {
            partial[li] = ss;
            GroupMemoryBarrierWithGroupSync();
            for (uint s = WG / 2u; s > 0u; s /= 2u) {
                if (li < s) partial[li] += partial[li + s];
                GroupMemoryBarrierWithGroupSync();
            }
            precise float scale = 1.0f / sqrt(partial[0] / float(k) + paramf(4));
            for (uint i = li; i < k / 4u; i += WG) {
                xs[i] = xs[i] * scale * asfloat(ro(4).Load4(base(4) + 16u * i));
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    const bool pair = (flags & SILU_PAIR) != 0u;
    precise float acc = 0.0f, acc2 = 0.0f;
    if (j < n) {
        const uint wb = j * units, wb2 = (half_rows + j) * units;
        for (uint u = lane; u < units; u += LANES) {
            float4 x0, x1;
            if (staged) {
                x0 = xs[2u * u];
                x1 = xs[2u * u + 1u];
            } else {
                x0 = asfloat(rw(1).Load4(base(1) + 16u * (xb + 2u * u)));
                x1 = asfloat(rw(1).Load4(base(1) + 16u * (xb + 2u * u + 1u)));
            }
            acc += dot8(x0, x1, ro(2).Load4(base(2) + 16u * (wb + u)));
            if (pair) acc2 += dot8(x0, x1, ro(2).Load4(base(2) + 16u * (wb2 + u)));
        }
    }
    partial[li] = acc;
    partial2[li] = acc2;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = LANES / 2u; s > 0u; s /= 2u) {
        if (lane < s) {
            partial[li] += partial[li + s];
            partial2[li] += partial2[li + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u && j < n) {
        const uint o = t * out_stride + col0 + j;
        precise float v = partial[li];
        if (pair) v = v / (1.0f + exp(-v)) * partial2[li];
        if ((flags & ACCUMULATE) != 0u) v += ldf(3, o);
        stf(3, o, v);
    }
}
