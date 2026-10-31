// Phased vLLM-order GPU sampling: penalties, greedy, min-p, radix top-k,
// top-p and token-order inverse CDF (kernels/sample.wgsl). The RNG draw is
// supplied as step.u, exactly as in WGSL; no shader-local RNG is used.

#define WG 256
#define PER 4
#define SPAN (WG * PER)
#define CAP 1024
static const uint NONE = 0xffffffffu;
static const float LOWEST = -3.4028234663852886e+38f;
static const uint PREP = 0u, REDUCE = 1u, HIST = 2u, DIGIT = 3u;
static const uint GATHER = 4u, FINAL = 5u, MASS = 6u, PICK = 7u;
static const uint BISECT_INIT = 8u, BISECT_MASS = 9u, BISECT_STEP = 10u, UNBUMP = 11u;
static const uint S_TOKEN = 0u, S_STOP = 1u, S_TOP = 2u, S_ARGMAX = 3u;
static const uint S_TAU = 4u, S_PREFIX = 5u, S_LEFT = 6u, S_LO = 7u;
static const uint S_HI = 8u, S_NCAND = 9u, S_GOAL = 10u, S_SKIP = 11u;
static const uint S_HIST = 16u, S_PART = S_HIST + 256u;

// Params are binding 9, not the phase uniform at binding 0. Stops start
// at word 12: array<vec4u, 4> is 16-byte aligned in both layouts.
#define p_vocab ldu(9, 0)
#define p_n_bias ldu(9, 1)
#define p_n_stop ldu(9, 2)
#define p_greedy ldu(9, 3)
#define p_temperature ldf(9, 4)
#define p_top_k ldu(9, 5)
#define p_top_p ldf(9, 6)
#define p_min_p ldf(9, 7)
#define p_repetition ldf(9, 8)
#define p_presence ldf(9, 9)
#define p_frequency ldf(9, 10)
#define step_u ldf(8, 2)

groupshared float rf[WG];
groupshared uint ru[WG];
groupshared uint wh[256];
groupshared uint sk[CAP];
groupshared uint si[CAP];
groupshared uint flag;
groupshared uint pick_group;
groupshared float pick_base;

uint ld(uint i) {
    uint v;
    rw(6).InterlockedOr(base(6) + 4u * i, 0u, v);
    return v;
}
void sto(uint i, uint v) {
    uint old;
    rw(6).InterlockedExchange(base(6) + 4u * i, v, old);
}
uint groups() { return (p_vocab + SPAN - 1u) / SPAN; }
uint cand_base() { return S_PART + 2u * groups(); }
uint key(float x) {
    const uint b = asuint(x);
    return (b >> 31u) == 1u ? ~b : b ^ 0x80000000u;
}
float top() { return asfloat(ld(S_TOP)); }
bool kept(float v, float t, uint tau) {
    precise float difference = v - t;
    return key(v) >= tau && (p_min_p <= 0.0f || exp(difference) >= p_min_p);
}
float2 bias_of(uint i) {
    uint lo = 0u;
    uint hi = p_n_bias;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2u;
        const uint2 b = uint2(ldu(2, 2u * mid), ldu(2, 2u * mid + 1u));
        if (b.x == i) return float2(1.0f, asfloat(b.y));
        if (b.x < i) lo = mid + 1u;
        else hi = mid;
    }
    return float2(0.0f, 0.0f);
}
void finish(uint token) {
    uint stop = 0u;
    for (uint i = 0u; i < p_n_stop; i++) {
        if (ldu(9, 12u + i) == token) stop = 1u;
    }
    sto(S_TOKEN, token);
    sto(S_STOP, stop);
    stu(7, 0, token);
    stu(4, token, ldu(4, token) + 1u);
}
uint argmax(uint lid, float v, uint i) {
    rf[lid] = v;
    ru[lid] = i;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            const float ov = rf[lid + s];
            const uint oi = ru[lid + s];
            if (ov > rf[lid] || (ov == rf[lid] && oi < ru[lid])) {
                rf[lid] = ov;
                ru[lid] = oi;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    return ru[0];
}
float group_mass(uint lid, uint g, float t, uint tau) {
    const uint i0 = g * SPAN + lid * PER;
    precise float s = 0.0f;
    for (uint j = 0u; j < PER; j++) {
        const uint i = i0 + j;
        if (i < p_vocab) {
            const float v = ldf(5, i);
            if (kept(v, t, tau)) s += exp(v - t);
        }
    }
    rf[lid] = s;
    GroupMemoryBarrierWithGroupSync();
    precise float total = 0.0f;
    if (lid == 0u) {
        for (uint k = 0u; k < WG; k++) total += rf[k];
    }
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[numthreads(WG, 1, 1)]
void main(uint3 wg : SV_GroupID, uint3 lid3 : SV_GroupThreadID) {
    const uint lid = lid3.x;
    const uint g = wg.x;
    const uint phase = param(0);
    switch (phase) {
        case PREP: {
            float best = LOWEST;
            uint best_i = NONE;
            for (uint j = 0u; j < PER; j++) {
                const uint i = g * SPAN + j * WG + lid;
                if (i < p_vocab) {
                    precise float v = ldf(1, i);
                    if (p_n_bias > 0u) {
                        const float2 b = bias_of(i);
                        if (b.x != 0.0f) v += b.y;
                    }
                    const uint n = ldu(4, i);
                    if (ldu(3, i) != 0u || n != 0u) {
                        v = v > 0.0f ? v / p_repetition : v * p_repetition;
                    }
                    v -= p_frequency * float(n);
                    v -= p_presence * (n != 0u ? 1.0f : 0.0f);
                    if (p_greedy == 0u) v = v / p_temperature;
                    stf(5, i, v);
                    if (v > best || (v == best && i < best_i)) {
                        best = v;
                        best_i = i;
                    }
                }
            }
            const uint bi = argmax(lid, best, best_i);
            if (lid == 0u) {
                sto(S_PART + 2u * g, asuint(rf[0]));
                sto(S_PART + 2u * g + 1u, bi);
            }
            break;
        }
        case REDUCE: {
            float best = LOWEST;
            uint best_i = NONE;
            for (uint w = lid; w < groups(); w += WG) {
                const float v = asfloat(ld(S_PART + 2u * w));
                const uint i = ld(S_PART + 2u * w + 1u);
                if (v > best || (v == best && i < best_i)) {
                    best = v;
                    best_i = i;
                }
            }
            const uint bi = argmax(lid, best, best_i);
            sto(S_HIST + lid, 0u);
            if (lid == 0u) {
                if (p_greedy != 0u) finish(bi);
                else {
                    sto(S_TOP, asuint(rf[0]));
                    sto(S_ARGMAX, bi);
                    sto(S_TAU, 0u);
                    sto(S_PREFIX, 0u);
                    sto(S_LEFT, p_top_k);
                    sto(S_NCAND, 0u);
                    sto(S_SKIP, 0u);
                }
            }
            break;
        }
        case HIST: {
            if (lid == 0u) flag = ld(S_SKIP);
            GroupMemoryBarrierWithGroupSync(); // workgroupUniformLoad(flag)
            if (flag != 0u) return;
            uint old_hist;
            InterlockedExchange(wh[lid], 0u, old_hist);
            GroupMemoryBarrierWithGroupSync();
            const float t = top();
            const uint r = param(1);
            const uint prefix = ld(S_PREFIX);
            for (uint j = 0u; j < PER; j++) {
                const uint i = g * SPAN + j * WG + lid;
                if (i < p_vocab) {
                    const float v = ldf(5, i);
                    if (kept(v, t, 0u)) {
                        const uint k = key(v);
                        bool match_ = true;
                        if (r > 0u) match_ = (k >> (32u - 8u * r)) == prefix;
                        if (match_) InterlockedAdd(wh[(k >> (24u - 8u * r)) & 255u], 1u);
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            uint c;
            InterlockedOr(wh[lid], 0u, c);
            if (c > 0u) {
                uint old;
                rw(6).InterlockedAdd(base(6) + 4u * (S_HIST + lid), c, old);
            }
            break;
        }
        case DIGIT: {
            if (lid == 0u && ld(S_SKIP) == 0u) {
                const uint r = param(1);
                uint left = ld(S_LEFT);
                uint total = 0u;
                if (r == 0u) {
                    for (uint d = 0u; d < 256u; d++) total += ld(S_HIST + d);
                }
                if (r == 0u && total <= left) sto(S_SKIP, 1u);
                else {
                    uint above = 0u;
                    uint d = 255u;
                    for (;;) {
                        const uint c = ld(S_HIST + d);
                        if (above + c >= left || d == 0u) break;
                        above += c;
                        d -= 1u;
                    }
                    const uint prefix = (ld(S_PREFIX) << 8u) | d;
                    sto(S_PREFIX, prefix);
                    sto(S_LEFT, left - above);
                    if (r == 3u) sto(S_TAU, prefix);
                }
            }
            GroupMemoryBarrierWithGroupSync();
            sto(S_HIST + lid, 0u);
            break;
        }
        case GATHER: {
            const float t = top();
            const uint tau = ld(S_TAU);
            const uint cb = cand_base();
            for (uint j = 0u; j < PER; j++) {
                const uint i = g * SPAN + j * WG + lid;
                if (i < p_vocab) {
                    const float v = ldf(5, i);
                    if (kept(v, t, tau)) {
                        uint slot;
                        rw(6).InterlockedAdd(base(6) + 4u * S_NCAND, 1u, slot);
                        if (slot < CAP) {
                            sto(cb + 2u * slot, i);
                            sto(cb + 2u * slot + 1u, asuint(v));
                        }
                    }
                }
            }
            break;
        }
        case FINAL: {
            const uint n = min(ld(S_NCAND), CAP);
            const uint cb = cand_base();
            for (uint j = lid; j < CAP; j += WG) {
                if (j < n) {
                    si[j] = ld(cb + 2u * j);
                    sk[j] = key(asfloat(ld(cb + 2u * j + 1u)));
                } else {
                    si[j] = NONE;
                    sk[j] = 0u;
                }
            }
            GroupMemoryBarrierWithGroupSync();
            // Bitonic key-descending, token-ascending sort.
            for (uint size = 2u; size <= CAP; size *= 2u) {
                for (uint stride = size / 2u; stride > 0u; stride /= 2u) {
                    for (uint q = lid; q < CAP / 2u; q += WG) {
                        const uint a = 2u * stride * (q / stride) + q % stride;
                        const uint b = a + stride;
                        const bool up = (a & size) == 0u;
                        const bool a_first = sk[a] > sk[b] || (sk[a] == sk[b] && si[a] <= si[b]);
                        if (a_first != up) {
                            const uint tk = sk[a];
                            sk[a] = sk[b];
                            sk[b] = tk;
                            const uint ti = si[a];
                            si[a] = si[b];
                            si[b] = ti;
                        }
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
            }
            if (lid == 0u) {
                const float t = top();
                uint kept_n = n;
                if (p_top_p < 1.0f) {
                    precise float z = 0.0f;
                    for (uint j = 0u; j < n; j++) z += exp(ldf(5, si[j]) - t);
                    precise float above = 0.0f;
                    uint last = 0u;
                    for (uint j = 0u; j < n; j++) {
                        if (above >= p_top_p * z) break;
                        last = j;
                        above += exp(ldf(5, si[j]) - t);
                    }
                    kept_n = last + 1u;
                    while (kept_n < n && sk[kept_n] == sk[last]) kept_n++;
                }
                flag = kept_n;
            }
            GroupMemoryBarrierWithGroupSync();
            GroupMemoryBarrierWithGroupSync(); // workgroupUniformLoad(flag)
            const uint kept_n = flag;
            for (uint j = lid; j < CAP; j += WG) {
                if (j >= kept_n) si[j] = NONE;
            }
            GroupMemoryBarrierWithGroupSync();
            // Token-id order for the inverse CDF.
            for (uint size = 2u; size <= CAP; size *= 2u) {
                for (uint stride = size / 2u; stride > 0u; stride /= 2u) {
                    for (uint q = lid; q < CAP / 2u; q += WG) {
                        const uint a = 2u * stride * (q / stride) + q % stride;
                        const uint b = a + stride;
                        const bool up = (a & size) == 0u;
                        if ((si[a] <= si[b]) != up) {
                            const uint ti = si[a];
                            si[a] = si[b];
                            si[b] = ti;
                        }
                    }
                    GroupMemoryBarrierWithGroupSync();
                }
            }
            if (lid == 0u) {
                const float t = top();
                precise float z = 0.0f;
                for (uint j = 0u; j < kept_n; j++) z += exp(ldf(5, si[j]) - t);
                precise float at = step_u * z;
                precise float c = 0.0f;
                uint token = si[kept_n - 1u];
                for (uint j = 0u; j < kept_n; j++) {
                    c += exp(ldf(5, si[j]) - t);
                    if (c > at) {
                        token = si[j];
                        break;
                    }
                }
                finish(token);
            }
            break;
        }
        case MASS:
        case BISECT_MASS: {
            uint tau = ld(S_TAU);
            if (phase == BISECT_MASS) {
                if (lid == 0u) flag = ld(S_HI) - ld(S_LO) <= 1u ? 1u : 0u;
                GroupMemoryBarrierWithGroupSync(); // workgroupUniformLoad(flag)
                if (flag != 0u) return;
                const uint lo = ld(S_LO);
                tau = lo + (ld(S_HI) - lo) / 2u;
            }
            const float m = group_mass(lid, g, top(), tau);
            if (lid == 0u) sto(S_PART + g, asuint(m));
            break;
        }
        case BISECT_INIT:
        case BISECT_STEP: {
            if (lid == 0u) {
                precise float z = 0.0f;
                for (uint w = 0u; w < groups(); w++) z += asfloat(ld(S_PART + w));
                if (phase == BISECT_INIT) {
                    sto(S_GOAL, asuint(p_top_p * z));
                    sto(S_LO, ld(S_TAU));
                    sto(S_HI, key(top()) + 1u);
                } else {
                    const uint lo = ld(S_LO);
                    const uint hi = ld(S_HI);
                    if (hi - lo > 1u) {
                        const uint mid = lo + (hi - lo) / 2u;
                        if (z >= asfloat(ld(S_GOAL))) sto(S_LO, mid);
                        else sto(S_HI, mid);
                    }
                    sto(S_TAU, ld(S_LO));
                }
            }
            break;
        }
        case UNBUMP: {
            if (lid == 0u) {
                const uint token = ld(S_TOKEN);
                stu(4, token, ldu(4, token) - 1u);
            }
            break;
        }
        case PICK: {
            const float t = top();
            const uint tau = ld(S_TAU);
            if (lid == 0u) {
                precise float z = 0.0f;
                for (uint w = 0u; w < groups(); w++) z += asfloat(ld(S_PART + w));
                precise float at = step_u * z;
                precise float c = 0.0f;
                uint found = NONE;
                uint last = NONE;
                for (uint w = 0u; w < groups(); w++) {
                    const float m = asfloat(ld(S_PART + w));
                    if (m > 0.0f) {
                        last = w;
                        if (found == NONE && c <= at && at < c + m) {
                            found = w;
                            pick_base = c;
                        }
                    }
                    c += m;
                }
                if (found == NONE) {
                    found = last;
                    pick_base = c;
                }
                pick_group = found;
            }
            GroupMemoryBarrierWithGroupSync(); // workgroupUniformLoad(pick_group)
            const uint pg = pick_group;
            const uint i0 = pg * SPAN + lid * PER;
            precise float s = 0.0f;
            for (uint j = 0u; j < PER; j++) {
                const uint i = i0 + j;
                if (i < p_vocab) {
                    const float v = ldf(5, i);
                    if (kept(v, t, tau)) s += exp(v - t);
                }
            }
            rf[lid] = s;
            GroupMemoryBarrierWithGroupSync();
            if (lid == 0u) {
                precise float z = 0.0f;
                for (uint w = 0u; w < groups(); w++) z += asfloat(ld(S_PART + w));
                precise float target_at = step_u * z;
                precise float c = pick_base;
                uint token = NONE;
                uint last = NONE;
                for (uint k = 0u; k < WG && token == NONE; k++) {
                    const uint base_i = pg * SPAN + k * PER;
                    if (rf[k] > 0.0f && c + rf[k] > target_at) {
                        precise float cc = c;
                        for (uint j = 0u; j < PER; j++) {
                            const uint i = base_i + j;
                            if (i < p_vocab) {
                                const float v = ldf(5, i);
                                if (kept(v, t, tau)) {
                                    cc += exp(v - t);
                                    last = i;
                                    if (cc > target_at) {
                                        token = i;
                                        break;
                                    }
                                }
                            }
                        }
                    } else if (rf[k] > 0.0f) {
                        for (uint j = 0u; j < PER; j++) {
                            const uint i = base_i + j;
                            if (i < p_vocab && kept(ldf(5, i), t, tau)) last = i;
                        }
                    }
                    c += rf[k];
                }
                finish(token == NONE ? last : token);
            }
            break;
        }
        default: break;
    }
}
