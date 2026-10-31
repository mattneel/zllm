// The attention prologue for the step's m tokens, on the fused Q|K|V
// projection (row t: Q heads at 0, interleaved with their output gates when
// `gated`, else head_dim apart; K at k_off, V at v_off, `row` floats per
// token). One workgroup per (head x, token t), over heads + kv_heads:
//
//   x < heads    query head x: RMSNorm · q_norm (when qk_norm), then RoPE
//                on the first n_rot dims, in place
//   x ≥ heads    KV head g = x − heads: K gets RMSNorm · k_norm (when
//                qk_norm) and RoPE; K and V go to the cache at position
//                pos0 + t
//
// RoPE rotates pairs (i, i + n_rot/2) (NeoX), or (2i, 2i + 1) when `normal`,
// by the angle of position pos0 + t, read as (cos, sin) from a host-built
// table: WGSL sin/cos are only accurate near [−π, π]. The cache stores
// `kv_heads · head_dim` values per position as `kv` says (KvFormat): f32
// bits, f16 pairs, or FP8 (fp8.wgsl) with a scale per position and KV head
// at word `scales` on. head_dim ≤ WG.
//
// The f16 values are rounded to nearest, ties to even, as llama.cpp's cache
// rounds: WGSL lets pack2x16float round either way, and D3D12 truncates,
// which biases every cached value toward zero.

struct Params {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    n_rot: u32,
    row: u32,
    k_off: u32,
    v_off: u32,
    kv: u32,
    eps: f32,
    gated: u32,
    qk_norm: u32,
    normal: u32,
    scales: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> q_norm: array<f32>;
@group(0) @binding(3) var<storage, read> k_norm: array<f32>;
@group(0) @binding(4) var<storage, read> rope: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> kc: array<u32>;
@group(0) @binding(6) var<storage, read_write> vc: array<u32>;
@group(0) @binding(7) var<uniform> step: Step;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, WG>;
var<workgroup> vals: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let i = lid3.x;
    let x = wg.x;
    let t = wg.y;
    let hd = p.head_dim;
    let is_q = x < p.heads;
    let g = x - select(p.heads, 0u, is_q);
    let q_head = select(hd, 2u * hd, p.gated != 0u);
    let base = t * p.row + select(p.k_off + g * hd, x * q_head, is_q);

    var v = 0.0;
    if (i < hd) {
        v = qkv[base + i];
    }
    partial[i] = v * v;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (i < s) {
            partial[i] += partial[i + s];
        }
        workgroupBarrier();
    }
    let scale = 1.0 / sqrt(partial[0] / f32(hd) + p.eps);
    if (i < hd) {
        if (p.qk_norm != 0u) {
            vals[i] = v * scale * select(k_norm[i], q_norm[i], is_q);
        } else {
            vals[i] = v;
        }
    }
    workgroupBarrier();

    let half = p.n_rot / 2u;
    let pos = step.pos0 + t;
    if (i < half) {
        let cs = rope[pos * half + i];
        // The pair's two dims: (i, i + half), or (2i, 2i + 1).
        let a = select(i, 2u * i, p.normal != 0u);
        let b = select(i + half, 2u * i + 1u, p.normal != 0u);
        let x0 = vals[a];
        let x1 = vals[b];
        vals[a] = x0 * cs.x - x1 * cs.y;
        vals[b] = x0 * cs.y + x1 * cs.x;
    }
    workgroupBarrier();

    if (is_q) {
        if (i < hd) {
            qkv[base + i] = vals[i];
        }
        return;
    }
    let width = p.kv_heads * hd;
    let e = pos * width + g * hd;
    let vb = t * p.row + p.v_off + g * hd;
    if (p.kv == 2u) {
        // FP8: each row by its power-of-two scale (fp8.wgsl), one thread
        // per word of four values.
        let k_exp = fp8_scale_exp(row_max(select(0.0, abs(vals[i]), i < hd), i));
        let v_exp = fp8_scale_exp(row_max(select(0.0, abs(qkv[vb + min(i, hd - 1u)]), i < hd), i));
        if (i < hd / 4u) {
            var kw = 0u;
            var vw = 0u;
            for (var j = 0u; j < 4u; j++) {
                kw |= fp8_bits(vals[4u * i + j] * pow2(-k_exp)) << (8u * j);
                vw |= fp8_bits(qkv[vb + 4u * i + j] * pow2(-v_exp)) << (8u * j);
            }
            kc[e / 4u + i] = kw;
            vc[e / 4u + i] = vw;
        }
        if (i == 0u) {
            kc[p.scales + pos * p.kv_heads + g] = bitcast<u32>(pow2(k_exp));
            vc[p.scales + pos * p.kv_heads + g] = bitcast<u32>(pow2(v_exp));
        }
        return;
    }
    // K and V to the cache, one thread per pair of values.
    if (i < hd / 2u) {
        let k0 = vals[2u * i];
        let k1 = vals[2u * i + 1u];
        let v0 = qkv[vb + 2u * i];
        let v1 = qkv[vb + 2u * i + 1u];
        if (p.kv == 1u) {
            kc[e / 2u + i] = pack_f16(vec2f(k0, k1));
            vc[e / 2u + i] = pack_f16(vec2f(v0, v1));
        } else {
            kc[e + 2u * i] = bitcast<u32>(k0);
            kc[e + 2u * i + 1u] = bitcast<u32>(k1);
            vc[e + 2u * i] = bitcast<u32>(v0);
            vc[e + 2u * i + 1u] = bitcast<u32>(v1);
        }
    }
}

// The largest `a` over the workgroup's threads. Every thread calls it.
fn row_max(a: f32, i: u32) -> f32 {
    partial[i] = a;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (i < s) {
            partial[i] = max(partial[i], partial[i + s]);
        }
        workgroupBarrier();
    }
    let m = partial[0];
    workgroupBarrier();
    return m;
}
