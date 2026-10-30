// Attention, part 1 (flash-decoding), for decode and prefill steps: one
// workgroup per (KV head g, block b of BLOCK cache positions, query token
// row0 + t). Prefill steps run it in chunks of rows starting at row0, so
// `part` holds one chunk. For the GQA group of
// query heads that read KV head g, it scores the block's keys (positions up
// to pos0 + t, causal), takes a softmax local to the block, and writes per
// head (max score, sum of exp, Σ exp · v) to `part` for attn_combine.
//
// Scores: 4 threads per key, each over a quarter of head_dim; the group's
// queries sit in workgroup memory, so each K row is read once for the whole
// group. Cache format as in `attention` (f16 pairs or f32 bits).
//
// MAX_GROUP, the most query heads per KV head, is prepended (kernels.zig:
// attn_split, attn_split_wide).
//
// part layout per (t, head, block): [max, sum, o_0 .. o_{head_dim−1}], at
// ((t · heads + head) · blocks + b) · (head_dim + 2), blocks = max_blocks,
// t counted from row0.
//
// A token's result depends only on its position and the fixed blocks, not
// on how the sequence was split into steps or chunks.

struct Params {
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    f16: u32,
    scale: f32,
    max_blocks: u32,
    q_row: u32,
    row0: u32,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qg: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<u32>;
@group(0) @binding(3) var<storage, read> vc: array<u32>;
@group(0) @binding(4) var<storage, read_write> part: array<f32>;
@group(0) @binding(5) var<uniform> step: Step;

const WG: u32 = 256u;
const BLOCK: u32 = 64u;
const SPLIT: u32 = WG / BLOCK;
const MAX_HEAD_DIM: u32 = 256u;

// At MAX_GROUP 12, 16,384 bytes in all, WebGPU's guaranteed limit; once
// the scores are reduced, `red` holds each head's (max, sum).
var<workgroup> qs: array<f32, MAX_GROUP * MAX_HEAD_DIM>;
var<workgroup> red: array<f32, WG>;
var<workgroup> sc: array<f32, MAX_GROUP * BLOCK>;

fn k_pair(i: u32) -> vec2f {
    if (p.f16 != 0u) {
        return unpack2x16float(kc[i]);
    }
    return vec2f(bitcast<f32>(kc[2u * i]), bitcast<f32>(kc[2u * i + 1u]));
}

fn v_at(i: u32) -> f32 {
    if (p.f16 != 0u) {
        let pair = unpack2x16float(vc[i / 2u]);
        return select(pair.x, pair.y, (i & 1u) == 1u);
    }
    return bitcast<f32>(vc[i]);
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let lid = lid3.x;
    let g = wg.x;
    let b = wg.y;
    let t = wg.z;
    let row = p.row0 + t;
    let hd = p.head_dim;
    let group = p.heads / p.kv_heads;
    let keys = step.pos0 + row + 1u;
    let k0 = b * BLOCK;
    let kv_stride = p.kv_heads * hd;
    let q_stride = p.q_row;

    for (var i = lid; i < group * hd; i += WG) {
        let h = g * group + i / hd;
        qs[i] = qg[row * q_stride + 2u * h * hd + i % hd];
    }
    workgroupBarrier();

    // Partial scores: thread (key j, part s) over dims [s·hd/4, (s+1)·hd/4).
    let j = lid / SPLIT;
    let s = lid % SPLIT;
    let key = k0 + j;
    let span = hd / SPLIT;
    var dot: array<f32, MAX_GROUP>;
    for (var h = 0u; h < MAX_GROUP; h++) {
        dot[h] = 0.0;
    }
    if (key < keys) {
        let kb = (key * kv_stride + g * hd + s * span) / 2u;
        for (var d = 0u; d < span / 2u; d++) {
            let kk = k_pair(kb + d);
            let e = s * span + 2u * d;
            for (var h = 0u; h < group; h++) {
                dot[h] += qs[h * hd + e] * kk.x + qs[h * hd + e + 1u] * kk.y;
            }
        }
    }
    for (var h = 0u; h < group; h++) {
        red[lid] = dot[h];
        workgroupBarrier();
        if (s == 0u) {
            let total = red[lid] + red[lid + 1u] + red[lid + 2u] + red[lid + 3u];
            sc[h * BLOCK + j] = select(-0x1.fffffep+127f, total * p.scale, key < keys);
        }
        workgroupBarrier();
    }

    // Block softmax per head: thread h finds the max and the sum.
    if (lid < group) {
        let n = min(BLOCK, keys - min(keys, k0));
        var mx = -0x1.fffffep+127f;
        for (var i = 0u; i < n; i++) {
            mx = max(mx, sc[lid * BLOCK + i]);
        }
        var sum = 0.0;
        for (var i = 0u; i < n; i++) {
            let e = exp(sc[lid * BLOCK + i] - mx);
            sc[lid * BLOCK + i] = e;
            sum += e;
        }
        for (var i = n; i < BLOCK; i++) {
            sc[lid * BLOCK + i] = 0.0;
        }
        red[2u * lid] = mx;
        red[2u * lid + 1u] = sum;
    }
    workgroupBarrier();

    // Σ exp · v: thread d reads each V element once for the whole group.
    let n = min(BLOCK, keys - min(keys, k0));
    for (var d = lid; d < hd; d += WG) {
        var acc: array<f32, MAX_GROUP>;
        for (var h = 0u; h < MAX_GROUP; h++) {
            acc[h] = 0.0;
        }
        for (var i = 0u; i < n; i++) {
            let v = v_at((k0 + i) * kv_stride + g * hd + d);
            for (var h = 0u; h < group; h++) {
                acc[h] += sc[h * BLOCK + i] * v;
            }
        }
        for (var h = 0u; h < group; h++) {
            let head = g * group + h;
            let base = ((t * p.heads + head) * p.max_blocks + b) * (hd + 2u);
            part[base + 2u + d] = acc[h];
            if (d == 0u) {
                part[base] = red[2u * h];
                part[base + 1u] = red[2u * h + 1u];
            }
        }
    }
}
