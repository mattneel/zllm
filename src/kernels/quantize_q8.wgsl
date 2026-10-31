// Rounds rows of activations to 8-bit blocks for the quantized prefill
// multiply, computing them first by `mode`, so the kernel that would
// produce them needn't run:
//
// - 0: x as given.
// - 1: RMSNorm of x times w (as rmsnorm.wgsl); each workgroup sums the
//   row's squares itself.
// - 2: silu(gate) · up over x holding rows of gate | up, each `width` wide
//   (as silu_mul.wgsl).
//
// The values are rounded as llama.cpp's CPU backend rounds them for a dot product with
// quantized weights (ref/matmul.zig, roundLikeLlama): x ≈ q · d.
//
// - block = 256 (Q8_K, for the K-quants): d = −max / 127, where max is the
//   block's value of largest magnitude (the first, on ties); q = −127 · x /
//   max, rounded, at most 127.
// - block = 32 (Q8_0): d = amax / 127 rounded to f16; q = x * (127 / amax).
//
// Both round halfway to even, matching .fixtures/llama.cpp's active
// ggml/src/ggml-cpu/arch/x86/quants.c quantize_row_q8_0 (not the scalar
// quantize_row_q8_0_ref, which rounds halfway away from zero).
// q goes to `qx`, four per word
// (dot4I8Packed order), d to `dx`, one per block. A workgroup takes 256
// values of one row: one Q8_K block, or eight Q8_0 blocks of 8 threads each.
//
// The divisions and the f16 rounding are exact (div_rn, f16_rne): WGSL's
// division may be off by 2.5 ULP, as it is on D3D12, and a scale one ULP
// off moves some values across a rounding boundary, changing q by one.

struct Params {
    k: u32,
    block: u32,
    mode: u32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> qx: array<u32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;
// The RMSNorm weights (mode 1; bound to anything otherwise).
@group(0) @binding(4) var<storage, read> nw: array<vec4f>;

var<workgroup> squares: array<vec2f, 64>;
var<workgroup> row_scale: f32;
// The workgroup's computed values.
var<workgroup> vals: array<vec4f, 64>;

// Per thread: the largest magnitude among its values, and its index.
var<workgroup> best: array<vec2f, 64>;

// x ≥ 0 rounded to f16 precision, to nearest even, back to f32.
fn f16_rne(x: f32) -> f32 {
    let u = bitcast<u32>(x);
    if (((u >> 23u) & 0xffu) < 113u) {
        // Below f16's normal range: multiples of 2^-24.
        return round(x * 16777216.0) * (1.0 / 16777216.0);
    }
    return bitcast<f32>((u + 0xfffu + ((u >> 13u) & 1u)) & 0xffffe000u);
}

fn round_even(y: f32) -> f32 {
    return f32(i32(bitcast<u32>(y + 12582912.0)) - 0x4b400000);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
    let row = wg.y;
    let base = row * (p.k / 4u) + wg.x * 64u;
    var v: vec4f;
    if (p.mode == 2u) {
        let src = row * (p.k / 2u) + wg.x * 64u + li;
        let g = x[src];
        v = g / (1.0 + exp(-g)) * x[src + p.k / 4u];
    } else {
        v = x[base + li];
    }
    if (p.mode == 1u) {
        var sum = vec2f(0.0);
        for (var i = li; i < p.k / 4u; i += 64u) {
            let r = x[row * (p.k / 4u) + i];
            let square = r * r;
            sum = norm_add(sum, square.x);
            sum = norm_add(sum, square.y);
            sum = norm_add(sum, square.z);
            sum = norm_add(sum, square.w);
        }
        squares[li] = sum;
        workgroupBarrier();
        for (var s = 32u; s > 0u; s /= 2u) {
            if (li < s) {
                squares[li] = norm_merge(squares[li], squares[li + s]);
            }
            workgroupBarrier();
        }
        if (li == 0u) {
            row_scale = norm_scale(squares[0], p.k, p.eps);
        }
        workgroupBarrier();
        v = v * row_scale * nw[wg.x * 64u + li];
    }
    vals[li] = v;
    let a = abs(v);
    // The first of the largest, as a strict > scan from the left.
    var m = vec2f(a.x, 0.0);
    if (a.y > m.x) {
        m = vec2f(a.y, 1.0);
    }
    if (a.z > m.x) {
        m = vec2f(a.z, 2.0);
    }
    if (a.w > m.x) {
        m = vec2f(a.w, 3.0);
    }
    best[li] = vec2f(m.x, f32(4u * li) + m.y);
    workgroupBarrier();
    // Reduce within each block: 64 threads for Q8_K, 8 for Q8_0.
    let span = p.block / 4u;
    for (var s = span / 2u; s > 0u; s /= 2u) {
        if (li % span < s) {
            let o = best[li + s];
            let b = best[li];
            if (o.x > b.x || (o.x == b.x && o.y < b.y)) {
                best[li] = o;
            }
        }
        workgroupBarrier();
    }
    let lead = best[li - li % span];
    var q = vec4f(0.0);
    var d = 0.0;
    if (p.block == 256u) {
        if (lead.x != 0.0) {
            let at = u32(lead.y);
            let max = vals[at / 4u][at % 4u];
            let iscale = div_rn(-127.0, max);
            d = div_rn(1.0, iscale);
            q = min(vec4f(127.0), vec4f(round_even(iscale * v.x), round_even(iscale * v.y), round_even(iscale * v.z), round_even(iscale * v.w)));
        }
    } else {
        if (lead.x != 0.0) {
            let d32 = div_rn(lead.x, 127.0);
            let id = div_rn(127.0, lead.x);
            d = f16_rne(d32);
            q = vec4f(round_even(v.x * id), round_even(v.y * id), round_even(v.z * id), round_even(v.w * id));
        }
    }
    let qi = vec4i(q);
    qx[base + li] = (u32(qi.x) & 0xffu) | ((u32(qi.y) & 0xffu) << 8u) | ((u32(qi.z) & 0xffu) << 16u) | ((u32(qi.w) & 0xffu) << 24u);
    if (li % span == 0u) {
        dx[(row * p.k + wg.x * 256u + 4u * li) / p.block] = d;
    }
}
