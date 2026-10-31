// .fixtures/llama.cpp/ggml/src/ggml-cpu/ops.cpp RMSNorm sums f32 products
// in f64, divides in f64, casts the mean to f32, then uses f32 sqrt,
// reciprocal and x * scale * w.
// Two-float sums retain the f32 squares' rounding without native f64.
// Keep this ordering identical to agility/kernels/norm.hlsli.
fn norm_add(sum: vec2f, value: f32) -> vec2f {
    let hi = sum.x + value;
    let v = hi - sum.x;
    let error = (sum.x - (hi - v)) + (value - v);
    return vec2f(hi, sum.y + error);
}

fn norm_merge(a: vec2f, b: vec2f) -> vec2f {
    let sum = norm_add(a, b.x);
    return vec2f(sum.x, sum.y + b.y);
}

// a / b rounded to nearest even, for normal a, b and quotient: long
// division of the 24-bit significands, then a guard bit and the remainder
// as sticky. Callers guard zero (a zero norm uses eps directly).
fn div_rn(a: f32, b: f32) -> f32 {
    let ua = bitcast<u32>(a);
    let ub = bitcast<u32>(b);
    let sign = (ua ^ ub) & 0x80000000u;
    var e = i32((ua >> 23u) & 0xffu) - i32((ub >> 23u) & 0xffu) + 127;
    let ma = (ua & 0x7fffffu) | 0x800000u;
    let mb = (ub & 0x7fffffu) | 0x800000u;
    var rem = ma;
    if (ma < mb) {
        rem = ma << 1u;
        e -= 1;
    }
    var q = 0u;
    for (var i = 0u; i < 25u; i++) {
        q <<= 1u;
        if (rem >= mb) {
            rem -= mb;
            q |= 1u;
        }
        rem <<= 1u;
    }
    var m = q >> 1u;
    if ((q & 1u) == 1u && (rem != 0u || (m & 1u) == 1u)) {
        m += 1u;
    }
    if (m == 0x1000000u) {
        m = 0x800000u;
        e += 1;
    }
    return bitcast<f32>(sign | (u32(e) << 23u) | (m & 0x7fffffu));
}

// Dekker product residual: HLSL's fma is double-only and mad need not
// be fused, so use identical explicitly ordered f32 operations instead.
fn norm_product_error(a: f32, b: f32, product: f32) -> f32 {
    let sa = 4097.0 * a;
    let ah = sa - (sa - a);
    let al = a - ah;
    let sb = 4097.0 * b;
    let bh = sb - (sb - b);
    let bl = b - bh;
    return ((ah * bh - product) + ah * bl + al * bh) + al * bl;
}

fn norm_scale(sum: vec2f, dim: u32, eps: f32) -> f32 {
    var mean = 0.0;
    if (sum.x != 0.0) {
        let n = f32(dim);
        let hi = div_rn(sum.x, n);
        // Correct before rounding the mean, rather than round the sum first.
        let product = hi * n;
        let error = norm_product_error(hi, n, product);
        let residual = ((sum.x - product) - error) + sum.y;
        mean = hi + residual / n;
    }
    return div_rn(1.0, sqrt(mean + eps));
}
