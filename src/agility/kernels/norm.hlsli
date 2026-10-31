// Match kernels/norm.wgsl: f32 squares, two-float sum, corrected f32 mean,
// f32 sqrt and exact reciprocal. Not an f64 square or inverse-sqrt estimate.
float2 norm_add(float2 sum, float value) {
    precise float hi = sum.x + value;
    precise float v = hi - sum.x;
    precise float error = (sum.x - (hi - v)) + (value - v);
    precise float lo = sum.y + error;
    return float2(hi, lo);
}
float2 norm_merge(float2 a, float2 b) {
    float2 sum = norm_add(a, b.x);
    precise float lo = sum.y + b.y;
    return float2(sum.x, lo);
}
// Normal operands and quotient only; callers guard zero.
float div_rn(float a, float b) {
    uint ua = asuint(a), ub = asuint(b);
    uint signbit = (ua ^ ub) & 0x80000000u;
    int e = int((ua >> 23u) & 0xffu) - int((ub >> 23u) & 0xffu) + 127;
    uint ma = (ua & 0x7fffffu) | 0x800000u;
    uint mb = (ub & 0x7fffffu) | 0x800000u;
    uint rem = ma;
    if (ma < mb) { rem = ma << 1u; e -= 1; }
    uint q = 0u;
    for (uint i = 0u; i < 25u; i++) {
        q <<= 1u;
        if (rem >= mb) { rem -= mb; q |= 1u; }
        rem <<= 1u;
    }
    uint m = q >> 1u;
    if ((q & 1u) == 1u && (rem != 0u || (m & 1u) == 1u)) m += 1u;
    if (m == 0x1000000u) { m = 0x800000u; e += 1; }
    return asfloat(signbit | (uint(e) << 23u) | (m & 0x7fffffu));
}
float norm_product_error(float a, float b, float product) {
    precise float sa = 4097.0 * a;
    precise float ah = sa - (sa - a);
    precise float al = a - ah;
    precise float sb = 4097.0 * b;
    precise float bh = sb - (sb - b);
    precise float bl = b - bh;
    precise float error = ((ah * bh - product) + ah * bl + al * bh) + al * bl;
    return error;
}
float norm_scale(float2 sum, uint dim, float eps) {
    precise float mean = 0.0;
    if (sum.x != 0.0) {
        float n = float(dim);
        float hi = div_rn(sum.x, n);
        precise float product = hi * n;
        float error = norm_product_error(hi, n, product);
        precise float residual = ((sum.x - product) - error) + sum.y;
        mean = hi + residual / n;
    }
    precise float root = sqrt(mean + eps);
    return div_rn(1.0, root);
}
