//! CPU f32 reference operators: straightforward loops with f32 accumulation.
//! They define correct results for the GPU kernels, so clarity comes first.
//! The matrix multiply, which dominates run time, is tuned separately in
//! matmul.zig.

const std = @import("std");

const vec_len = 16;
const V = @Vector(vec_len, f32);

pub fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + vec_len <= a.len) : (i += vec_len) {
        const va: V = a[i..][0..vec_len].*;
        const vb: V = b[i..][0..vec_len].*;
        acc += va * vb;
    }
    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

/// In place: x / sqrt(mean(x²) + eps) * weight.
pub fn rmsNorm(x: []f32, weight: []const f32, eps: f32) void {
    std.debug.assert(x.len == weight.len);
    const mean = dot(x, x) / @as(f32, @floatFromInt(x.len));
    const scale = 1.0 / @sqrt(mean + eps);
    for (x, weight) |*v, w| v.* = v.* * scale * w;
}

/// In place: x / sqrt(sum(x²) + eps).
pub fn l2Norm(x: []f32, eps: f32) void {
    const scale = 1.0 / @sqrt(dot(x, x) + eps);
    for (x) |*v| v.* *= scale;
}

pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// log(1 + exp(x)), linear above 20 as in ggml.
pub fn softplus(x: f32) f32 {
    return if (x > 20) x else @log(1.0 + @exp(x));
}

/// NeoX-style rotary embedding on the first `n_rot` dims of one head:
/// pairs (i, i + n_rot/2) rotate by pos * base^(-2i/n_rot).
pub fn rope(x: []f32, pos: u32, n_rot: usize, base: f32) void {
    const half = n_rot / 2;
    for (0..half) |i| {
        const cs = ropeCosSin(pos, i, n_rot, base);
        const x0 = x[i];
        const x1 = x[i + half];
        x[i] = x0 * cs[0] - x1 * cs[1];
        x[i + half] = x0 * cs[1] + x1 * cs[0];
    }
}

/// Normal (GPT-J) rotary embedding on the first `n_rot` dims of one head:
/// pairs (2i, 2i + 1) rotate by pos * base^(-2i/n_rot) / factors[i]
/// (Llama 3's `rope_freqs`; none is 1).
pub fn ropeNormal(x: []f32, pos: u32, n_rot: usize, base: f32, factors: ?[]const f32) void {
    for (0..n_rot / 2) |i| {
        const cs = ropeCosSinScaled(pos, i, n_rot, base, if (factors) |f| f[i] else 1);
        const x0 = x[2 * i];
        const x1 = x[2 * i + 1];
        x[2 * i] = x0 * cs[0] - x1 * cs[1];
        x[2 * i + 1] = x0 * cs[1] + x1 * cs[0];
    }
}

/// (cos, sin) of the rotation applied to pair `i` at position `pos`. GPU
/// kernels read a table of these rather than evaluating trig themselves.
pub fn ropeCosSin(pos: u32, i: usize, n_rot: usize, base: f32) [2]f32 {
    return ropeCosSinScaled(pos, i, n_rot, base, 1);
}

/// As `ropeCosSin`, the angle divided by `factor`.
pub fn ropeCosSinScaled(pos: u32, i: usize, n_rot: usize, base: f32, factor: f32) [2]f32 {
    const exponent = -2.0 * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n_rot));
    const theta = @as(f32, @floatFromInt(pos)) * std.math.pow(f32, base, exponent) / factor;
    return .{ @cos(theta), @sin(theta) };
}

/// In-place softmax.
pub fn softmax(x: []f32) void {
    var max: f32 = -std.math.inf(f32);
    for (x) |v| max = @max(max, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - max);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

test "rope preserves norms, is identity at position 0 and leaves tail dims" {
    var x: [8]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const orig = x;
    rope(&x, 0, 4, 10000);
    try std.testing.expectEqualSlices(f32, &orig, &x);

    rope(&x, 7, 4, 10000);
    // Pairs (0,2) and (1,3) keep their lengths; dims 4.. are untouched.
    try std.testing.expectApproxEqAbs(@as(f32, 1 * 1 + 3 * 3), x[0] * x[0] + x[2] * x[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2 * 2 + 4 * 4), x[1] * x[1] + x[3] * x[3], 1e-4);
    try std.testing.expectEqualSlices(f32, orig[4..], x[4..]);
    // The first pair rotates by exactly `pos` radians.
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 7)) * 1 - @sin(@as(f32, 7)) * 3, x[0], 1e-5);
}

test "norms" {
    var a: [4]f32 = .{ 3, 4, 0, 0 };
    l2Norm(&a, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 1), dot(&a, &a), 1e-6);

    var b: [2]f32 = .{ 2, -2 };
    rmsNorm(&b, &.{ 1, 0.5 }, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 1), b[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), b[1], 1e-6);
}
