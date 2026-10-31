//! CPU reference operators. RMSNorm follows pinned ggml's mixed-precision
//! reduction; other reductions use f32 accumulation.
//! They define correct results for the GPU kernels, so clarity comes first.
//! The matrix multiply is tuned separately in matmul.zig.

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
/// Shared by both activation modes: pinned ggml-cpu/ops.cpp RMSNorm sums f32
/// squares in f64, divides in f64, then uses an f32 mean, sqrt and reciprocal.
/// Keep this accurate normalization global rather than duplicating every caller
/// for the rounded-reference diagnostic. Multiplications remain x * scale * w.
pub fn rmsNorm(x: []f32, weight: []const f32, eps: f32) void {
    @setFloatMode(.strict);
    std.debug.assert(x.len == weight.len);
    var sum: f64 = 0;
    for (x) |v| {
        const square: f32 = v * v;
        sum += square;
    }
    const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(x.len)));
    const scale: f32 = 1.0 / @sqrt(mean + eps);
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

test "RMSNorm matches pinned ggml arithmetic bitwise" {
    // Independent oracle: each input square rounded to f32, sequential f64
    // sum/division, f32 mean/add/sqrt/reciprocal, then f32 x*scale followed by *w.
    // This fixture distinguishes f64 squares, f32 reduction, f64 mean/sqrt,
    // division by sqrt instead of reciprocal multiplication, and x*(scale*w).
    const input = [_]u32{
        0xbf857c32, 0xbf8d2c0b, 0xbfa5eb34, 0x3ddf5988, 0xbdc29599,
        0x3e230df4, 0xc10ccddd, 0x4052123d, 0xbf446fd1, 0xbf128c7d,
        0xbe8b3a39, 0x3f1207f1, 0xbf78507a, 0x3d3ab508, 0x406c1281,
        0x3e8f5b63, 0x3f2c4248,
    };
    const weight = [_]u32{
        0x3e9d12e6, 0x3f923ccb, 0xbf00548d, 0x3f07a020, 0x3f840c9f,
        0x3f407d1a, 0xbfeefd55, 0xbf682ff8, 0x3e78a781, 0x3ea24f78,
        0xbf9ced24, 0x3e55d151, 0x3fd00030, 0x3ffb9078, 0xbfab6e01,
        0xbf50616f, 0x3fa4b9e8,
    };
    const expected = [_]u32{
        0xbe019766, 0xbeff32a0, 0x3e839a3d, 0x3cbb39dd, 0xbd1ecfe4,
        0x3d41fd3a, 0x40cffc64, 0xbf96bc37, 0xbd96f2c6, 0xbd930467,
        0x3e070a0f, 0x3d40fcae, 0xbf1f9dac, 0x3d11269c, 0xbffa21fc,
        0xbdb8a2bc, 0x3eaf61ac,
    };
    var x: [input.len]f32 = undefined;
    var w: [weight.len]f32 = undefined;
    for (&x, input) |*v, bits| v.* = @bitCast(bits);
    for (&w, weight) |*v, bits| v.* = @bitCast(bits);
    rmsNorm(&x, &w, 1e-5);
    for (x, expected) |v, bits| try std.testing.expectEqual(bits, @as(u32, @bitCast(v)));
}
