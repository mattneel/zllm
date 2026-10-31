//! FP8 E4M3FN, as the FP8 KV cache stores it (kernels/fp8.wgsl,
//! agility/kernels/fp8.hlsli): 1 sign, 4 exponent (bias 7) and 3 mantissa
//! bits, no infinities, largest finite 448. Each (position, KV head) row is
//! divided by its scale: the smallest power of two with amax / scale ≤ 448,
//! so applying it is exact on every backend.

const std = @import("std");

pub const max: f32 = 448;

/// |x| ≤ 448 (clamped) → E4M3FN bits, rounded to nearest, ties to even.
pub fn encode(x: f32) u8 {
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    const a = @min(@abs(x), max);
    if (a < 0x1p-6) {
        // Subnormals and zero: a · 2^9 is exact.
        const s = a * 512;
        var q: u8 = @intFromFloat(@floor(s));
        const r = s - @as(f32, @floatFromInt(q));
        if (r > 0.5 or (r == 0.5 and q & 1 == 1)) q += 1;
        return sign | q;
    }
    const ab: u32 = @bitCast(a);
    var e: u32 = (ab >> 23) - 120;
    const mant = ab & 0x7fffff;
    var q = mant >> 20;
    const rem = mant & 0xfffff;
    if (rem > 0x80000 or (rem == 0x80000 and q & 1 == 1)) q += 1;
    if (q == 8) {
        q = 0;
        e += 1;
    }
    return sign | @as(u8, @intCast(@min((e << 3) | q, 0x7e)));
}

pub fn decode(b: u8) f32 {
    const e: u5 = @intCast((b >> 3) & 15);
    const m: f32 = @floatFromInt(b & 7);
    const mag = if (e == 0) m * 0x1p-9 else std.math.ldexp(1 + m / 8, @as(i32, e) - 7);
    return if (b & 0x80 != 0) -mag else mag;
}

/// A row's scale: the smallest power of two 2^k with its largest magnitude
/// m ≤ 448 · 2^k (448 = 1.75 · 2^8); 1 for a zero row or a subnormal m,
/// which GPUs may flush.
pub fn rowScale(row: []const f32) f32 {
    var m: f32 = 0;
    for (row) |v| m = @max(m, @abs(v));
    const bits: u32 = @bitCast(m);
    const e: i32 = @intCast((bits >> 23) & 255);
    if (e == 0) return 1;
    const k = e - 127 - 8 + @as(i32, @intFromBool(bits & 0x7fffff > 0x600000));
    return std.math.ldexp(@as(f32, 1), k);
}

test "every finite code decodes and encodes back to itself" {
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (b & 0x7f == 0x7f) continue; // NaN
        try std.testing.expectEqual(b, encode(decode(b)));
    }
}

test "encode rounds to nearest, ties to even, and clamps" {
    try std.testing.expectEqual(@as(f32, 448), decode(encode(1000)));
    try std.testing.expectEqual(@as(f32, -448), decode(encode(-449)));
    // 1.0625 lies halfway between 1 (even mantissa) and 1.125.
    try std.testing.expectEqual(@as(f32, 1), decode(encode(1.0625)));
    // 1.1875 lies halfway between 1.125 (odd) and 1.25 (even).
    try std.testing.expectEqual(@as(f32, 1.25), decode(encode(1.1875)));
    try std.testing.expectEqual(@as(f32, 1.125), decode(encode(1.07)));
    // Subnormals: steps of 2^-9; halfway 2^-10 rounds to 0 (even), 3·2^-10 to 2·2^-9.
    try std.testing.expectEqual(@as(f32, 0), decode(encode(0x1p-10)));
    try std.testing.expectEqual(@as(f32, 0x1p-8), decode(encode(3 * 0x1p-10)));
    // Just below the smallest normal rounds up to it.
    try std.testing.expectEqual(@as(f32, 0x1p-6), decode(encode(0x1p-6 - 0x1p-12)));
    try std.testing.expectEqual(@as(u8, 0x80), encode(-0.0));
}

test "a row's scale is the smallest power of two that fits it" {
    try std.testing.expectEqual(@as(f32, 1), rowScale(&.{ 0, -0.0 }));
    try std.testing.expectEqual(@as(f32, 1), rowScale(&.{ 448, -3 }));
    try std.testing.expectEqual(@as(f32, 2), rowScale(&.{ 449, 1 }));
    try std.testing.expectEqual(@as(f32, 0x1p-8), rowScale(&.{-1.75}));
    try std.testing.expectEqual(@as(f32, 0x1p-7), rowScale(&.{1.76}));
}
