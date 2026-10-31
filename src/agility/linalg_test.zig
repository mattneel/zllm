//! Mutation checks for LinAlg's activation high/residual representation and
//! exact half/integer weights with per-block scales outside matrix products.
//! Actual matrix arithmetic, tails, strides and bitwise token splitting are
//! checked by kernels/check.zig on the device at the unchanged 1e-5 bound.
const std = @import("std");

const Parts = struct {
    hi: f32,
    lo: f32,

    fn split(value: f32) Parts {
        const hi: f32 = @as(f16, @floatCast(value));
        return .{ .hi = hi, .lo = @as(f16, @floatCast(value - hi)) };
    }
};

fn product(x: f32, weight: f16, scale: f16, residual: bool, apply_scale: bool) f64 {
    const a = Parts.split(x);
    var sum = @as(f64, a.hi) * @as(f64, weight);
    if (residual) sum += @as(f64, a.lo) * @as(f64, weight);
    return if (apply_scale) sum * @as(f64, scale) else sum;
}

fn accurate(value: f64, want: f64) bool {
    return @abs(value - want) / @abs(want) <= 1e-5;
}

test "LinAlg operand checker rejects dropped activation correction or scale" {
    const x: f32 = 1.0003;
    // Both exact int8 values (Q8_0) and half codebook values (TQ2_T).
    for ([_]f16{ 127, -0.12504 }) |weight| {
        const scale: f16 = 0.03125;
        const want = @as(f64, x) * @as(f64, weight) * @as(f64, scale);
        try std.testing.expect(accurate(product(x, weight, scale, true, true), want));
        try std.testing.expect(!accurate(product(x, weight, scale, false, true), want));
        try std.testing.expect(!accurate(product(x, weight, scale, true, false), want));
    }
}

test "LinAlg compensation retains signed and small operand products" {
    for ([_]f32{ -1.0003, 0.002003, 1.0003 }) |x| {
        for ([_]f16{ -127, -0.12504, 17.123 }) |weight| {
            const scale: f16 = 0.03125;
            const want = @as(f64, x) * @as(f64, weight) * @as(f64, scale);
            try std.testing.expect(accurate(product(x, weight, scale, true, true), want));
        }
    }
}
