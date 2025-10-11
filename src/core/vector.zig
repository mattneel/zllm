//! Fixed-size vector operations with SIMD support
//!
//! Tiger Style:
//! - Fixed dimensions known at compile time
//! - SIMD-aligned memory layout (64 bytes)
//! - Zero allocations
//! - Explicit error handling

const std = @import("std");
const Config = @import("lib.zig").Config;
const CoreError = @import("lib.zig").CoreError;

/// Fixed-size vector with SIMD alignment
pub fn Vector(comptime dim: usize) type {
    comptime {
        if (dim == 0) @compileError("Vector dimension must be greater than 0");
        if (dim > Config.max_hidden_dim) @compileError("Vector dimension exceeds maximum");
    }

    return struct {
        /// Data aligned for SIMD operations (64-byte cache line)
        data: [dim]f32 align(64) = [_]f32{0} ** dim,

        const Self = @This();

        /// Initialize vector with zeros
        pub fn init() Self {
            return .{};
        }

        /// Initialize vector with a constant value
        pub fn splat(value: f32) Self {
            var result = Self{};
            for (&result.data) |*elem| {
                elem.* = value;
            }
            return result;
        }

        /// Dot product: self · other
        pub fn dot(self: Self, other: Self) f32 {
            var sum: f32 = 0;

            // TODO: SIMD optimization with @Vector
            for (self.data, other.data) |a, b| {
                sum += a * b;
            }

            return sum;
        }

        /// Element-wise addition: self + other
        pub fn add(self: Self, other: Self) Self {
            var result = Self{};
            for (&result.data, self.data, other.data) |*r, a, b| {
                r.* = a + b;
            }
            return result;
        }

        /// Scalar multiplication: self * scalar
        pub fn scale(self: Self, scalar: f32) Self {
            var result = Self{};
            for (&result.data, self.data) |*r, a| {
                r.* = a * scalar;
            }
            return result;
        }

        /// L2 norm (Euclidean length)
        pub fn norm(self: Self) f32 {
            return @sqrt(self.dot(self));
        }
    };
}

test "vector initialization" {
    const V = Vector(4);
    const v = V.init();
    try std.testing.expectEqual(@as(f32, 0), v.data[0]);
}

test "vector dot product" {
    const V = Vector(3);
    var v1 = V.init();
    v1.data = .{ 1, 2, 3 };
    var v2 = V.init();
    v2.data = .{ 4, 5, 6 };

    const result = v1.dot(v2);
    try std.testing.expectEqual(@as(f32, 32), result); // 1*4 + 2*5 + 3*6 = 32
}

test "vector addition" {
    const V = Vector(3);
    var v1 = V.init();
    v1.data = .{ 1, 2, 3 };
    var v2 = V.init();
    v2.data = .{ 4, 5, 6 };

    const result = v1.add(v2);
    try std.testing.expectEqual(@as(f32, 5), result.data[0]);
    try std.testing.expectEqual(@as(f32, 7), result.data[1]);
    try std.testing.expectEqual(@as(f32, 9), result.data[2]);
}
