//! Fixed-size matrix operations optimized for CPU cache
//!
//! Tiger Style:
//! - Dimensions known at compile time
//! - Row-major layout for cache-friendly access
//! - Fixed-size buffers, zero allocations
//! - Explicit bounds checking

const std = @import("std");
const Config = @import("lib.zig").Config;
const CoreError = @import("lib.zig").CoreError;

/// Fixed-size matrix with compile-time dimensions
pub fn Matrix(comptime rows: usize, comptime cols: usize) type {
    comptime {
        if (rows == 0 or cols == 0) @compileError("Matrix dimensions must be greater than 0");
        if (rows > Config.max_seq_len) @compileError("Matrix rows exceed maximum");
        if (cols > Config.max_hidden_dim) @compileError("Matrix cols exceed maximum");
    }

    return struct {
        /// Row-major data layout for cache-friendly access
        data: [rows][cols]f32 align(64) = [_][cols]f32{[_]f32{0} ** cols} ** rows,

        const Self = @This();

        /// Initialize matrix with zeros
        pub fn init() Self {
            return .{};
        }

        /// Initialize matrix with a constant value
        pub fn splat(value: f32) Self {
            var result = Self{};
            for (&result.data) |*row| {
                for (row) |*elem| {
                    elem.* = value;
                }
            }
            return result;
        }

        /// Get element at (i, j)
        pub fn get(self: *const Self, i: usize, j: usize) f32 {
            std.debug.assert(i < rows);
            std.debug.assert(j < cols);
            return self.data[i][j];
        }

        /// Set element at (i, j)
        pub fn set(self: *Self, i: usize, j: usize, value: f32) void {
            std.debug.assert(i < rows);
            std.debug.assert(j < cols);
            self.data[i][j] = value;
        }

        /// Matrix-matrix multiplication: self * other
        /// Returns a new matrix of size (rows × other_cols)
        pub fn matmul(self: Self, comptime other_cols: usize, other: Matrix(cols, other_cols)) Matrix(rows, other_cols) {
            // Napkin math: rows * other_cols * cols multiplications
            var result = Matrix(rows, other_cols).init();

            // Cache-friendly: iterate rows of A, then columns of B
            for (0..rows) |i| {
                for (0..other_cols) |j| {
                    var sum: f32 = 0;
                    for (0..cols) |k| {
                        sum += self.data[i][k] * other.data[k][j];
                    }
                    result.data[i][j] = sum;
                }
            }

            return result;
        }
    };
}

test "matrix initialization" {
    const M = Matrix(2, 3);
    const m = M.init();
    try std.testing.expectEqual(@as(f32, 0), m.data[0][0]);
}

test "matrix get/set" {
    const M = Matrix(2, 2);
    var m = M.init();
    m.set(0, 1, 42);
    try std.testing.expectEqual(@as(f32, 42), m.get(0, 1));
}

test "matrix multiplication" {
    const M1 = Matrix(2, 3);
    const M2 = Matrix(3, 2);

    var a = M1.init();
    a.data = .{
        .{ 1, 2, 3 },
        .{ 4, 5, 6 },
    };

    var b = M2.init();
    b.data = .{
        .{ 7, 8 },
        .{ 9, 10 },
        .{ 11, 12 },
    };

    const c = a.matmul(2, b);

    // First row, first col: 1*7 + 2*9 + 3*11 = 58
    try std.testing.expectEqual(@as(f32, 58), c.data[0][0]);
    // Second row, second col: 4*8 + 5*10 + 6*12 = 154
    try std.testing.expectEqual(@as(f32, 154), c.data[1][1]);
}
