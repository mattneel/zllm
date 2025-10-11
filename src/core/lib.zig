//! Core primitives for zLLM.
//!
//! This module provides fundamental data structures and operations
//! optimized for CPU execution with SIMD and cache awareness.
//!
//! Safety: All operations are bounded with explicit limits.
//! Performance: Cache-aligned, SIMD-optimized, zero allocations in hot paths.
//! Developer experience: Clear naming, simple interfaces, comprehensive tests.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

// Fixed limits for safety (Tiger Style: set fixed limits).
pub const Limits = struct {
    pub const max_dim = 16384;
    pub const max_seq_len = 8192;
    pub const max_batch_size = 256;
    pub const max_threads = 128;
    pub const cache_line_size = 64;
};

// Re-export core components.
pub const vector = @import("vector.zig");
pub const matrix = @import("matrix.zig");
pub const tensor = @import("tensor.zig");
pub const thread_pool = @import("thread_pool.zig");
pub const simd = @import("simd.zig");

// Core error set (Tiger Style: explicit error handling).
pub const CoreError = error{
    DimensionMismatch,
    DimensionTooLarge,
    InvalidConfiguration,
    ThreadPoolExhausted,
    NumericalInstability,
};

// Configuration with compile-time validation.
pub const Config = struct {
    enable_simd: bool = build_options.enable_simd,
    max_threads: u32 = build_options.max_threads,

    // Validate at compile time.
    comptime {
        if (build_options.max_threads > Limits.max_threads) {
            @compileError("max_threads exceeds limit");
        }
    }
};

// Global configuration instance.
var config = Config{};

// Pre-allocated global workspace to avoid runtime allocations.
pub const GlobalWorkspace = struct {
    // Fixed-size buffers for temporary computations.
    temp_vectors: [16][Limits.max_dim]f32 align(64) = [_][Limits.max_dim]f32{[_]f32{0} ** Limits.max_dim} ** 16,
    temp_matrices: [4][1024][1024]f32 align(64) = undefined,
    initialized: bool = false,

    pub fn init() GlobalWorkspace {
        return .{ .initialized = true };
    }
};

var global_workspace = GlobalWorkspace{};

/// Initialize the core module with custom configuration.
/// All allocations happen here, none in hot paths.
pub fn init(allocator: std.mem.Allocator, custom_config: ?Config) CoreError!void {
    _ = allocator; // Reserved for future thread pool.

    if (custom_config) |cfg| {
        config = cfg;
    }

    // Initialize global workspace.
    global_workspace = GlobalWorkspace.init();

    // Runtime CPU feature detection if SIMD enabled.
    if (config.enable_simd) {
        // TODO: Detect AVX2, AVX-512, NEON.
    }
}

/// Deinitialize the core module.
pub fn deinit() void {
    // Clean up global resources.
    global_workspace.initialized = false;
}

/// Vector with compile-time size and SIMD optimization.
/// Tiger Style: explicitly sized types, cache alignment, simple interfaces.
pub fn Vector(comptime dim: usize) type {
    // Compile-time validation.
    comptime {
        if (dim == 0) @compileError("Vector dimension must be positive");
        if (dim > Limits.max_dim) @compileError("Vector dimension exceeds maximum");
    }

    return struct {
        // Cache-aligned for SIMD (Tiger Style: ensure predictability).
        data: [dim]f32 align(Limits.cache_line_size) = [_]f32{0} ** dim,

        const Self = @This();

        /// Initialize zero vector.
        pub fn init() Self {
            return .{};
        }

        /// Initialize with random values.
        /// Scale parameter controls standard deviation.
        pub fn initRandom(rng: std.Random, scale_factor: f32) Self {
            // Assert preconditions (Tiger Style: use assertions).
            std.debug.assert(scale_factor > 0);
            std.debug.assert(scale_factor < 1000); // Sanity bound.

            var v = Self.init();
            for (&v.data) |*val| {
                val.* = rng.floatNorm(f32) * scale_factor;
            }

            // Assert postconditions.
            std.debug.assert(!std.math.isNan(v.data[0]));

            return v;
        }

        /// Compute dot product with another vector.
        /// Performance: Uses SIMD when available for ~8x speedup.
        pub fn dot(self: Self, other: Self) f32 {
            // Use SIMD if available and dimension is suitable.
            if (comptime build_options.enable_simd) {
                if (comptime dim >= 8) {
                    return self.dotSimd(other);
                }
            }

            // Scalar fallback.
            var sum: f32 = 0;
            for (self.data, other.data) |a, b| {
                sum += a * b;
            }

            // Assert result is valid.
            std.debug.assert(!std.math.isNan(sum));
            std.debug.assert(!std.math.isInf(sum));

            return sum;
        }

        /// SIMD-optimized dot product (AVX2: 8-wide).
        fn dotSimd(self: Self, other: Self) f32 {
            var sum_vec: @Vector(8, f32) = @splat(0);

            // Process 8 elements at a time.
            comptime var i = 0;
            inline while (i + 8 <= dim) : (i += 8) {
                const a: @Vector(8, f32) = self.data[i..][0..8].*;
                const b: @Vector(8, f32) = other.data[i..][0..8].*;
                sum_vec += a * b;
            }

            // Horizontal sum.
            var sum = @reduce(.Add, sum_vec);

            // Handle remainder.
            inline while (i < dim) : (i += 1) {
                sum += self.data[i] * other.data[i];
            }

            return sum;
        }

        /// Element-wise addition.
        pub fn add(self: Self, other: Self) Self {
            var result = Self.init();

            if (comptime build_options.enable_simd and dim >= 8) {
                // SIMD path.
                comptime var i = 0;
                inline while (i + 8 <= dim) : (i += 8) {
                    const a: @Vector(8, f32) = self.data[i..][0..8].*;
                    const b: @Vector(8, f32) = other.data[i..][0..8].*;
                    result.data[i..][0..8].* = a + b;
                }
                // Remainder.
                inline while (i < dim) : (i += 1) {
                    result.data[i] = self.data[i] + other.data[i];
                }
            } else {
                // Scalar path.
                for (&result.data, self.data, other.data) |*r, a, b| {
                    r.* = a + b;
                }
            }

            return result;
        }

        /// Scale vector by scalar.
        pub fn scale(self: Self, scalar: f32) Self {
            // Precondition.
            std.debug.assert(!std.math.isNan(scalar));
            std.debug.assert(@abs(scalar) < 1e10); // Sanity check.

            var result = Self.init();

            if (comptime build_options.enable_simd and dim >= 8) {
                const scalar_vec: @Vector(8, f32) = @splat(scalar);

                comptime var i = 0;
                inline while (i + 8 <= dim) : (i += 8) {
                    const v: @Vector(8, f32) = self.data[i..][0..8].*;
                    result.data[i..][0..8].* = v * scalar_vec;
                }
                inline while (i < dim) : (i += 1) {
                    result.data[i] = self.data[i] * scalar;
                }
            } else {
                for (&result.data, self.data) |*r, val| {
                    r.* = val * scalar;
                }
            }

            return result;
        }

        /// Compute L2 norm.
        pub fn norm(self: Self) f32 {
            const sum_sq = self.dot(self);
            return @sqrt(sum_sq);
        }

        /// Normalize to unit vector.
        pub fn normalize(self: Self) CoreError!Self {
            const n = self.norm();
            if (n < 1e-10) return CoreError.NumericalInstability;
            return self.scale(1.0 / n);
        }
    };
}

// Napkin math documentation (Tiger Style: design for performance early).
//
// Vector operations performance:
// - Dot product (AVX2): ~8x speedup over scalar
// - Memory bandwidth: 50 GB/s typical
// - L1 cache: 32 KB, ~4 cycles latency
// - L2 cache: 256 KB, ~12 cycles latency
// - L3 cache: 8 MB, ~40 cycles latency
// - RAM: ~100 cycles latency
//
// For dim=4096 vector:
// - Size: 4096 * 4 bytes = 16 KB (fits in L1)
// - Dot product: 4096 ops / 8 (SIMD) = 512 cycles theoretical
// - Actual: ~1000 cycles with memory access

// Tests follow Tiger Style: test correctness, properties, and performance.
test "Vector: basic operations" {
    const V = Vector(8);

    var v1 = V.init();
    var v2 = V.init();

    v1.data = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    v2.data = .{ 8, 7, 6, 5, 4, 3, 2, 1 };

    // Test dot product.
    const dot_result = v1.dot(v2);
    try std.testing.expectEqual(@as(f32, 120), dot_result);

    // Test addition.
    const sum = v1.add(v2);
    try std.testing.expectEqual(@as(f32, 9), sum.data[0]);
    try std.testing.expectEqual(@as(f32, 9), sum.data[7]);

    // Test scaling.
    const scaled = v1.scale(2.0);
    try std.testing.expectEqual(@as(f32, 2), scaled.data[0]);
    try std.testing.expectEqual(@as(f32, 16), scaled.data[7]);
}

test "Vector: properties" {
    const V = Vector(16);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const v = V.initRandom(rng, 1.0);
    const zero = V.init();

    // Property: v + 0 = v
    const v_plus_zero = v.add(zero);
    for (v.data, v_plus_zero.data) |a, b| {
        try std.testing.expectApproxEqRel(a, b, 1e-6);
    }

    // Property: v · v >= 0
    const dot_self = v.dot(v);
    try std.testing.expect(dot_self >= 0);
}

test "Vector: bounds checking" {
    // This should fail to compile if uncommented:
    // const V = Vector(Limits.max_dim + 1);

    // Valid dimension.
    const V = Vector(1024);
    const v = V.init();
    try std.testing.expect(v.data.len == 1024);
}

test "Vector: performance baseline" {
    // Establish performance baseline for CI.
    const V = Vector(4096);
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const v1 = V.initRandom(rng, 1.0);
    const v2 = V.initRandom(rng, 1.0);

    const start = std.time.nanoTimestamp();
    var sum: f32 = 0;
    for (0..1000) |_| {
        sum += v1.dot(v2);
    }
    const elapsed = std.time.nanoTimestamp() - start;

    // Prevent optimization.
    std.mem.doNotOptimizeAway(&sum);

    // Should complete in reasonable time (adjust based on CI hardware).
    // Debug builds are slower, so use a more generous timeout.
    const timeout_ns = if (std.debug.runtime_safety) 100_000_000 else 10_000_000; // 100ms debug, 10ms release
    try std.testing.expect(elapsed < timeout_ns);
}
