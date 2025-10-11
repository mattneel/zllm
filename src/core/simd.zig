//! SIMD intrinsics and vectorized operations
//!
//! Tiger Style:
//! - Platform-specific SIMD at compile time
//! - Fallback to scalar operations when SIMD unavailable
//! - Explicit alignment requirements
//! - Zero overhead abstractions

const std = @import("std");
const Config = @import("lib.zig").Config;
const builtin = @import("builtin");

/// SIMD vector width for current platform
pub const simd_width: comptime_int = switch (builtin.cpu.arch) {
    .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) 8 else 4,
    .aarch64 => 4, // NEON
    else => 1, // Scalar fallback
};

/// Whether SIMD is enabled for this build
pub const simd_enabled: bool = Config.enable_simd and simd_width > 1;

/// SIMD-optimized dot product
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    if (!simd_enabled or a.len < simd_width) {
        // Scalar fallback
        var sum: f32 = 0;
        for (a, b) |ai, bi| {
            sum += ai * bi;
        }
        return sum;
    }

    // TODO: Implement SIMD version using @Vector
    var sum: f32 = 0;
    for (a, b) |ai, bi| {
        sum += ai * bi;
    }
    return sum;
}

/// SIMD-optimized vector addition
pub fn vectorAdd(dst: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(dst.len == a.len);
    std.debug.assert(dst.len == b.len);

    if (!simd_enabled or dst.len < simd_width) {
        // Scalar fallback
        for (dst, a, b) |*d, ai, bi| {
            d.* = ai + bi;
        }
        return;
    }

    // TODO: Implement SIMD version using @Vector
    for (dst, a, b) |*d, ai, bi| {
        d.* = ai + bi;
    }
}

test "simd configuration" {
    try std.testing.expect(simd_width >= 1);
}

test "dot product" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    const result = dotProduct(&a, &b);
    try std.testing.expectEqual(@as(f32, 70), result); // 1*5 + 2*6 + 3*7 + 4*8
}

test "vector addition" {
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var dst: [4]f32 = undefined;

    vectorAdd(&dst, &a, &b);

    try std.testing.expectEqual(@as(f32, 6), dst[0]);
    try std.testing.expectEqual(@as(f32, 8), dst[1]);
    try std.testing.expectEqual(@as(f32, 10), dst[2]);
    try std.testing.expectEqual(@as(f32, 12), dst[3]);
}
