//! Velocity's kernels: for each kernels.Kernel it implements, the HLSL in
//! kernels/<name>.hlsl, behind the shared header kernels/common.hlsli. Each
//! keeps its WGSL counterpart's contract (params block, bindings, grid) and
//! is written for Direct3D 12. Kernels not listed are missing on Velocity
//! (kernels.Set.missing).

const std = @import("std");
const Kernel = @import("../kernels/kernels.zig").Kernel;

const common = @embedFile("kernels/common.hlsli");

/// The kernels Velocity implements.
const implemented = [_]Kernel{.rmsnorm};

/// Kernel `k`'s HLSL with the shared header, or null where Velocity does
/// not implement it.
pub fn source(k: Kernel) ?[]const u8 {
    inline for (implemented) |i| {
        if (k == i) return comptime common ++ @embedFile("kernels/" ++ @tagName(i) ++ ".hlsl");
    }
    return null;
}

test "every implemented kernel's source is unique and non-empty" {
    inline for (implemented, 0..) |a, i| {
        try std.testing.expect(source(a).?.len > common.len);
        inline for (implemented[i + 1 ..]) |b| try std.testing.expect(a != b);
    }
}
