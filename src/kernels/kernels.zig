//! WGSL compute kernels and how to launch them through the command stream.
//!
//! Each kernel is a `<name>.wgsl` file with one entry point, `main`, and its
//! parameters in a uniform `Params` struct at binding 0. The structs below
//! mirror those, field for field, padded to 16 bytes. Weights stay bf16 on
//! the GPU (two per u32); activations and accumulation are f32.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cmd = @import("../gpu/cmd.zig");

pub const Kernel = enum {
    rmsnorm,
    matmul_bf16,
    silu_mul,
    add,
    embed_bf16,

    pub fn source(k: Kernel) []const u8 {
        return switch (k) {
            inline else => |tag| @embedFile(@tagName(tag) ++ ".wgsl"),
        };
    }
};

/// Matrix multiply tile sizes (matmul_bf16.wgsl: TN outputs × TT tokens).
pub const matmul_tile_n = 64;
pub const matmul_tile_t = 16;
/// Reduction step; `k` must be a multiple.
pub const matmul_tile_k = 32;

pub const params = struct {
    pub const RmsNorm = extern struct { rows: u32, dim: u32, eps: f32, _pad: u32 = 0 };
    pub const MatmulBf16 = extern struct {
        rows: u32,
        n: u32,
        k: u32,
        col0: u32,
        out_stride: u32,
        _pad: [3]u32 = @splat(0),
    };
    pub const Count = extern struct { count: u32, _pad: [3]u32 = @splat(0) };
    pub const EmbedBf16 = extern struct { dim: u32, row0: u32, chunk_rows: u32, _pad: u32 = 0 };
};

/// One pipeline per kernel.
pub const Set = struct {
    pipelines: std.EnumArray(Kernel, cmd.Pipeline),

    /// Records shader and pipeline creation for every kernel.
    pub fn create(gpa: Allocator, rec: *cmd.Recorder) Allocator.Error!Set {
        var set: Set = .{ .pipelines = undefined };
        for (std.enums.values(Kernel)) |k| {
            const shader = try rec.createShader(gpa, k.source());
            set.pipelines.set(k, try rec.createPipeline(gpa, shader));
        }
        return set;
    }
};

/// A buffer range bound at binding 1, 2, ... in kernel order.
pub const Arg = struct {
    buffer: cmd.Buffer,
    offset: u64 = 0,
    size: u64,
};

/// Records kernel launches. Parameter blocks are copied into `arena`, which
/// must outlive execution of the recorded batch.
pub const Launcher = struct {
    gpa: Allocator,
    arena: Allocator,
    rec: *cmd.Recorder,
    set: *const Set,

    pub fn launch(l: Launcher, kernel: Kernel, p: anytype, args: []const Arg, groups: [3]u32) Allocator.Error!void {
        const P = @TypeOf(p);
        comptime std.debug.assert(@sizeOf(P) % 16 == 0);
        const bytes = try l.arena.dupe(u8, std.mem.asBytes(&p));
        const ub = try l.rec.createBuffer(l.gpa, @sizeOf(P), .{ .uniform = true, .copy_dst = true });
        try l.rec.writeBuffer(l.gpa, ub, 0, bytes);

        var entries: [16]cmd.Binding = undefined;
        entries[0] = .{ .binding = 0, .buffer = ub, .size = @sizeOf(P) };
        for (args, 1..) |a, i| entries[i] = .{ .binding = @intCast(i), .buffer = a.buffer, .offset = a.offset, .size = a.size };
        const pipeline = l.set.pipelines.get(kernel);
        const bg = try l.rec.createBindGroup(l.gpa, pipeline, 0, entries[0 .. args.len + 1]);
        try l.rec.dispatch(l.gpa, pipeline, groups, &.{bg});
    }

    /// Workgroup grid for `count` threads at 256 per group, folded into y
    /// past the 65535-per-dimension limit.
    pub fn elementwise(count: usize) [3]u32 {
        const groups = std.math.divCeil(usize, count, 256) catch unreachable;
        if (groups <= 65535) return .{ @intCast(groups), 1, 1 };
        return .{ 65535, @intCast(std.math.divCeil(usize, groups, 65535) catch unreachable), 1 };
    }
};
