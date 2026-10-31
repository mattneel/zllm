//! The fusion pass (spec §6): a step program's ops as a small graph over
//! buffers, fused where legal, then lowered to launches. A fused kernel is
//! generated from hand-written fragments: the backend's kernel templates
//! take the fused part as a template parameter (a prologue staged with
//! each input), so fusing never leaves the hand-written code.
//!
//! Legality is the backend-independent part, decided here over the graph:
//! a producer fuses into its consumer only if its output has that one
//! reader (in the graph, and none outside it, which the caller declares),
//! the consumer's kernel has the fragment for every weight chunk, and the
//! shapes fit the fragment's layout. What the fragments are written in is
//! the backend's (today WGSL, kernels/matvec_quant.wgsl.hbs).

const std = @import("std");
const Allocator = std.mem.Allocator;
const kernels = @import("../kernels/kernels.zig");
const common = @import("../plan/common.zig");
const Arg = kernels.Arg;
const P = kernels.params;

pub const Op = union(enum) {
    /// dst[t][j] = silu(src[t · stride + j] / div), j < rank (hc_act).
    act: struct { src: Arg, dst: Arg, stride: u32, rank: u32, div: u32 },
    /// out = w · x on the decode path, with the multiply's own fusions.
    vector: struct { out: Arg, x: Arg, w: *const common.Matrix, fuse: common.Mm.Fuse = .{} },

    fn reads(op: Op, a: Arg) bool {
        return switch (op) {
            .act => |o| same(o.src, a),
            .vector => |o| same(o.x, a) or (if (o.fuse.norm) |n| same(n, a) else false),
        };
    }
};

fn same(a: Arg, b: Arg) bool {
    return a.buffer == b.buffer and a.offset == b.offset;
}

pub const Options = struct {
    /// Off: every op lowers to its own launches (the unfused reference the
    /// fused path is checked against).
    enabled: bool = true,
    /// Buffers read outside the ops (never fused away).
    live_out: []const Arg = &.{},
};

/// What the pass did, for reports.
pub const Stats = struct { fused: u32 = 0, kept: u32 = 0 };

/// Lowers `ops` to launches, fusing where legal.
pub fn lower(arena: Allocator, mm: common.Mm, ops: []const Op, o: Options, stats: ?*Stats) common.BuildError![]const kernels.Launch {
    var out: std.ArrayList(kernels.Launch) = .empty;
    var skip = try arena.alloc(bool, ops.len);
    @memset(skip, false);
    for (ops, 0..) |op, i| {
        if (skip[i]) continue;
        switch (op) {
            .act => |a| {
                if (o.enabled) if (actConsumer(ops, i, o.live_out)) |c| {
                    var v = ops[c].vector;
                    v.fuse.act = .{ .stride = a.stride, .div = a.div };
                    try out.appendSlice(arena, try mm.vector(v.out, a.src, v.w, v.fuse));
                    skip[c] = true;
                    if (stats) |s| s.fused += 1;
                    continue;
                };
                if (stats) |s| s.kept += 1;
                try out.append(arena, try mm.b.add(.hc_act, P.HcAct{
                    .count = kernels.decode_max_rows * a.rank,
                    .hc = a.div,
                    .rank = a.rank,
                    .stride = a.stride,
                }, &.{ a.src, a.dst }, .{ .elems = a.rank }));
            },
            .vector => |v| try out.appendSlice(arena, try mm.vector(v.out, v.x, v.w, v.fuse)),
        }
    }
    return out.items;
}

/// The op `act` (ops[i]) can fuse into, if legal: the one reader of its
/// output, a plain decode multiply whose every chunk is quantized (the
/// prologue is in the quantized template), with whole vectors per row.
fn actConsumer(ops: []const Op, i: usize, live_out: []const Arg) ?usize {
    const a = ops[i].act;
    for (live_out) |l| if (same(l, a.dst)) return null;
    var reader: ?usize = null;
    for (ops, 0..) |op, j| {
        if (j == i or !op.reads(a.dst)) continue;
        if (reader != null) return null;
        reader = j;
    }
    const c = reader orelse return null;
    if (c < i or ops[c] != .vector) return null;
    const v = ops[c].vector;
    if (!same(v.x, a.dst) or v.fuse.norm != null or v.fuse.act != null or v.w.k != a.rank) return null;
    if (a.stride % 4 != 0 or a.rank % 4 != 0 or a.div == 0) return null;
    for (v.w.chunks) |ch| if (kernels.isHalf(ch.type)) return null;
    return c;
}

test "act fuses into its one reader only, and only over quantized weights" {
    const buf = struct {
        fn arg(id: u32) Arg {
            return .{ .buffer = @fromBackingInt(@intCast(id)), .size = 1024 };
        }
    }.arg;
    const q8 = [_]common.Matrix.Chunk{.{ .arg = buf(9), .row0 = 0, .rows = 16, .type = .q8_0 }};
    const half = [_]common.Matrix.Chunk{.{ .arg = buf(9), .row0 = 0, .rows = 16, .type = .bf16 }};
    const up: common.Matrix = .{ .n = 16, .k = 64, .chunks = &q8 };
    const up_half: common.Matrix = .{ .n = 16, .k = 64, .chunks = &half };
    const act: Op = .{ .act = .{ .src = buf(1), .dst = buf(2), .stride = 68, .rank = 64, .div = 4 } };
    const use: Op = .{ .vector = .{ .out = buf(3), .x = buf(2), .w = &up } };

    try std.testing.expectEqual(@as(?usize, 1), actConsumer(&.{ act, use }, 0, &.{}));
    // A second reader keeps the activation's output.
    try std.testing.expectEqual(@as(?usize, null), actConsumer(&.{ act, use, use }, 0, &.{}));
    // So does a reader outside the graph.
    try std.testing.expectEqual(@as(?usize, null), actConsumer(&.{ act, use }, 0, &.{buf(2)}));
    // The prologue is in the quantized kernel only.
    const use_half: Op = .{ .vector = .{ .out = buf(3), .x = buf(2), .w = &up_half } };
    try std.testing.expectEqual(@as(?usize, null), actConsumer(&.{ act, use_half }, 0, &.{}));
    // Rows must be whole vectors.
    var odd = act;
    odd.act.stride = 66;
    try std.testing.expectEqual(@as(?usize, null), actConsumer(&.{ odd, use }, 0, &.{}));
}
