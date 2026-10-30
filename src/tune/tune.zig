//! Load-time tuning (spec §5): for each shape of quantized decode multiply
//! in the model, every MatvecVariant candidate the device fits runs on the
//! shape's real weights. Each is checked against the default variant first
//! (diverging candidates are dropped), then timed: 2 warm-up batches more,
//! then the median of 7. The fastest per shape goes into plan_common.Tuning.
//!
//! A non-blocking state machine, like the engine: the host runs each batch
//! `next` returns and passes its time to `done`, until `next` returns null.
//! Times are of whole batches (submission to completion), which every host
//! measures; each batch repeats its dispatch to last about a millisecond, so
//! the fixed cost of a submission stays small next to it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cmd = @import("../gpu/cmd.zig");
const gguf = @import("../gguf/gguf.zig");
const kernels = @import("../kernels/kernels.zig");
const plan_common = @import("../plan/common.zig");

const P = kernels.params;
const Variant = kernels.MatvecVariant;
const Arg = kernels.Arg;

/// Timed batches per candidate, and warm-up batches before them (the first
/// is the correctness run).
pub const timed = 7;
pub const warmup = 3;
/// Relative L2 error a candidate may have against the default variant: they
/// differ only in the order of each row's sums.
pub const tolerance = 1e-5;

/// One shape to tune: the first chunk of a matrix of its (type, n, k).
const Shape = struct {
    key: plan_common.Tuning.Entry,
    chunk: Arg,
    rows: u32,
    /// Times each batch repeats the dispatch.
    reps: u32,
};

pub const Sweep = struct {
    gpa: Allocator,
    /// Rendered variant sources and readbacks; must outlive the sweep.
    arena: Allocator,
    set: *kernels.Set,
    /// The loader's: object ids are dense per recorder, so the sweep's
    /// buffers and pipelines are created through it. Empty between batches.
    rec: *cmd.Recorder,
    shapes: []const Shape,
    candidates: []const Variant,
    tokens: u32,
    x: Arg,
    out: Arg,
    /// The default variant's output for the current shape.
    reference: []f32,
    got: []f32,
    params: Arg,
    param_value: P.MatvecBf16 = undefined,
    /// The current candidate's pipeline and bindings.
    pipeline: cmd.Pipeline = undefined,
    bind_group: cmd.BindGroup = undefined,

    shape: usize = 0,
    candidate: usize = 0,
    batch: u32 = 0,
    times: [timed]u64 = undefined,
    diverged: bool = false,
    best: ?struct { v: Variant, ns: u64 } = null,
    entries: std.ArrayList(plan_common.Tuning.Entry) = .empty,
    /// Candidates dropped for diverging from the default.
    dropped: u32 = 0,

    /// A sweep over `m`'s quantized decode multiplies, `tokens` per step (1,
    /// or 2 where decoding speculates): `m` is any model with `eachMatrix`
    /// (plan.Model, or one architecture's). Records the scratch buffers'
    /// creation into `rec`, for the first batch.
    pub fn init(gpa: Allocator, arena: Allocator, rec: *cmd.Recorder, set: *kernels.Set, m: anytype, d: kernels.DeviceLimits, tokens: u32) !Sweep {
        std.debug.assert(tokens >= 1 and tokens <= kernels.decode_max_rows);
        const Collect = struct {
            arena: Allocator,
            list: std.ArrayList(Shape) = .empty,
            k_max: u32 = 0,
            n_max: u32 = 0,
            failed: bool = false,
            fn add(c: *@This(), w: *const plan_common.Matrix) void {
                for (w.chunks) |ch| {
                    if (kernels.isHalf(ch.type)) continue;
                    const key: plan_common.Tuning.Entry = .{ .type = ch.type, .n = w.n, .k = w.k, .variant = .{} };
                    const seen = for (c.list.items) |s| {
                        if (s.key.type == key.type and s.key.n == key.n and s.key.k == key.k) break true;
                    } else false;
                    if (seen) continue;
                    const bytes = ch.arg.size;
                    const reps: u32 = @intCast(std.math.clamp(@divTrunc(256 << 20, @max(bytes, 1)), 2, 64));
                    c.list.append(c.arena, .{ .key = key, .chunk = ch.arg, .rows = ch.rows, .reps = reps }) catch {
                        c.failed = true;
                    };
                    c.k_max = @max(c.k_max, w.k);
                    c.n_max = @max(c.n_max, ch.rows);
                }
            }
        };
        var c: Collect = .{ .arena = arena };
        m.eachMatrix(&c, Collect.add);
        if (c.failed) return error.OutOfMemory;

        var candidates: std.ArrayList(Variant) = .empty;
        for (Variant.candidates) |v| if (v.fits(d)) try candidates.append(arena, v);
        // The default first: it is the reference the others must match.
        std.debug.assert(std.meta.eql(Variant.candidates[0], Variant{}));

        var s: Sweep = .{
            .gpa = gpa,
            .arena = arena,
            .set = set,
            .rec = rec,
            .shapes = c.list.items,
            .candidates = candidates.items,
            .tokens = tokens,
            .x = undefined,
            .out = undefined,
            .reference = try arena.alloc(f32, @as(usize, tokens) * c.n_max),
            .got = try arena.alloc(f32, @as(usize, tokens) * c.n_max),
            .params = undefined,
        };
        if (s.shapes.len == 0) return s;
        const x_count = @as(u64, tokens) * c.k_max;
        s.x = .{ .buffer = try s.rec.createBuffer(gpa, x_count * 4, .{ .storage = true, .copy_dst = true }), .size = x_count * 4 };
        const out_bytes = @as(u64, tokens) * c.n_max * 4;
        s.out = .{ .buffer = try s.rec.createBuffer(gpa, out_bytes, .{ .storage = true, .copy_src = true }), .size = out_bytes };
        s.params = .{ .buffer = try s.rec.createBuffer(gpa, @sizeOf(P.MatvecBf16), .{ .uniform = true, .copy_dst = true }), .size = @sizeOf(P.MatvecBf16) };
        // Deterministic inputs of RMS about 1, as normed activations are.
        const xs = try arena.alloc(f32, x_count);
        for (xs, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.7);
        try s.rec.writeBuffer(gpa, s.x.buffer, 0, std.mem.sliceAsBytes(xs));
        return s;
    }

    /// The next batch to run, or null when the sweep is done.
    pub fn next(s: *Sweep) !?[]const u32 {
        if (s.shape == s.shapes.len) return null;
        const sh = s.shapes[s.shape];
        const v = s.candidates[s.candidate];
        if (s.batch == 0) {
            const kernel = kernels.Kernel.weight(.matvec, sh.key.type).?;
            s.pipeline = try s.set.matvec(s.gpa, s.arena, s.rec, kernel, v);
            s.param_value = .{ .n = sh.rows, .k = sh.key.k, .col0 = 0, .out_stride = sh.rows, .eps = 1e-6, .flags = .{} };
            try s.rec.writeBuffer(s.gpa, s.params.buffer, 0, std.mem.asBytes(&s.param_value));
            const x_bytes = @as(u64, s.tokens) * sh.key.k * 4;
            s.bind_group = try s.rec.createBindGroup(s.gpa, s.pipeline, 0, &.{
                .{ .binding = 0, .buffer = s.params.buffer, .offset = 0, .size = s.params.size },
                .{ .binding = 1, .buffer = s.x.buffer, .offset = 0, .size = x_bytes },
                .{ .binding = 2, .buffer = sh.chunk.buffer, .offset = sh.chunk.offset, .size = sh.chunk.size },
                .{ .binding = 3, .buffer = s.out.buffer, .offset = 0, .size = @as(u64, s.tokens) * sh.rows * 4 },
                // The norm weights, unread without the norm flag.
                .{ .binding = 4, .buffer = s.x.buffer, .offset = 0, .size = x_bytes },
            });
        }
        const groups: [3]u32 = .{ std.math.divCeil(u32, sh.rows, v.groupRows()) catch unreachable, s.tokens, 1 };
        const reps: u32 = if (s.batch == 0) 1 else sh.reps;
        for (0..reps) |_| try s.rec.dispatch(s.gpa, s.pipeline, groups, &.{s.bind_group});
        if (s.batch == 0) {
            const dst = if (s.candidate == 0) s.reference else s.got;
            try s.rec.readBuffer(s.gpa, read_token, s.out.buffer, 0, std.mem.sliceAsBytes(dst[0 .. @as(usize, s.tokens) * sh.rows]));
        }
        try s.rec.submit(s.gpa, 0);
        return s.rec.batch();
    }

    /// The batch `next` returned ran in `elapsed_ns` (its readback done).
    pub fn done(s: *Sweep, elapsed_ns: u64) !void {
        s.rec.clear();
        const sh = s.shapes[s.shape];
        const n = @as(usize, s.tokens) * sh.rows;
        if (s.batch == 0 and s.candidate > 0) s.diverged = relErr(s.got[0..n], s.reference[0..n]) > tolerance;
        if (s.batch >= warmup) s.times[s.batch - warmup] = elapsed_ns;
        s.batch += 1;
        // A diverging candidate stops after its correctness run.
        if (!s.diverged and s.batch < warmup + timed) return;

        if (s.diverged) {
            s.dropped += 1;
        } else {
            std.mem.sort(u64, &s.times, {}, std.sort.asc(u64));
            const ns = s.times[timed / 2];
            if (s.best == null or ns < s.best.?.ns) s.best = .{ .v = s.candidates[s.candidate], .ns = ns };
        }
        s.batch = 0;
        s.diverged = false;
        s.candidate += 1;
        if (s.candidate < s.candidates.len) return;

        // The shape is done: keep its winner unless it is the default.
        const best = s.best.?.v;
        if (!std.meta.eql(best, Variant{})) {
            var e = sh.key;
            e.variant = best;
            try s.entries.append(s.arena, e);
        }
        s.best = null;
        s.candidate = 0;
        s.shape += 1;
    }

    /// The winners (valid while the arena is).
    pub fn tuning(s: *const Sweep) plan_common.Tuning {
        return .{ .entries = s.entries.items };
    }

    /// Shapes tuned so far, and in total.
    pub fn progress(s: *const Sweep) [2]usize {
        return .{ s.shape, s.shapes.len };
    }
};

pub const read_token = 0xffff_fffd;

fn relErr(got: []const f32, want: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, want) |g, w| {
        const d = @as(f64, g) - w;
        num += d * d;
        den += @as(f64, w) * w;
    }
    if (std.math.isNan(num)) return std.math.inf(f64);
    return if (den == 0) @sqrt(num) else @sqrt(num / den);
}
