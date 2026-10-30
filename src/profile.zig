//! `zllm profile`: GPU time per kernel, from timestamp queries around every
//! dispatch (Host.startProfile). After filling the context to `at` tokens,
//! it runs prefill steps of `chunk` random tokens, then decode steps with
//! sampling, then (with an MTP block) speculative rounds: the block over two
//! positions with the draft, then the two-position verify step. It reports
//! each kernel's time per step or round. Each dispatch then runs in its own compute pass, so the totals
//! can exceed an unprofiled step, which is timed too for comparison.

const std = @import("std");
const Io = std.Io;
const zllm = @import("zllm");
const gen = @import("generate.zig");
const Kernel = zllm.kernels.Kernel;

const prefill_steps = 6;
const decode_steps = 16;

pub fn run(rt: *gen.Runtime, chunk: u32, at: u32, out: *Io.Writer) !void {
    const gpa = rt.gpa;
    const p = &rt.program;
    var rec: zllm.gpu.cmd.Recorder = .{};
    defer rec.deinit(gpa);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const total = at + chunk * (prefill_steps + 1);
    if (total + decode_steps * 4 > p.context() or chunk > p.maxTokens()) return error.ChunkTooLarge;
    var prng: std.Random.DefaultPrng = .init(1);
    const tokens = try arena.alloc(u32, total);
    for (tokens) |*t| t.* = prng.random().uintLessThan(u32, p.vocab());
    // Decode steps process the previous sample, or where the program needs
    // the token from the host, a fixed one.
    const decode_token: ?[]const u32 = if (p.chainsSamples()) null else tokens[0..1];
    rt.engine.forget();
    try p.recordReset(gpa, &rec);
    try p.recordRequest(gpa, &rec, .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .presence_penalty = 1.5 }, tokens, &.{});

    // Fill to `at`, unprofiled, in steps of `chunk`.
    var pos: u32 = 0;
    while (pos < at) {
        const m = @min(chunk, at - pos);
        try p.recordStep(gpa, &rec, .{ .pos0 = pos, .tokens = tokens[pos..][0..m] });
        try rec.submit(gpa, 1);
        _ = try time(rt, &rec);
        pos += m;
    }
    // One unprofiled step to warm up and to time a normal step.
    try p.recordStep(gpa, &rec, .{ .pos0 = pos, .tokens = tokens[pos..][0..chunk] });
    try rec.submit(gpa, 1);
    const plain_prefill = try time(rt, &rec);
    pos += chunk;

    try rt.host.startProfile();
    defer rt.host.stopProfile();
    var wall: u64 = 0;
    for (0..prefill_steps) |_| {
        try p.recordStep(gpa, &rec, .{ .pos0 = pos, .tokens = tokens[pos..][0..chunk] });
        try rec.submit(gpa, 1);
        wall += try time(rt, &rec);
        pos += chunk;
    }
    try out.print("{s}: prefill steps of {d} tokens at positions {d}..{d}: {d:.2} ms per step unprofiled, {d:.2} ms profiled\n", .{
        rt.adapter, chunk, at + chunk, pos, ms(plain_prefill), ms(wall / prefill_steps),
    });
    try report(rt, prefill_steps, out);

    rt.host.stopProfile();
    var sampled: [2]u32 = undefined;
    try p.recordStep(gpa, &rec, .{ .pos0 = pos, .tokens = decode_token, .logit_rows = 1, .draw = 0.5, .sampled = &sampled });
    try rec.submit(gpa, 1);
    const plain_decode = try time(rt, &rec);
    pos += 1;
    try rt.host.startProfile();
    wall = 0;
    for (0..decode_steps) |_| {
        try p.recordStep(gpa, &rec, .{ .pos0 = pos, .tokens = decode_token, .logit_rows = 1, .draw = 0.5, .sampled = &sampled });
        try rec.submit(gpa, 1);
        wall += try time(rt, &rec);
        pos += 1;
    }
    try out.print("\ndecode steps at position {d}: {d:.2} ms per step unprofiled, {d:.2} ms profiled\n", .{
        pos, ms(plain_decode), ms(wall / decode_steps),
    });
    try report(rt, decode_steps, out);

    if (p.mtp()) {
        rt.host.stopProfile();
        var verdicts: [2][2]u32 = undefined;
        var draft: u32 = undefined;
        const Round = struct {
            fn record(pp: *zllm.plan.Program, a: std.mem.Allocator, r: *zllm.gpu.cmd.Recorder, at_pos: u32, v: *[2][2]u32, d: *u32) !void {
                try pp.recordMtp(a, r, .{ .pos0 = at_pos - 2, .tokens = &.{ 1, 2 }, .draft = true });
                try pp.recordVerify(a, r, .{ .pos0 = at_pos, .token = 2, .draws = .{ 0.5, 0.5 }, .sampled = v, .draft = d });
            }
        };
        try Round.record(p, gpa, &rec, pos, &verdicts, &draft);
        try rec.submit(gpa, 1);
        const plain_round = try time(rt, &rec);
        pos += 2;
        try rt.host.startProfile();
        wall = 0;
        for (0..decode_steps) |_| {
            try Round.record(p, gpa, &rec, pos, &verdicts, &draft);
            try rec.submit(gpa, 1);
            wall += try time(rt, &rec);
            pos += 2;
        }
        try out.print("\nspeculative rounds (MTP over 2 positions, then verify) at position {d}: {d:.2} ms per round unprofiled, {d:.2} ms profiled\n", .{
            pos, ms(plain_round), ms(wall / decode_steps),
        });
        try report(rt, decode_steps, out);
    }
    try out.flush();
}

/// Each kernel's GPU time and dispatches per step, largest first, then
/// the largest launches by grid (each a matrix shape, for the multiplies).
fn report(rt: *gen.Runtime, steps: u64, out: *Io.Writer) !void {
    const pr = rt.host.profile.?;
    const Total = zllm.gpu.Host.Profile.Total;
    const Row = struct { name: []const u8, grid: ?[3]u32 = null, t: Total };
    var names: std.AutoHashMapUnmanaged(u32, []const u8) = .empty;
    defer names.deinit(rt.gpa);
    for (std.enums.values(Kernel)) |k| try names.put(rt.gpa, @backingInt(rt.set.pipelines.get(k)), @tagName(k));

    var by_kernel: std.StringArrayHashMapUnmanaged(Total) = .empty;
    defer by_kernel.deinit(rt.gpa);
    var by_grid: std.ArrayList(Row) = .empty;
    defer by_grid.deinit(rt.gpa);
    var sum: u64 = 0;
    for (pr.totals.keys(), pr.totals.values()) |key, t| {
        const name = names.get(key.pipeline) orelse "?";
        const k = try by_kernel.getOrPutValue(rt.gpa, name, .{});
        k.value_ptr.ns += t.ns;
        k.value_ptr.count += t.count;
        try by_grid.append(rt.gpa, .{ .name = name, .grid = key.grid, .t = t });
        sum += t.ns;
    }
    var rows: std.ArrayList(Row) = .empty;
    defer rows.deinit(rt.gpa);
    for (by_kernel.keys(), by_kernel.values()) |name, t| try rows.append(rt.gpa, .{ .name = name, .t = t });
    const more = struct {
        fn f(_: void, a: Row, b: Row) bool {
            return a.t.ns > b.t.ns;
        }
    }.f;
    std.mem.sort(Row, rows.items, {}, more);
    std.mem.sort(Row, by_grid.items, {}, more);

    try out.print("{s:<20} {s:>12} {s:>14} {s:>12} {s:>7}\n", .{ "kernel", "ms per step", "dispatches", "us each", "share" });
    for (rows.items) |r| try printRow(out, r.name, null, r.t, steps, sum);
    try out.print("{s:<20} {d:>12.3}\n", .{ "sum", ms(sum / steps) });
    try out.print("largest launches by grid (x, y, z):\n", .{});
    for (by_grid.items[0..@min(12, by_grid.items.len)]) |r| try printRow(out, r.name, r.grid, r.t, steps, sum);
    if (pr.untimed > 0) try out.print("({d} dispatches past {d} per submission were not timed)\n", .{ pr.untimed, zllm.gpu.Host.Profile.capacity });
}

fn printRow(out: *Io.Writer, name: []const u8, grid: ?[3]u32, t: zllm.gpu.Host.Profile.Total, steps: u64, sum: u64) !void {
    var label_buf: [64]u8 = undefined;
    const label = if (grid) |g| try std.fmt.bufPrint(&label_buf, "{s} {d},{d},{d}", .{ name, g[0], g[1], g[2] }) else name;
    try out.print("{s:<20} {d:>12.3} {d:>14} {d:>12.1} {d:>6.1}%\n", .{
        label,
        ms(t.ns / steps),
        t.count / steps,
        @as(f64, @floatFromInt(t.ns)) / @as(f64, @floatFromInt(t.count)) / 1e3,
        100 * @as(f64, @floatFromInt(t.ns)) / @as(f64, @floatFromInt(sum)),
    });
}

fn time(rt: *gen.Runtime, rec: *zllm.gpu.cmd.Recorder) !u64 {
    const t0 = Io.Timestamp.now(rt.io, .awake);
    try rt.execute(rec.batch());
    rec.clear();
    return @intCast(t0.durationTo(Io.Timestamp.now(rt.io, .awake)).toNanoseconds());
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}
