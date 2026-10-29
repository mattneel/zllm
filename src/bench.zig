//! `zllm bench`: where a decode step's time goes. After prefilling a
//! context of random tokens, it times whole decode steps (sampling with the
//! Qwen thinking defaults) and splits off the per-submission overhead. Then
//! each kernel class's share: the time a step's launches save, replayed in
//! order, when that class is left out.

const std = @import("std");
const Io = std.Io;
const zllm = @import("zllm");
const gen = @import("generate.zig");
const plan = zllm.plan.qwen35;
const Kernel = zllm.kernels.Kernel;

const steps = 32;

pub fn run(rt: *gen.Runtime, context: u32, out: *Io.Writer) !void {
    const gpa = rt.gpa;
    const p = &rt.program;
    var rec: zllm.gpu.cmd.Recorder = .{};
    defer rec.deinit(gpa);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Prefill `context` random tokens.
    var prng: std.Random.DefaultPrng = .init(1);
    const tokens = try arena.alloc(u32, context);
    for (tokens) |*t| t.* = prng.random().uintLessThan(u32, p.hp.vocab);
    rt.engine.forget();
    try p.recordReset(gpa, &rec);
    try p.recordRequest(gpa, &rec, .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .presence_penalty = 1.5 }, tokens, &.{});
    var pos: u32 = 0;
    while (pos < context) {
        const m = @min(p.cfg.max_tokens, context - pos);
        try p.recordStep(gpa, &rec, .{ .pos0 = pos, .tokens = tokens[pos..][0..m] });
        try rec.submit(gpa, 1);
        _ = try time(rt, &rec);
        pos += m;
    }

    // Whole decode steps, sampling.
    var sampled: [2]u32 = undefined;
    var total: u64 = 0;
    var best: u64 = std.math.maxInt(u64);
    for (0..steps) |_| {
        try p.recordStep(gpa, &rec, .{ .pos0 = pos, .tokens = null, .logit_rows = 1, .draw = 0.5, .sampled = &sampled });
        try rec.submit(gpa, 1);
        const ns = try time(rt, &rec);
        total += ns;
        best = @min(best, ns);
        pos += 1;
    }
    try rec.submit(gpa, 1);
    var empty: u64 = std.math.maxInt(u64);
    for (0..8) |_| empty = @min(empty, try time(rt, &rec));

    // The same step without sampling, alone and twice per submission:
    // the difference is one step's encoding and GPU time, the rest of a
    // lone step is per-submission overhead.
    var one: u64 = std.math.maxInt(u64);
    var two: u64 = std.math.maxInt(u64);
    var encode: u64 = std.math.maxInt(u64);
    for (0..8) |_| {
        const t0 = Io.Timestamp.now(rt.io, .awake);
        try p.recordStep(gpa, &rec, .{ .pos0 = pos, .tokens = null, .logit_rows = 1 });
        encode = @min(encode, @as(u64, @intCast(t0.durationTo(Io.Timestamp.now(rt.io, .awake)).toNanoseconds())));
        try rec.submit(gpa, 1);
        one = @min(one, try time(rt, &rec));
        for (0..2) |_| try p.recordStep(gpa, &rec, .{ .pos0 = pos, .tokens = null, .logit_rows = 1 });
        try rec.submit(gpa, 1);
        two = @min(two, try time(rt, &rec));
    }

    try out.print("{s}: decode at position {d}: {d:.2} ms per step (best {d:.2}), {d:.0} tok/s; empty submission {d:.3} ms\n", .{
        rt.adapter, pos, ms(total / steps), ms(best), 1e9 / @as(f64, @floatFromInt(total / steps)), ms(empty),
    });
    try out.print("without sampling: {d:.2} ms alone, {d:.2} ms for two in one submission; per-submission overhead {d:.2} ms; recording a step {d:.3} ms\n", .{
        ms(one), ms(two), ms(2 * one -| two), ms(encode),
    });

    // The floor under every dispatch: the final RMSNorm over one row (one
    // workgroup), 400 times in a row, each depending on the last.
    const tiny = for (p.body) |item| switch (item) {
        .final_norm => |l| break l,
        else => {},
    } else unreachable;
    var chain: u64 = std.math.maxInt(u64);
    for (0..5) |_| {
        for (0..400) |_| try tiny.record(gpa, &rec, .{ .rows = 1 });
        try rec.submit(gpa, 1);
        chain = @min(chain, try time(rt, &rec));
    }
    try out.print("a tiny dependent dispatch costs {d:.1} us\n\n", .{@as(f64, @floatFromInt(chain)) / 400 / 1e3});

    // Each kernel class's share: the decode step replayed in order without
    // that class, so barriers and caching stay as in a real step.
    const n_class = std.enums.values(Kernel).len + 1;
    const head_class = n_class - 1;
    const Entry = struct { class: usize, launch: zllm.kernels.Launch };
    var seq: std.ArrayList(Entry) = .empty;
    for (p.body) |item| switch (item) {
        .launch => |l| try seq.append(arena, .{ .class = @backingInt(l.kernel), .launch = l }),
        .fork => |f| for (f.decode) |l| try seq.append(arena, .{ .class = @backingInt(l.kernel), .launch = l }),
        .final_norm, .layer_end => {},
    };
    for (p.head.decode) |l| try seq.append(arena, .{ .class = head_class, .launch = l });
    for (p.sample_steps[0..p.sample_len]) |ss| try seq.append(arena, .{ .class = @backingInt(Kernel.sample), .launch = p.sample.get(ss.phase)[ss.round] });

    const full = try replay(rt, &rec, seq.items, null, pos);
    try out.print("{s:<24} {s:>10} {s:>12} {s:>7}\n", .{ "kernel", "dispatches", "ms per step", "share" });
    var sum: f64 = 0;
    for (0..n_class) |class| {
        var count: usize = 0;
        for (seq.items) |e| count += @intFromBool(e.class == class);
        if (count == 0) continue;
        const without = try replay(rt, &rec, seq.items, class, pos);
        const saved = ms(full -| without);
        sum += saved;
        const name = if (class == head_class) "lm head" else @tagName(@as(Kernel, @fromBackingInt(@intCast(class))));
        try out.print("{s:<24} {d:>10} {d:>12.3} {d:>6.1}%\n", .{ name, count, saved, 100 * saved / ms(full) });
    }
    try out.print("{s:<24} {d:>10} {d:>12.3}\n{s:<24} {s:>10} {d:>12.3}\n", .{ "step (replayed)", seq.items.len, ms(full), "sum of shares", "", sum });
    try out.flush();
}

/// Best of five replays of a decode step's launches, without those of class
/// `skip`.
fn replay(rt: *gen.Runtime, rec: *zllm.gpu.cmd.Recorder, seq: anytype, skip: ?usize, pos: u32) !u64 {
    const p = &rt.program;
    var best: u64 = std.math.maxInt(u64);
    for (0..5) |_| {
        p.step_value = .{ .m = 1, .pos0 = pos, .u = 0.5 };
        try rec.writeBuffer(rt.gpa, p.step.buffer, 0, std.mem.asBytes(&p.step_value));
        for (seq) |e| if (skip != e.class) try e.launch.record(rt.gpa, rec, .{ .rows = 1, .keys = pos + 1 });
        try rec.submit(rt.gpa, 1);
        best = @min(best, try time(rt, rec));
    }
    return best;
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
