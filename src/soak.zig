//! `zllm soak`: the spec's GPU safety check (§1, §12). For `minutes` it runs
//! requests of every shape through the engine: prompts of 1 to 3,000 random
//! tokens, about half extending the previous prompt (prefix reuse and
//! checkpoint restores), varied sampling, replies of 1 to 256 tokens, and
//! cancellations. It times every submission and fails on any device error
//! or a submission past the spec's 250 ms hard cap. The result is one JSON
//! line, for the per-device benchmark record.

const std = @import("std");
const Io = std.Io;
const zllm = @import("zllm");
const gen = @import("generate.zig");
const sched = zllm.sched;

pub const Result = struct {
    adapter: []const u8,
    minutes: u32,
    requests: u64 = 0,
    submissions: u64 = 0,
    prompt_tokens: u64 = 0,
    cached_tokens: u64 = 0,
    generated_tokens: u64 = 0,
    cancelled: u64 = 0,
    max_submission_ms: f64 = 0,
    p99_submission_ms: f64 = 0,
    over_cap: u64 = 0,
    device_errors: u64 = 0,
};

pub fn run(rt: *gen.Runtime, minutes: u32, out: *Io.Writer) !bool {
    const gpa = rt.gpa;
    const p = &rt.program;
    var prng: std.Random.DefaultPrng = .init(0x50a1);
    const random = prng.random();
    var times: std.ArrayList(u64) = .empty;
    defer times.deinit(gpa);
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);

    var r: Result = .{ .adapter = rt.adapter, .minutes = minutes };
    const t_end = Io.Timestamp.now(rt.io, .awake).addDuration(.fromSeconds(@as(i64, minutes) * 60));
    const context = p.context();
    while (Io.Timestamp.now(rt.io, .awake).durationTo(t_end).toNanoseconds() > 0) {
        // Extend the previous prompt, or start a new one.
        const max_tokens = random.intRangeAtMost(u32, 1, 256);
        if (prompt.items.len == 0 or random.boolean() or prompt.items.len + 3000 + max_tokens > context) prompt.clearRetainingCapacity();
        const add = random.intRangeAtMost(u32, 1, @min(3000, context - max_tokens - @as(u32, @intCast(prompt.items.len))));
        for (0..add) |_| try prompt.append(gpa, random.uintLessThan(u32, p.vocab()));
        const sampling: zllm.sampler.Params = if (random.boolean())
            .{ .temperature = 0 }
        else
            .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .presence_penalty = 1.5, .seed = random.int(u64) };
        const cancel_after: ?u32 = if (random.uintLessThan(u32, 8) == 0) random.intRangeAtMost(u32, 1, 6) else null;

        try rt.engine.start(.{ .prompt = prompt.items, .max_tokens = max_tokens, .sampling = sampling });
        r.requests += 1;
        var submits: u32 = 0;
        while (true) {
            const ev = rt.engine.next() catch |err| return fail(&r, out, err);
            switch (ev) {
                .submit => |words| {
                    if (cancel_after) |n| if (submits == n) rt.engine.cancel();
                    const step = rt.engine.queue[rt.engine.queued - 1];
                    const t0 = Io.Timestamp.now(rt.io, .awake);
                    rt.execute(words) catch |err| {
                        r.device_errors += 1;
                        return fail(&r, out, err);
                    };
                    const ns: u64 = @intCast(t0.durationTo(Io.Timestamp.now(rt.io, .awake)).toNanoseconds());
                    try rt.engine.complete(ns);
                    try times.append(gpa, ns);
                    submits += 1;
                    if (ns > sched.hard_cap_ns) r.over_cap += 1;
                    if (ns > sched.budget_ns * 4 / 3) try out.print("slow submission: {d:.1} ms, {s} of {d} tokens at position {d} (fixed {d:.1} ms, {d:.2} ms per token)\n", .{
                        ms(ns), if (step.prefill) "prefill" else "decode", step.tokens, rt.engine.seq.len(), ms(rt.engine.step_ns), ms(rt.engine.token_ns),
                    });
                },
                .token => r.generated_tokens += 1,
                .wait => unreachable,
                .done => |d| {
                    r.prompt_tokens += d.usage.prompt_tokens;
                    r.cached_tokens += d.usage.cached_tokens;
                    if (d.finish == .abort) r.cancelled += 1;
                    break;
                },
            }
        }
        if (r.requests % 20 == 0) try out.print("[{d} requests, {d} submissions, max {d:.1} ms]\n", .{ r.requests, times.items.len, maxMs(times.items) });
        try out.flush();
    }
    r.submissions = times.items.len;
    r.max_submission_ms = maxMs(times.items);
    std.mem.sort(u64, times.items, {}, std.sort.asc(u64));
    if (times.items.len > 0) r.p99_submission_ms = ms(times.items[times.items.len * 99 / 100]);
    var s: std.json.Stringify = .{ .writer = out };
    try s.write(r);
    try out.writeByte('\n');
    try out.flush();
    return r.over_cap == 0 and r.device_errors == 0;
}

fn fail(r: *Result, out: *Io.Writer, err: anyerror) !bool {
    try out.print("soak failed after {d} requests: {t}\n", .{ r.requests, err });
    var s: std.json.Stringify = .{ .writer = out };
    try s.write(r.*);
    try out.writeByte('\n');
    try out.flush();
    return false;
}

fn maxMs(times: []const u64) f64 {
    var m: u64 = 0;
    for (times) |t| m = @max(m, t);
    return ms(m);
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}
