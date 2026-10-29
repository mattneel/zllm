//! `zllm engine-check`: prefix reuse must be invisible. Each scenario runs a
//! request through the engine after earlier requests left state behind (the
//! resident sequence, checkpoints), then again after `forget`, from a zeroed
//! state, and requires the logits at every sampled step to match, plus the
//! expected number of reused prompt tokens. They match bitwise wherever both
//! runs put each position through the same kernels; a prompt that extends a
//! turn reuses replies that went through the decode kernels, so it matches
//! within a rounding tolerance. Prompts are seeded random
//! token ids; sampling is seeded, with no stop tokens.
//!
//! Two scenarios cancel a request first, in decode and in prefill. The
//! cancelled request must finish with `abort`, and the next prompt must
//! reuse exactly the tokens its completed steps processed.
//!
//! One runs a request pipelined, as the browser does: each batch executes
//! at once but completes only when the engine asks (`.wait`), so every
//! decode step is recorded before the previous token is known. Stopped by a
//! stop token, it must sample the same tokens with bitwise identical logits
//! as an unpipelined run, keep the stop token resident (the step after it
//! processed it), and the next prompt must reuse exactly that.

const std = @import("std");
const Io = std.Io;
const zllm = @import("zllm");
const gen = @import("generate.zig");

const n_gen = 12;

const Run = struct { tokens: []u32, logits: []f32, cached: u32 };

const Check = struct {
    rt: *gen.Runtime,
    arena: std.mem.Allocator,
    out: *Io.Writer,
    ok: bool = true,

    fn run(c: *Check, prompt: []const u32) !Run {
        const vocab = c.rt.program.hp.vocab;
        const logits = try c.arena.alloc(f32, n_gen * vocab);
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(c.rt.gpa);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(c.rt.gpa);
        var discard: Io.Writer.Discarding = .init(&.{});
        const t = try c.rt.turn(.{
            .prompt = prompt,
            .max_tokens = n_gen,
            .sampling = .{ .temperature = 1.0, .top_k = 50, .seed = 7 },
            .logits = logits,
        }, &discard.writer, &text, &ids);
        if (t.usage.completion_tokens != n_gen) return error.ShortGeneration;
        return .{ .tokens = try c.arena.dupe(u32, ids.items), .logits = logits, .cached = t.usage.cached_tokens };
    }

    /// Runs `prompt` with batches completed only on `.wait`, stopping at
    /// `stop`. Returns the sampled tokens and logits, and the resident
    /// sequence length.
    fn pipelined(c: *Check, prompt: []const u32, stop: u32) !struct { run: Run, finish: zllm.sched.Finish, resident: u32 } {
        const e = &c.rt.engine;
        const vocab = c.rt.program.hp.vocab;
        const logits = try c.arena.alloc(f32, n_gen * vocab);
        var ids: std.ArrayList(u32) = .empty;
        try e.start(.{
            .prompt = prompt,
            .max_tokens = n_gen,
            .sampling = .{ .temperature = 1.0, .top_k = 50, .seed = 7 },
            .stop_tokens = &.{stop},
            .logits = logits,
        });
        // GPU times of executed batches not yet completed, oldest first.
        var times: [zllm.sched.max_in_flight]u64 = undefined;
        var outstanding: usize = 0;
        var most: usize = 0;
        const d = while (true) switch (try e.next()) {
            .submit => |words| {
                const t0 = Io.Timestamp.now(c.rt.io, .awake);
                try c.rt.execute(words);
                times[outstanding] = @intCast(t0.durationTo(Io.Timestamp.now(c.rt.io, .awake)).toNanoseconds());
                outstanding += 1;
                most = @max(most, outstanding);
            },
            .wait => {
                try e.complete(times[0]);
                std.mem.copyForwards(u64, times[0 .. outstanding - 1], times[1..outstanding]);
                outstanding -= 1;
            },
            .token => |t| try ids.append(c.arena, t.id),
            .done => |d| break d,
        };
        if (most != zllm.sched.max_in_flight) return error.NotPipelined;
        return .{
            .run = .{ .tokens = ids.items, .logits = logits, .cached = d.usage.cached_tokens },
            .finish = d.finish,
            .resident = @intCast(e.resident().len),
        };
    }

    /// Runs `prompt` and cancels it after `tokens` sampled tokens, or with
    /// null after its first prefill step. Returns the sampled tokens and
    /// the resident sequence length, or null (reported) if the request did
    /// not end as cancelled.
    fn cancelled(c: *Check, name: []const u8, prompt: []const u32, tokens: ?u32) !?struct { tokens: []u32, resident: u32 } {
        const e = &c.rt.engine;
        var ids: std.ArrayList(u32) = .empty;
        try e.start(.{ .prompt = prompt, .max_tokens = n_gen, .sampling = .{ .temperature = 1.0, .top_k = 50, .seed = 7 } });
        var steps: u32 = 0;
        const d = while (true) switch (try e.next()) {
            .submit => |words| {
                const t0 = Io.Timestamp.now(c.rt.io, .awake);
                try c.rt.execute(words);
                try e.complete(@intCast(t0.durationTo(Io.Timestamp.now(c.rt.io, .awake)).toNanoseconds()));
                steps += 1;
                if (tokens == null and steps == 1) e.cancel();
            },
            .token => |t| {
                try ids.append(c.arena, t.id);
                if (tokens) |n| if (ids.items.len == n) e.cancel();
            },
            .wait => unreachable,
            .done => |d| break d,
        };
        const want: u32 = tokens orelse 0;
        const ok = d.finish == .abort and d.usage.completion_tokens == want;
        if (!ok) {
            c.ok = false;
            try c.out.print("FAIL {s:<34} finished {t} after {d} tokens (want abort after {d})\n", .{ name, d.finish, d.usage.completion_tokens, want });
            return null;
        }
        return .{ .tokens = ids.items, .resident = @intCast(e.resident().len) };
    }

    /// Runs `prompt` on the state earlier scenarios left, then fresh. With
    /// `exact`, every sampled step's logits must be bitwise identical.
    /// Otherwise the reused state went through other kernels than a fresh
    /// prefill would (decode kernels for earlier replies), so each step's
    /// logits must agree within `cross_path_rel` (relative L2) while the
    /// sampled tokens agree.
    fn scenario(c: *Check, name: []const u8, prompt: []const u32, want_cached: u32, exact: bool) !Run {
        const reused = try c.run(prompt);
        c.rt.engine.forget();
        const fresh = try c.run(prompt);
        const vocab = c.rt.program.hp.vocab;
        var same = true;
        var worst: f64 = 0;
        for (0..n_gen) |i| {
            const a = reused.logits[i * vocab ..][0..vocab];
            const b = fresh.logits[i * vocab ..][0..vocab];
            if (exact) {
                same = same and std.mem.eql(u32, @ptrCast(a), @ptrCast(b));
            } else {
                var num: f64 = 0;
                var den: f64 = 0;
                for (a, b) |x, y| {
                    num += (@as(f64, x) - y) * (@as(f64, x) - y);
                    den += @as(f64, y) * y;
                }
                worst = @max(worst, @sqrt(num / den));
            }
            // Later steps continue from different tokens: stop comparing.
            if (reused.tokens[i] != fresh.tokens[i]) {
                same = same and !exact and i > 0;
                break;
            }
        }
        if (!exact) same = same and worst <= cross_path_rel;
        const pass = same and reused.cached == want_cached and fresh.cached == 0;
        c.ok = c.ok and pass;
        var buf: [64]u8 = undefined;
        const verdict = if (exact)
            (if (same) "bitwise identical to a fresh run" else "DIFFER from a fresh run")
        else
            std.fmt.bufPrint(&buf, "within {e:.1} of a fresh run (rel L2 {e:.1})", .{ cross_path_rel, worst }) catch unreachable;
        try c.out.print("{s} {s:<34} {d:>4} of {d:>4} prompt tokens reused (want {d}); {d} sampled steps {s}\n", .{
            if (pass) "PASS" else "FAIL", name, reused.cached, prompt.len, want_cached, n_gen, verdict,
        });
        try c.out.flush();
        return fresh;
    }
};

/// Logit agreement for state built through the decode kernels against the
/// same tokens through the prefill kernels.
const cross_path_rel = 1e-4;

pub fn run(rt: *gen.Runtime, out: *Io.Writer) !bool {
    var arena_state: std.heap.ArenaAllocator = .init(rt.gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(42);
    const vocab = rt.program.hp.vocab;
    const random = struct {
        fn f(a: std.mem.Allocator, r: std.Random, n: usize, v: u32) ![]u32 {
            const t = try a.alloc(u32, n);
            for (t) |*x| x.* = r.uintLessThan(u32, v);
            return t;
        }
    }.f;
    const p1 = try random(arena, prng.random(), 700, vocab);
    const other = try random(arena, prng.random(), 300, vocab);
    const interval = zllm.state.checkpoint_interval;

    var c: Check = .{ .rt = rt, .arena = arena, .out = out };
    // The sequence after a turn holds the prompt and every sampled token but
    // the last, which no step has processed yet.
    const first = try c.scenario("fresh prompt", p1, 0, true);
    const extended = try std.mem.concat(arena, u32, &.{ p1, first.tokens, other[0..40] });
    _ = try c.scenario("prompt extends the last turn", extended, @intCast(p1.len + n_gen - 1), false);
    const diverged = try std.mem.concat(arena, u32, &.{ p1[0..600], other[0..50] });
    _ = try c.scenario("prompt diverges after 600 tokens", diverged, interval, true);
    _ = try c.scenario("same prompt again", diverged, @intCast(diverged.len - 1), true);
    _ = try c.run(other);
    _ = try c.scenario("unrelated prompt after another", p1, 0, true);

    // Pipelined, stopped by the 6th sampled token of an unpipelined run (or
    // an earlier occurrence of it).
    {
        const name = "pipelined decode, stopped";
        rt.engine.forget();
        const plain = try c.run(p1);
        const stop = plain.tokens[5];
        const at = std.mem.indexOfScalar(u32, plain.tokens, stop).?;
        rt.engine.forget();
        const piped = try c.pipelined(p1, stop);
        const n = at + 1;
        const same = piped.finish == .stop and std.mem.eql(u32, piped.run.tokens, plain.tokens[0..n]) and
            std.mem.eql(u32, @ptrCast(piped.run.logits[0 .. n * vocab]), @ptrCast(plain.logits[0 .. n * vocab]));
        const resident_ok = piped.resident == p1.len + n;
        c.ok = c.ok and same and resident_ok;
        try out.print("{s} {s:<34} {d} tokens and logits {s} an unpipelined run; {d} resident (want {d}: the stop token too)\n", .{
            if (same and resident_ok) "PASS" else "FAIL", name, n, if (same) "bitwise identical to" else "DIFFER from", piped.resident, p1.len + n,
        });
        const next = try std.mem.concat(arena, u32, &.{ p1, plain.tokens[0..n], other[0..30] });
        _ = try c.scenario("prompt after a pipelined stop", next, @intCast(p1.len + n), false);
    }

    // Cancelled in decode after 3 tokens: the prompt and the first 2 sampled
    // tokens are resident (the third was never processed).
    if (try c.cancelled("cancel in decode", other, 3)) |r| {
        const next = try std.mem.concat(arena, u32, &.{ other, r.tokens, p1[0..30] });
        _ = try c.scenario("prompt after a cancelled decode", next, @intCast(other.len + 2), false);
    }
    // Cancelled after the first prefill step: that chunk is resident.
    if (try c.cancelled("cancel in prefill", p1, null)) |r| {
        _ = try c.scenario("prompt after a cancelled prefill", p1, r.resident, true);
    }
    return c.ok;
}
