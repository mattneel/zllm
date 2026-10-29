//! `zllm engine-check`: prefix reuse must be invisible. Each scenario runs a
//! request through the engine after earlier requests left state behind (the
//! resident sequence, checkpoints), then again after `forget`, from a zeroed
//! state, and requires bitwise identical logits at every sampled step, plus
//! the expected number of reused prompt tokens. Prompts are seeded random
//! token ids; sampling is seeded, with no stop tokens.

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

    /// Runs `prompt` on the state earlier scenarios left, then fresh.
    fn scenario(c: *Check, name: []const u8, prompt: []const u32, want_cached: u32) !Run {
        const reused = try c.run(prompt);
        c.rt.engine.forget();
        const fresh = try c.run(prompt);
        const same = std.mem.eql(u32, @ptrCast(reused.logits), @ptrCast(fresh.logits)) and
            std.mem.eql(u32, reused.tokens, fresh.tokens);
        const pass = same and reused.cached == want_cached and fresh.cached == 0;
        c.ok = c.ok and pass;
        try c.out.print("{s} {s:<34} {d:>4} of {d:>4} prompt tokens reused (want {d}); {d} sampled steps {s}\n", .{
            if (pass) "PASS" else "FAIL",                                                name,        reused.cached,
            prompt.len,                                                                  want_cached, n_gen,
            if (same) "bitwise identical to a fresh run" else "DIFFER from a fresh run",
        });
        try c.out.flush();
        return fresh;
    }
};

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
    const first = try c.scenario("fresh prompt", p1, 0);
    const extended = try std.mem.concat(arena, u32, &.{ p1, first.tokens, other[0..40] });
    _ = try c.scenario("prompt extends the last turn", extended, @intCast(p1.len + n_gen - 1));
    const diverged = try std.mem.concat(arena, u32, &.{ p1[0..600], other[0..50] });
    _ = try c.scenario("prompt diverges after 600 tokens", diverged, interval);
    _ = try c.scenario("same prompt again", diverged, @intCast(diverged.len - 1));
    _ = try c.run(other);
    _ = try c.scenario("unrelated prompt after another", p1, 0);
    return c.ok;
}
