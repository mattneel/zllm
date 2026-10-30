//! Sampling parameters, the per-step random draw, and a CPU reference for
//! the GPU sampler (kernels/sample.wgsl).
//!
//! Semantics follow vLLM's v1 sampler (vllm-project/vllm@0c81a03,
//! vllm/v1/sample/sampler.py): logit bias, then repetition, frequency and
//! presence penalties, then greedy argmax when the temperature is below
//! 1e-5, else temperature, min-p, top-k, top-p and a random draw. Two
//! deliberate differences: tokens tied at a top-k or top-p threshold are all
//! kept (vLLM splits ties by sort order), and the draw is an inverse CDF in
//! token-id order from one uniform u per step, not vLLM's exponential race,
//! so seeded streams differ from vLLM's.

const std = @import("std");
const Allocator = std.mem.Allocator;
const kernels = @import("../kernels/kernels.zig");

/// Temperatures below this sample greedily (vLLM's _SAMPLING_EPS).
pub const greedy_below = 1e-5;

/// A model's default sampling (defaults.zig).
pub const Defaults = @import("defaults.zig").Defaults;

pub const Bias = struct { token: u32, bias: f32 };

pub const Params = struct {
    temperature: f32 = 1.0,
    /// 0 disables.
    top_k: u32 = 0,
    top_p: f32 = 1.0,
    min_p: f32 = 0.0,
    repetition_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    /// Distinct tokens.
    logit_bias: []const Bias = &.{},
    seed: u64 = 0,

    pub const Error = error{InvalidSampling};

    /// vLLM's SamplingParams checks, plus the GPU sampler's list limits.
    pub fn validate(p: Params, vocab: u32) Error!void {
        const ok = p.temperature >= 0 and std.math.isFinite(p.temperature) and
            p.top_p > 0 and p.top_p <= 1 and
            p.min_p >= 0 and p.min_p <= 1 and
            p.repetition_penalty > 0 and std.math.isFinite(p.repetition_penalty) and
            p.presence_penalty >= -2 and p.presence_penalty <= 2 and
            p.frequency_penalty >= -2 and p.frequency_penalty <= 2 and
            p.logit_bias.len <= kernels.max_bias;
        if (!ok) return error.InvalidSampling;
        for (p.logit_bias, 0..) |b, i| {
            if (b.token >= vocab or !std.math.isFinite(b.bias)) return error.InvalidSampling;
            for (p.logit_bias[0..i]) |prev| if (prev.token == b.token) return error.InvalidSampling;
        }
    }

    pub fn greedy(p: Params) bool {
        return p.temperature < greedy_below;
    }

    /// The GPU parameter block for a request whose prompt has `n_prompt` tokens.
    pub fn gpu(p: Params, vocab: u32, n_prompt: u32, stops: []const u32) kernels.params.Sample {
        std.debug.assert(stops.len <= kernels.max_stops);
        var s: kernels.params.Sample = .{
            .vocab = vocab,
            .n_bias = @intCast(p.logit_bias.len),
            .n_stop = @intCast(stops.len),
            .greedy = @intFromBool(p.greedy()),
            .temperature = p.temperature,
            .top_k = if (p.top_k >= vocab) 0 else p.top_k,
            .top_p = p.top_p,
            .min_p = p.min_p,
            .repetition = p.repetition_penalty,
            .presence = p.presence_penalty,
            .frequency = p.frequency_penalty,
            .n_prompt = n_prompt,
            .stops = @splat(0),
        };
        @memcpy(s.stops[0..stops.len], stops);
        return s;
    }

    /// The bias list as the GPU reads it: (token, f32 bits) pairs, sorted
    /// by token for its binary search.
    pub fn biasWords(p: Params, out: [][2]u32) [][2]u32 {
        const words = out[0..p.logit_bias.len];
        for (p.logit_bias, words) |b, *w| w.* = .{ b.token, @bitCast(b.bias) };
        std.mem.sortUnstable([2]u32, words, {}, struct {
            fn lessThan(_: void, a: [2]u32, b: [2]u32) bool {
                return a[0] < b[0];
            }
        }.lessThan);
        return words;
    }
};

/// One uniform draw in [0, 1) per sampled token, from the request's seed.
pub const Draws = struct {
    prng: std.Random.DefaultPrng,

    pub fn init(seed: u64) Draws {
        return .{ .prng = .init(seed) };
    }

    pub fn next(d: *Draws) f32 {
        return @as(f32, @floatFromInt(d.prng.random().int(u24))) * 0x1p-24;
    }
};

/// Penalty state: which tokens the prompt holds and how often each was generated.
pub const History = struct {
    prompt: []const bool,
    counts: []const u32,
};

/// The CPU reference: the same steps as the GPU sampler, with the filters
/// and the draw in f64 over sorted values, as vLLM computes them.
pub fn reference(gpa: Allocator, logits: []const f32, p: Params, history: History, u: f32) Allocator.Error!u32 {
    const l = try process(gpa, logits, p, history);
    defer gpa.free(l);
    if (p.greedy()) return @intCast(std.mem.indexOfMax(f32, l));
    var f = try filter(gpa, l, p, 0);
    defer f.deinit(gpa);
    const at = @as(f64, u) * f.z;
    var c: f64 = 0;
    var last: u32 = 0;
    for (f.keep, f.e, 0..) |k, x, i| if (k) {
        c += x;
        last = @intCast(i);
        if (c > at) return @intCast(i);
    };
    return last;
}

/// How far the GPU's draw `token` is from the reference's at `u`, for
/// checks. The GPU accumulates in f32 with its own `exp`, so a draw near a
/// CDF boundary, or a token near a min-p or top-p cut, can land on the
/// neighbouring side. The result is the distance from u to the token's CDF
/// interval (0 inside it), as a fraction of the kept mass, minimized over
/// the min-p and top-p cuts moved by ±`slack` (relative). It is 1 when the
/// token is kept under none of them; greedy draws must match exactly.
pub fn deviation(gpa: Allocator, logits: []const f32, p: Params, history: History, u: f32, token: u32, slack: f64) Allocator.Error!f64 {
    const l = try process(gpa, logits, p, history);
    defer gpa.free(l);
    if (p.greedy()) return if (token == std.mem.indexOfMax(f32, l)) 0 else 1;
    var best: f64 = 1;
    for ([_]f64{ 0, slack, -slack }) |s| {
        var f = try filter(gpa, l, p, s);
        defer f.deinit(gpa);
        if (!f.keep[token]) continue;
        var lo: f64 = 0;
        for (f.keep[0..token], f.e[0..token]) |k, x| {
            if (k) lo += x;
        }
        const a = lo / f.z;
        const b = (lo + f.e[token]) / f.z;
        best = @min(best, if (u < a) a - u else if (u >= b) u - b else 0);
    }
    return best;
}

/// Bias and penalties, then the temperature (unless greedy): the same f32
/// operations as the GPU, so these values match it exactly.
fn process(gpa: Allocator, logits: []const f32, p: Params, history: History) Allocator.Error![]f32 {
    const l = try gpa.dupe(f32, logits);
    for (p.logit_bias) |b| l[b.token] += b.bias;
    for (l, history.prompt, history.counts) |*v, in_prompt, count| {
        if (in_prompt or count != 0) v.* = if (v.* > 0) v.* / p.repetition_penalty else v.* * p.repetition_penalty;
        v.* -= p.frequency_penalty * @as(f32, @floatFromInt(count));
        v.* -= p.presence_penalty * @as(f32, if (count != 0) 1 else 0);
    }
    if (!p.greedy()) for (l) |*v| {
        v.* /= p.temperature;
    };
    return l;
}

const Filtered = struct {
    keep: []bool,
    /// exp(l − max) for every token.
    e: []f64,
    /// The kept mass.
    z: f64,

    fn deinit(f: *Filtered, gpa: Allocator) void {
        gpa.free(f.keep);
        gpa.free(f.e);
    }
};

/// min-p, top-k and top-p over temperature-scaled logits. `slack` > 0
/// keeps more at the min-p and top-p cuts, < 0 fewer.
fn filter(gpa: Allocator, l: []const f32, p: Params, slack: f64) Allocator.Error!Filtered {
    const n = l.len;
    const top: f64 = l[std.mem.indexOfMax(f32, l)];
    const e = try gpa.alloc(f64, n);
    errdefer gpa.free(e);
    for (e, l) |*x, v| x.* = @exp(@as(f64, v) - top);
    const keep = try gpa.alloc(bool, n);
    errdefer gpa.free(keep);
    @memset(keep, true);

    if (p.min_p > 0) {
        // p_i < min_p · p_max, with p_max = 1 / Z.
        var z: f64 = 0;
        for (e) |x| z += x;
        for (keep, e) |*k, x| k.* = x / z >= p.min_p * (1 - slack) / z;
    }

    const order = try gpa.alloc(u32, n);
    defer gpa.free(order);
    const Desc = struct {
        l: []const f32,
        fn lessThan(ctx: @This(), a: u32, b: u32) bool {
            return ctx.l[a] > ctx.l[b];
        }
    };

    if (p.top_k > 0) {
        var kept: usize = 0;
        for (keep, 0..) |k, i| if (k) {
            order[kept] = @intCast(i);
            kept += 1;
        };
        if (kept > p.top_k) {
            std.mem.sortUnstable(u32, order[0..kept], Desc{ .l = l }, Desc.lessThan);
            const kth = l[order[p.top_k - 1]];
            for (keep, l) |*k, v| k.* = k.* and v >= kth;
        }
    }

    if (p.top_p < 1) {
        var kept: usize = 0;
        var z: f64 = 0;
        for (keep, 0..) |k, i| if (k) {
            order[kept] = @intCast(i);
            kept += 1;
            z += e[i];
        };
        std.mem.sortUnstable(u32, order[0..kept], Desc{ .l = l }, Desc.lessThan);
        // Keep the largest tokens while the mass ranked above them is < p.
        var above: f64 = 0;
        var last: usize = 0;
        for (order[0..kept], 0..) |i, rank| {
            if (above / z >= p.top_p * (1 + slack)) break;
            last = rank;
            above += e[i];
        }
        const threshold = l[order[last]];
        for (keep, l) |*k, v| k.* = k.* and v >= threshold;
    }

    var z: f64 = 0;
    for (keep, e) |k, x| {
        if (k) z += x;
    }
    return .{ .keep = keep, .e = e, .z = z };
}

test "greedy takes the first maximum after bias and penalties" {
    const gpa = std.testing.allocator;
    const logits = [_]f32{ 1, 3, 3, 2.5 };
    const prompt = [_]bool{ false, true, false, false };
    const counts = [_]u32{ 0, 0, 0, 0 };
    const h: History = .{ .prompt = &prompt, .counts = &counts };
    // Token 1 is in the prompt: 3 / 1.5 = 2 < 3, so token 2 wins.
    try std.testing.expectEqual(@as(u32, 2), try reference(gpa, &logits, .{ .temperature = 0, .repetition_penalty = 1.5 }, h, 0.5));
    // A bias of +1 on token 3 makes it the maximum.
    const bias = [_]Bias{.{ .token = 3, .bias = 1 }};
    try std.testing.expectEqual(@as(u32, 3), try reference(gpa, &logits, .{ .temperature = 0, .logit_bias = &bias }, h, 0.5));
    // Without either, ties go to the lower index.
    try std.testing.expectEqual(@as(u32, 1), try reference(gpa, &logits, .{ .temperature = 0 }, h, 0.5));
}

test "top-p keeps tokens while the mass above them is below p" {
    const gpa = std.testing.allocator;
    // Probabilities 0.5, 0.25, 0.125, 0.125 (log-space logits).
    const logits = [_]f32{ @log(0.5), @log(0.25), @log(0.125), @log(0.125) };
    const none: [4]bool = @splat(false);
    const zero: [4]u32 = @splat(0);
    const h: History = .{ .prompt = &none, .counts = &zero };
    // p = 0.6: token 0 (above 0) and token 1 (above 0.5 < 0.6) stay; the
    // draw renormalizes over 0.75, so u = 0.7 lands in token 1.
    try std.testing.expectEqual(@as(u32, 1), try reference(gpa, &logits, .{ .top_p = 0.6 }, h, 0.7));
    // p = 0.5: only token 0.
    try std.testing.expectEqual(@as(u32, 0), try reference(gpa, &logits, .{ .top_p = 0.5 }, h, 0.99));
    // top-k = 3 keeps tokens 2 and 3, tied at the third value.
    try std.testing.expectEqual(@as(u32, 3), try reference(gpa, &logits, .{ .top_k = 3 }, h, 0.99));
    // min-p = 0.3 drops p < 0.15: tokens 2 and 3.
    try std.testing.expectEqual(@as(u32, 1), try reference(gpa, &logits, .{ .min_p = 0.3 }, h, 0.99));
}

test "validate rejects out-of-range parameters and duplicate biases" {
    try (Params{}).validate(10);
    try std.testing.expectError(error.InvalidSampling, (Params{ .top_p = 0 }).validate(10));
    try std.testing.expectError(error.InvalidSampling, (Params{ .temperature = -1 }).validate(10));
    const dup = [_]Bias{ .{ .token = 1, .bias = 1 }, .{ .token = 1, .bias = 2 } };
    try std.testing.expectError(error.InvalidSampling, (Params{ .logit_bias = &dup }).validate(10));
    const out_of_range = [_]Bias{.{ .token = 10, .bias = 1 }};
    try std.testing.expectError(error.InvalidSampling, (Params{ .logit_bias = &out_of_range }).validate(10));
}
