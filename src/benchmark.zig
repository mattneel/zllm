//! `zllm benchmark`: the per-device benchmark record (spec §12): load
//! phases, prefill throughput on a 1,024-token prompt and greedy decode
//! throughput over 128 tokens, through the engine as `generate` runs them,
//! as one JSON line. Decode runs twice: as the engine runs it by default
//! (speculating with the model's MTP block where calibration found it pays;
//! with the drafts accepted) and one token per step. The prompt is random token ids, so the drafts' hit
//! rate says little about real text.

const std = @import("std");
const Io = std.Io;
const zllm = @import("zllm");
const gen = @import("generate.zig");

const prompt_tokens = 1024;
const decode_tokens = 128;

pub const Record = struct {
    adapter: []const u8,
    model: []const u8,
    model_bytes: u64,
    load_ms: struct { total: u64, pipelines: u64, weights: u64, program: u64, calibration: u64 },
    prefill: struct { tokens: u32, steps: u32, ms: f64, tok_s: f64 },
    decode: Decode,
    decode_plain: Decode,

    const Decode = struct { tokens: u32, ms: f64, tok_s: f64, drafted: u32, accepted: u32 };
};

/// The prompt's prefill and decode from a zeroed state.
fn turn(rt: *gen.Runtime, prompt: []const u32, speculate: ?bool) !zllm.sched.Usage {
    rt.engine.forget();
    try rt.engine.start(.{ .prompt = prompt, .max_tokens = decode_tokens, .sampling = .{ .temperature = 0 }, .speculate = speculate });
    while (true) switch (try rt.engine.next()) {
        .submit => |words| {
            const t0 = Io.Timestamp.now(rt.io, .awake);
            try rt.execute(words);
            try rt.engine.complete(@intCast(t0.durationTo(Io.Timestamp.now(rt.io, .awake)).toNanoseconds()));
        },
        .token => {},
        .wait => unreachable,
        .done => |d| return d.usage,
    };
}

fn decode(u: zllm.sched.Usage) Record.Decode {
    const decode_ms = @as(f64, @floatFromInt(u.decode_ns)) / 1e6;
    return .{ .tokens = u.completion_tokens, .ms = decode_ms, .tok_s = @as(f64, @floatFromInt(u.completion_tokens)) / decode_ms * 1e3, .drafted = u.drafted, .accepted = u.accepted };
}

pub fn run(rt: *gen.Runtime, model: []const u8, model_bytes: u64, load_ns: u64, out: *Io.Writer) !void {
    var prng: std.Random.DefaultPrng = .init(7);
    var prompt: [prompt_tokens]u32 = undefined;
    for (&prompt) |*t| t.* = prng.random().uintLessThan(u32, rt.program.vocab());
    // The first turn after load runs slower: warm up first.
    _ = try turn(rt, &prompt, false);
    const usage = try turn(rt, &prompt, null);
    const plain = try turn(rt, &prompt, false);
    const ms = struct {
        fn f(ns: u64) u64 {
            return ns / std.time.ns_per_ms;
        }
    }.f;
    const prefilled = usage.prompt_tokens - usage.cached_tokens;
    const prefill_ms = @as(f64, @floatFromInt(usage.prefill_ns)) / 1e6;
    const ld = rt.load;
    const r: Record = .{
        .adapter = rt.adapter,
        .model = std.fs.path.basename(model),
        .model_bytes = model_bytes,
        .load_ms = .{ .total = ms(load_ns), .pipelines = ms(ld.pipelines), .weights = ms(ld.weights), .program = ms(ld.program), .calibration = ms(ld.calibration) },
        .prefill = .{ .tokens = prefilled, .steps = usage.prefill_steps, .ms = prefill_ms, .tok_s = @as(f64, @floatFromInt(prefilled)) / prefill_ms * 1e3 },
        .decode = decode(usage),
        .decode_plain = decode(plain),
    };
    var s: std.json.Stringify = .{ .writer = out };
    try s.write(r);
    try out.writeByte('\n');
    try out.flush();
}
