//! `zllm benchmark`: the per-device benchmark record (spec §12): load
//! phases, prefill throughput on a 1,024-token prompt and greedy decode
//! throughput over 128 tokens, through the engine as `generate` runs them,
//! as one JSON line.

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
    decode: struct { tokens: u32, ms: f64, tok_s: f64 },
};

pub fn run(rt: *gen.Runtime, model: []const u8, model_bytes: u64, load_ns: u64, out: *Io.Writer) !void {
    var prng: std.Random.DefaultPrng = .init(7);
    var prompt: [prompt_tokens]u32 = undefined;
    for (&prompt) |*t| t.* = prng.random().uintLessThan(u32, rt.program.hp.vocab);
    rt.engine.forget();
    try rt.engine.start(.{ .prompt = &prompt, .max_tokens = decode_tokens, .sampling = .{ .temperature = 0 } });
    var usage: zllm.sched.Usage = undefined;
    while (true) switch (try rt.engine.next()) {
        .submit => |words| {
            const t0 = Io.Timestamp.now(rt.io, .awake);
            try rt.execute(words);
            try rt.engine.complete(@intCast(t0.durationTo(Io.Timestamp.now(rt.io, .awake)).toNanoseconds()));
        },
        .token => {},
        .wait => unreachable,
        .done => |d| {
            usage = d.usage;
            break;
        },
    };
    const ms = struct {
        fn f(ns: u64) u64 {
            return ns / std.time.ns_per_ms;
        }
    }.f;
    const prefilled = usage.prompt_tokens - usage.cached_tokens;
    const prefill_ms = @as(f64, @floatFromInt(usage.prefill_ns)) / 1e6;
    const decode_ms = @as(f64, @floatFromInt(usage.decode_ns)) / 1e6;
    const ld = rt.load;
    const r: Record = .{
        .adapter = rt.adapter,
        .model = std.fs.path.basename(model),
        .model_bytes = model_bytes,
        .load_ms = .{ .total = ms(load_ns), .pipelines = ms(ld.pipelines), .weights = ms(ld.weights), .program = ms(ld.program), .calibration = ms(ld.calibration) },
        .prefill = .{ .tokens = prefilled, .steps = usage.prefill_steps, .ms = prefill_ms, .tok_s = @as(f64, @floatFromInt(prefilled)) / prefill_ms * 1e3 },
        .decode = .{ .tokens = usage.completion_tokens, .ms = decode_ms, .tok_s = @as(f64, @floatFromInt(usage.completion_tokens)) / decode_ms * 1e3 },
    };
    var s: std.json.Stringify = .{ .writer = out };
    try s.write(r);
    try out.writeByte('\n');
    try out.flush();
}
