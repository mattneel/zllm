//! `zllm generate` and `zllm chat`: text generation through the engine on
//! a Dawn host. The host is blocking: it executes each submission the engine
//! hands it, timing it for the engine's budget.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const zllm = @import("zllm");
const plan = zllm.plan.qwen35;
const plan_common = zllm.plan.common;
const chat = zllm.chat;
const sampler = zllm.sampler;

pub const usage =
    \\  generate <model.gguf> <prompt> [options]
    \\                                       continue the raw prompt text, streaming to stdout
    \\  chat <model.gguf> [options]          chat on stdin/stdout in the fixed Qwen chat template;
    \\                                       each turn reuses the conversation so far
    \\
    \\generation options:
    \\  --max-tokens N        tokens to generate per turn (default 4096)
    \\  --ctx N               context capacity in tokens (default 8192)
    \\  --kv f16|f32          KV cache format (default f16)
    \\  --temperature F --top-k N --top-p F --min-p F
    \\  --repetition-penalty F --presence-penalty F --frequency-penalty F
    \\                        sampling, as in vLLM; `chat` defaults to the Qwen 3.5
    \\                        model card's text settings, `generate` to plain sampling
    \\  --seed N              seed for the draws (default: from the clock)
    \\  --stop S              stop string, repeatable
    \\  --system S            (chat) system prompt
    \\  --no-think            (chat) answer without a reasoning block
    \\  --no-tune             skip the load-time tuning of the decode multiplies
    \\  --speculate, --no-speculate
    \\                        draft tokens with the model's MTP block and verify
    \\                        them two at a time, or not; the output is the same
    \\                        (default: where the draft's weights are at most an
    \\                        eighth of the model's)
    \\
    \\
;

pub const Options = struct {
    max_tokens: u32 = 4096,
    context: u32 = 8192,
    kv: plan_common.KvFormat = .f16,
    temperature: ?f32 = null,
    top_k: ?u32 = null,
    top_p: ?f32 = null,
    min_p: ?f32 = null,
    repetition: ?f32 = null,
    presence: ?f32 = null,
    frequency: ?f32 = null,
    seed: ?u64 = null,
    stops: []const []const u8 = &.{},
    system: ?[]const u8 = null,
    think: bool = true,
    speculate: ?bool = null,
    /// Tune the decode multiplies at load.
    tune: bool = true,

    pub fn parse(arena: Allocator, args: []const []const u8) error{ InvalidUsage, OutOfMemory }!Options {
        var o: Options = .{};
        var stops: std.ArrayList([]const u8) = .empty;
        var i: usize = 0;
        while (i < args.len) : (i += 1) {
            const flag = args[i];
            if (std.mem.eql(u8, flag, "--no-think")) {
                o.think = false;
                continue;
            }
            if (std.mem.eql(u8, flag, "--no-tune")) {
                o.tune = false;
                continue;
            }
            if (std.mem.eql(u8, flag, "--speculate") or std.mem.eql(u8, flag, "--no-speculate")) {
                o.speculate = std.mem.eql(u8, flag, "--speculate");
                continue;
            }
            if (i + 1 == args.len) return error.InvalidUsage;
            i += 1;
            const v = args[i];
            if (std.mem.eql(u8, flag, "--max-tokens")) {
                o.max_tokens = try int(u32, v);
            } else if (std.mem.eql(u8, flag, "--ctx")) {
                o.context = try int(u32, v);
            } else if (std.mem.eql(u8, flag, "--kv")) {
                o.kv = std.meta.stringToEnum(plan_common.KvFormat, v) orelse return error.InvalidUsage;
            } else if (std.mem.eql(u8, flag, "--temperature")) {
                o.temperature = try float(v);
            } else if (std.mem.eql(u8, flag, "--top-k")) {
                o.top_k = try int(u32, v);
            } else if (std.mem.eql(u8, flag, "--top-p")) {
                o.top_p = try float(v);
            } else if (std.mem.eql(u8, flag, "--min-p")) {
                o.min_p = try float(v);
            } else if (std.mem.eql(u8, flag, "--repetition-penalty")) {
                o.repetition = try float(v);
            } else if (std.mem.eql(u8, flag, "--presence-penalty")) {
                o.presence = try float(v);
            } else if (std.mem.eql(u8, flag, "--frequency-penalty")) {
                o.frequency = try float(v);
            } else if (std.mem.eql(u8, flag, "--seed")) {
                o.seed = try int(u64, v);
            } else if (std.mem.eql(u8, flag, "--stop")) {
                try stops.append(arena, v);
            } else if (std.mem.eql(u8, flag, "--system")) {
                o.system = v;
            } else return error.InvalidUsage;
        }
        o.stops = stops.items;
        if (o.max_tokens == 0 or o.context == 0) return error.InvalidUsage;
        return o;
    }

    /// `defaults` overridden by every option given.
    fn sampling(o: Options, defaults: sampler.Params, seed: u64) sampler.Params {
        var p = defaults;
        if (o.temperature) |v| p.temperature = v;
        if (o.top_k) |v| p.top_k = v;
        if (o.top_p) |v| p.top_p = v;
        if (o.min_p) |v| p.min_p = v;
        if (o.repetition) |v| p.repetition_penalty = v;
        if (o.presence) |v| p.presence_penalty = v;
        if (o.frequency) |v| p.frequency_penalty = v;
        p.seed = seed;
        return p;
    }

    fn int(comptime T: type, s: []const u8) error{InvalidUsage}!T {
        return std.fmt.parseInt(T, s, 10) catch error.InvalidUsage;
    }

    fn float(s: []const u8) error{InvalidUsage}!f32 {
        return std.fmt.parseFloat(f32, s) catch error.InvalidUsage;
    }
};

/// Qwen 3.5 model card, text tasks.
const qwen_thinking: sampler.Params = .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .presence_penalty = 1.5 };
const qwen_instruct: sampler.Params = .{ .temperature = 1.0, .top_p = 1.0, .top_k = 20, .presence_penalty = 2.0 };

/// The model on the GPU and an engine over it. Heap-allocated: the engine
/// and the program must not move.
pub const Runtime = struct {
    gpa: Allocator,
    io: Io,
    host: *zllm.gpu.Host,
    arena_state: std.heap.ArenaAllocator,
    tok: zllm.tokenizer.Tokenizer,
    set: zllm.kernels.Set,
    model: plan.Model,
    program: plan.Program,
    engine: zllm.sched.Engine,
    completions: std.ArrayList(zllm.gpu.cmd.Completion) = .empty,
    /// <|im_end|> and the end-of-sequence token.
    stop_tokens: std.ArrayList(u32) = .empty,
    /// "<device> (<backend>)".
    adapter: []const u8 = "",
    /// Where loading went, in nanoseconds.
    load: Load = .{},
    /// Shapes the sweep tuned, and how many got a variant other than the
    /// default.
    tuned: [2]u32 = .{ 0, 0 },

    pub const Load = struct {
        /// Shaders and pipelines compiled.
        pipelines: u64 = 0,
        /// Weights read from the file onto the GPU (and laid out).
        weights: u64 = 0,
        /// The decode multiplies' tuning sweep.
        tuning: u64 = 0,
        /// The step program: buffers and bind groups.
        program: u64 = 0,
        /// The engine's calibration batches (the first compiles nothing but
        /// warms every pipeline up).
        calibration: u64 = 0,
    };

    pub fn create(gpa: Allocator, io: Io, host: *zllm.gpu.Host, file: zllm.gguf.File, bytes: []const u8, o: Options) !*Runtime {
        const rt = try gpa.create(Runtime);
        errdefer gpa.destroy(rt);
        rt.* = .{
            .gpa = gpa,
            .io = io,
            .host = host,
            .arena_state = .init(gpa),
            .tok = try .load(gpa, file),
            .set = undefined,
            .model = undefined,
            .program = undefined,
            .engine = undefined,
        };
        const arena = rt.arena_state.allocator();
        const d = host.d;
        var raw: zllm.gpu.dawn.c.WGPUAdapterInfo = undefined;
        const info = d.info(&raw);
        rt.adapter = try std.fmt.allocPrint(arena, "{s} ({s})", .{ info.device, info.backend });
        d.freeInfo(raw);
        const limits: plan_common.Limits = .{
            .max_binding = d.limits.maxStorageBufferBindingSize,
            .offset_align = d.limits.minStorageBufferOffsetAlignment,
        };
        var rec: zllm.gpu.cmd.Recorder = .{};
        defer rec.deinit(gpa);
        var t = Io.Timestamp.now(io, .awake);
        rt.set = try .create(gpa, arena, &rec);
        try rt.executeTimed(&rec, &t, &rt.load.pipelines);
        host.file = bytes;
        rt.model = try plan.Model.upload(arena, gpa, &rec, &rt.set, file, limits);
        try rt.executeTimed(&rec, &t, &rt.load.weights);
        const mtp = plan.canSpeculate(&rt.model, limits);
        var tuning: plan_common.Tuning = .{};
        if (o.tune) {
            // Decoding mostly runs steps of two where it speculates.
            var sweep = try zllm.tune.Sweep.init(gpa, arena, &rec, &rt.set, &rt.model, .{}, if (mtp and plan.speculationPays(&rt.model)) 2 else 1);
            while (try sweep.next()) |words| {
                const t0 = Io.Timestamp.now(io, .awake);
                try rt.execute(words);
                try sweep.done(@intCast(t0.durationTo(Io.Timestamp.now(io, .awake)).toNanoseconds()));
            }
            tuning = sweep.tuning();
            rt.tuned = .{ @intCast(sweep.shapes.len), @intCast(tuning.entries.len) };
            const now = Io.Timestamp.now(io, .awake);
            rt.load.tuning = @intCast(t.durationTo(now).toNanoseconds());
            t = now;
        }
        rt.program = try plan.Program.build(arena, gpa, &rec, &rt.set, &rt.model, .{
            .context = o.context,
            .max_tokens = @min(512, o.context),
            .kv = o.kv,
            .mtp = mtp,
            .logit_rows = 2,
            .tuning = tuning,
        }, limits);
        try rt.executeTimed(&rec, &t, &rt.load.program);
        rt.engine = try .init(gpa, &rt.program, &rt.tok);
        while (try rt.engine.calibration()) |words| {
            const t0 = Io.Timestamp.now(io, .awake);
            try rt.execute(words);
            rt.engine.calibrated(@intCast(t0.durationTo(Io.Timestamp.now(io, .awake)).toNanoseconds()));
        }
        rt.load.calibration = @intCast(t.durationTo(Io.Timestamp.now(io, .awake)).toNanoseconds());

        if (rt.tok.ids.get("<|im_end|>")) |id| try rt.stop_tokens.append(gpa, id);
        if (rt.tok.eos) |id| if (std.mem.indexOfScalar(u32, rt.stop_tokens.items, id) == null) try rt.stop_tokens.append(gpa, id);
        return rt;
    }

    pub fn destroy(rt: *Runtime) void {
        const gpa = rt.gpa;
        rt.engine.deinit();
        rt.completions.deinit(gpa);
        rt.stop_tokens.deinit(gpa);
        rt.tok.deinit(gpa);
        rt.arena_state.deinit();
        gpa.destroy(rt);
    }

    /// Submits and runs what `rec` holds, adding the time since `t` to
    /// `phase` and moving `t` on.
    fn executeTimed(rt: *Runtime, rec: *zllm.gpu.cmd.Recorder, t: *Io.Timestamp, phase: *u64) !void {
        try rec.submit(rt.gpa, 0);
        try rt.execute(rec.batch());
        rec.clear();
        const now = Io.Timestamp.now(rt.io, .awake);
        phase.* = @intCast(t.durationTo(now).toNanoseconds());
        t.* = now;
    }

    pub fn execute(rt: *Runtime, words: []const u32) !void {
        rt.completions.clearRetainingCapacity();
        rt.host.execute(words, &rt.completions) catch |err| {
            if (err == error.GpuError or err == error.DeviceLost) std.log.err("{s}", .{rt.host.d.errorMessage()});
            return err;
        };
    }

    pub const Turn = struct { finish: zllm.sched.Finish, usage: zllm.sched.Usage };

    /// Generates from `req.prompt`, streaming text to `out` and appending
    /// it to `text` (and the token ids to `ids`, if given).
    pub fn turn(rt: *Runtime, req: zllm.sched.Request, out: *Io.Writer, text: *std.ArrayList(u8), ids: ?*std.ArrayList(u32)) !Turn {
        try rt.engine.start(req);
        while (true) switch (try rt.engine.next()) {
            .submit => |words| {
                const t0 = Io.Timestamp.now(rt.io, .awake);
                try rt.execute(words);
                const ns = t0.durationTo(Io.Timestamp.now(rt.io, .awake)).toNanoseconds();
                try rt.engine.complete(@intCast(ns));
            },
            .token => |t| {
                try out.writeAll(t.text);
                try out.flush();
                try text.appendSlice(rt.gpa, t.text);
                if (ids) |list| try list.append(rt.gpa, t.id);
            },
            // Each batch completes before the next call.
            .wait => unreachable,
            .done => |d| return .{ .finish = d.finish, .usage = d.usage },
        };
    }
};

fn defaultSeed(io: Io) u64 {
    return @truncate(@as(u96, @bitCast(Io.Timestamp.now(io, .real).toNanoseconds())));
}

fn reportTurn(t: Runtime.Turn) void {
    const u = t.usage;
    const prefilled = u.prompt_tokens - u.cached_tokens;
    const ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
        }
    }.f;
    const rate = struct {
        fn f(n: u32, ns: u64) f64 {
            return if (ns == 0) 0 else @as(f64, @floatFromInt(n)) * std.time.ns_per_s / @as(f64, @floatFromInt(ns));
        }
    }.f;
    std.debug.print("\n[{t}; prompt {d} tokens ({d} reused), prefill {d} in {d} steps, {d:.0} ms ({d:.0} tok/s); {d} generated in {d:.0} ms ({d:.1} tok/s)", .{
        t.finish,            u.prompt_tokens, u.cached_tokens,                             prefilled, u.prefill_steps, ms(u.prefill_ns), rate(prefilled, u.prefill_ns),
        u.completion_tokens, ms(u.decode_ns), rate(u.completion_tokens -| 1, u.decode_ns),
    });
    if (u.drafted > 0) std.debug.print("; {d} of {d} drafts accepted", .{ u.accepted, u.drafted });
    std.debug.print("]\n", .{});
}

pub fn generate(rt: *Runtime, prompt: []const u8, o: Options, out: *Io.Writer) !void {
    const gpa = rt.gpa;
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    if (rt.tok.add_bos) if (rt.tok.bos) |b| try ids.append(gpa, b);
    try rt.tok.encode(gpa, prompt, &ids);
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    try out.writeAll(prompt);
    const t = try rt.turn(.{
        .prompt = ids.items,
        .max_tokens = o.max_tokens,
        .sampling = o.sampling(.{}, o.seed orelse defaultSeed(rt.io)),
        .stop_tokens = rt.stop_tokens.items,
        .stop = o.stops,
        .speculate = o.speculate,
    }, out, &text, null);
    try out.writeByte('\n');
    try out.flush();
    reportTurn(t);
}

pub fn chatLoop(rt: *Runtime, o: Options, in: *Io.Reader, out: *Io.Writer) !void {
    const gpa = rt.gpa;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var messages: std.ArrayList(chat.Message) = .empty;
    if (o.system) |s| try messages.append(arena, .{ .role = .system, .content = s });
    var prompt: std.ArrayList(u8) = .empty;
    defer prompt.deinit(gpa);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var reply: std.ArrayList(u8) = .empty;
    defer reply.deinit(gpa);
    const seed = o.seed orelse defaultSeed(rt.io);
    const defaults = if (o.think) qwen_thinking else qwen_instruct;

    var turn_index: u64 = 0;
    while (true) : (turn_index += 1) {
        try out.writeAll("> ");
        try out.flush();
        const line = (try in.takeDelimiter('\n')) orelse break;
        const user = chat.trim(line);
        if (user.len == 0) continue;
        try messages.append(arena, .{ .role = .user, .content = try arena.dupe(u8, user) });

        prompt.clearRetainingCapacity();
        try chat.render(gpa, messages.items, .{ .enable_thinking = o.think }, &prompt);
        ids.clearRetainingCapacity();
        try rt.tok.encode(gpa, prompt.items, &ids);
        if (ids.items.len >= o.context) return error.ConversationTooLong;

        reply.clearRetainingCapacity();
        if (o.think) try out.writeAll("<think>\n");
        const t = try rt.turn(.{
            .prompt = ids.items,
            .max_tokens = @min(o.max_tokens, o.context - @as(u32, @intCast(ids.items.len))),
            .sampling = o.sampling(defaults, seed +% turn_index),
            .stop_tokens = rt.stop_tokens.items,
            .stop = o.stops,
            .speculate = o.speculate,
        }, out, &reply, null);
        try out.writeByte('\n');
        try out.flush();
        reportTurn(t);
        try messages.append(arena, .{ .role = .assistant, .content = try arena.dupe(u8, reply.items) });
    }
}
