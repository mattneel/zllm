//! zllm.wasm: the core for the browser, driven by the TypeScript shim
//! (web/src/zllm). The shim owns the GPU and the model file; the core sees
//! only the GGUF header and hands back command batches.
//!
//!     zllm_open(header)      parse the header (asks for a longer prefix)
//!     zllm_load(options)     build the model, the step program, the engine
//!     zllm_next()            submit | token | done | idle | wait
//!     zllm_complete(ms)      the oldest batch the shim has executed
//!     zllm_start(request)    a generation request (raw text, ids or chat)
//!
//! Loading runs through `zllm_next` too: the setup batch (buffers, weights,
//! pipelines), then the engine's two calibration steps, then idle; each is
//! completed before the next. During a request the shim may keep up to two
//! batches in flight: after `submit` it calls `zllm_next` again, and on
//! `wait` it completes its oldest batch (see sched.zig). Results
//! land in one block (`Out`, at `zllm_out()`); any negative return leaves
//! an error message at `Out.ptr`/`Out.len`. Every export runs to completion:
//! there is no blocking and no Asyncify.

const std = @import("std");
const Allocator = std.mem.Allocator;
const zllm = @import("zllm");
const plan = zllm.plan.qwen35;
const sampler = zllm.sampler;
const sched = zllm.sched;

const gpa = std.heap.wasm_allocator;

pub const std_options: std.Options = .{ .logFn = log };

pub const panic = std.debug.FullPanic(onPanic);

extern "env" fn zllm_log(ptr: [*]const u8, len: usize) void;

fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    var buf: [1024]u8 = undefined;
    const prefix = if (scope == .default) "" else "(" ++ @tagName(scope) ++ ") ";
    const text = std.fmt.bufPrint(&buf, level.asText() ++ ": " ++ prefix ++ format, args) catch &buf;
    zllm_log(text.ptr, text.len);
}

fn onPanic(msg: []const u8, _: ?usize) noreturn {
    zllm_log(msg.ptr, msg.len);
    @trap();
}

/// The result block the shim reads (web/src/zllm/abi.ts `Out`).
const Out = extern struct {
    token: u32 = 0,
    finish: u32 = 0,
    ptr: u32 = 0,
    len: u32 = 0,
    count: u32 = 0,
    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    prefill_steps: u32 = 0,
    _pad: u32 = 0,
    prefill_ms: f64 = 0,
    decode_ms: f64 = 0,
};

const Event = enum(i32) { submit = 0, token = 1, done = 2, idle = 3, wait = 4 };

var out: Out = .{};
var message: std.ArrayList(u8) = .empty;

/// The parsed header, alive until unload: the metadata slices into it.
var header: ?struct { bytes: []u8, file: zllm.gguf.File } = null;
var runtime: ?*Runtime = null;

/// Reports `err` with its context; returns the error result.
fn fail(context: []const u8, err: anyerror) i32 {
    message.clearRetainingCapacity();
    message.print(gpa, "{s}: {t}", .{ context, err }) catch {};
    setText(message.items);
    return -1;
}

fn setText(text: []const u8) void {
    out.ptr = @intFromPtr(text.ptr);
    out.len = @intCast(text.len);
}

export fn zllm_out() *Out {
    return &out;
}

export fn zllm_alloc(len: usize) ?[*]u8 {
    const mem = gpa.alloc(u8, len) catch return null;
    return mem.ptr;
}

export fn zllm_free(ptr: [*]u8, len: usize) void {
    gpa.free(ptr[0..len]);
}

/// Parses a GGUF header from a prefix of the file, taking ownership of the
/// allocation. Returns 1 when the prefix is too short.
export fn zllm_open(ptr: [*]u8, len: usize) i32 {
    const bytes = ptr[0..len];
    if (header != null) {
        gpa.free(bytes);
        return fail("open", error.AlreadyOpen);
    }
    const file = zllm.gguf.parse(gpa, bytes, false) catch |err| {
        gpa.free(bytes);
        if (err == error.Truncated) return 1;
        return fail("model header", err);
    };
    header = .{ .bytes = bytes, .file = file };
    return 0;
}

const LoadOptions = struct {
    context: u32 = 8192,
    kv: plan.KvFormat = .f16,
    max_binding: u64,
    offset_align: u32,
};

export fn zllm_load(ptr: [*]const u8, len: usize) i32 {
    if (runtime != null) return fail("load", error.AlreadyLoaded);
    const h = header orelse return fail("load", error.NotOpen);
    const opts = std.json.parseFromSlice(LoadOptions, gpa, ptr[0..len], .{ .ignore_unknown_fields = true }) catch |err| return fail("load options", err);
    defer opts.deinit();
    runtime = Runtime.create(h.file, opts.value) catch |err| return fail("load", err);
    return 0;
}

export fn zllm_next() i32 {
    const rt = runtime orelse return @backingInt(Event.idle);
    const ev = rt.next() catch |err| return fail("next", err);
    return @backingInt(ev);
}

export fn zllm_complete(elapsed_ms: f64) i32 {
    const rt = runtime orelse return fail("complete", error.NotLoaded);
    rt.complete(@intFromFloat(@max(0, elapsed_ms) * std.time.ns_per_ms)) catch |err| return fail("complete", err);
    return 0;
}

export fn zllm_start(ptr: [*]const u8, len: usize) i32 {
    const rt = runtime orelse return fail("start", error.NotLoaded);
    rt.start(ptr[0..len]) catch |err| return fail("request", err);
    return 0;
}

export fn zllm_cancel() void {
    if (runtime) |rt| rt.engine.cancel();
}

export fn zllm_tokenize(ptr: [*]const u8, len: usize) i32 {
    const rt = runtime orelse return fail("tokenize", error.NotLoaded);
    rt.ids.clearRetainingCapacity();
    rt.tok.encode(gpa, ptr[0..len], &rt.ids) catch |err| return fail("tokenize", err);
    out.ptr = @intFromPtr(rt.ids.items.ptr);
    out.count = @intCast(rt.ids.items.len);
    return 0;
}

export fn zllm_detokenize(ptr: [*]const u32, count: usize) i32 {
    const rt = runtime orelse return fail("detokenize", error.NotLoaded);
    rt.text.clearRetainingCapacity();
    rt.tok.decode(gpa, ptr[0..count], &rt.text) catch |err| return fail("detokenize", err);
    setText(rt.text.items);
    return 0;
}

export fn zllm_info() i32 {
    const rt = runtime orelse return fail("info", error.NotLoaded);
    const hp = rt.program.hp;
    const cfg = rt.program.cfg;
    rt.text.clearRetainingCapacity();
    rt.text.print(gpa, "{f}", .{std.json.fmt(.{
        .architecture = "qwen35",
        .n_layer = hp.n_layer,
        .n_embd = hp.n_embd,
        .vocab = hp.vocab,
        .context = cfg.context,
        .max_tokens = cfg.max_tokens,
        .kv = cfg.kv,
        .resident_tokens = rt.engine.resident().len,
    }, .{})}) catch |err| return fail("info", err);
    setText(rt.text.items);
    return 0;
}

export fn zllm_unload() void {
    if (runtime) |rt| rt.destroy();
    runtime = null;
    if (header) |*h| {
        h.file.deinit(gpa);
        gpa.free(h.bytes);
    }
    header = null;
}

/// Qwen 3.5 model card sampling for text tasks, as `zllm chat` uses.
const qwen_thinking: sampler.Params = .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .presence_penalty = 1.5 };
const qwen_instruct: sampler.Params = .{ .temperature = 1.0, .top_p = 1.0, .top_k = 20, .presence_penalty = 2.0 };

/// A request as the shim sends it (web/src/zllm/engine.ts `RequestJson`):
/// exactly one of `prompt` (raw text), `tokens` or `messages` (chat).
const RequestJson = struct {
    prompt: ?[]const u8 = null,
    tokens: ?[]const u32 = null,
    messages: ?[]const zllm.chat.Message = null,
    /// Functions offered to a chat (zllm.chat.Options.tools).
    tools: []const std.json.Value = &.{},
    enable_thinking: bool = true,
    reasoning_effort: ?[]const u8 = null,
    max_tokens: u32,
    stop: []const []const u8 = &.{},
    temperature: ?f32 = null,
    top_k: ?u32 = null,
    top_p: ?f32 = null,
    min_p: ?f32 = null,
    repetition_penalty: ?f32 = null,
    presence_penalty: ?f32 = null,
    frequency_penalty: ?f32 = null,
    seed: u64 = 0,
    logit_bias: []const sampler.Bias = &.{},
};

/// The model on the GPU and the engine over it. Heap-allocated: the engine
/// and the program must not move.
const Runtime = struct {
    arena_state: std.heap.ArenaAllocator,
    /// The current request's memory, alive until the next `start`.
    request_arena: std.heap.ArenaAllocator,
    tok: zllm.tokenizer.Tokenizer,
    set: zllm.kernels.Set,
    model: plan.Model,
    program: plan.Program,
    engine: sched.Engine,
    setup: zllm.gpu.cmd.Recorder = .{},
    phase: Phase = .setup,
    /// A setup or calibration batch is out for execution.
    pending: bool = false,
    /// A request is running.
    running: bool = false,
    /// <|im_end|> and the end-of-sequence token.
    stop_tokens: std.ArrayList(u32) = .empty,
    ids: std.ArrayList(u32) = .empty,
    text: std.ArrayList(u8) = .empty,

    /// Setup batch, cold calibration (warms pipelines up), warm calibration
    /// (measures a step), then ready for requests.
    const Phase = enum { setup, cold, warm, ready };

    fn create(file: zllm.gguf.File, o: LoadOptions) !*Runtime {
        const rt = try gpa.create(Runtime);
        errdefer gpa.destroy(rt);
        rt.* = .{
            .arena_state = .init(gpa),
            .request_arena = .init(gpa),
            .tok = try .load(gpa, file),
            .set = undefined,
            .model = undefined,
            .program = undefined,
            .engine = undefined,
        };
        const arena = rt.arena_state.allocator();
        const limits: plan.Limits = .{ .max_binding = o.max_binding, .offset_align = o.offset_align };
        rt.set = try .create(gpa, &rt.setup);
        rt.model = try plan.Model.upload(arena, gpa, &rt.setup, &rt.set, file, limits);
        rt.program = try plan.Program.build(arena, gpa, &rt.setup, &rt.set, &rt.model, .{
            .context = o.context,
            .max_tokens = @min(512, o.context),
            .kv = o.kv,
        }, limits);
        try rt.setup.submit(gpa, 0);
        rt.engine = try .init(gpa, &rt.program, &rt.tok);
        if (rt.tok.ids.get("<|im_end|>")) |id| try rt.stop_tokens.append(gpa, id);
        if (rt.tok.eos) |id| if (std.mem.indexOfScalar(u32, rt.stop_tokens.items, id) == null) try rt.stop_tokens.append(gpa, id);
        return rt;
    }

    fn destroy(rt: *Runtime) void {
        rt.engine.deinit();
        rt.setup.deinit(gpa);
        rt.stop_tokens.deinit(gpa);
        rt.ids.deinit(gpa);
        rt.text.deinit(gpa);
        rt.tok.deinit(gpa);
        rt.request_arena.deinit();
        rt.arena_state.deinit();
        gpa.destroy(rt);
    }

    fn submit(rt: *Runtime, words: []const u32) Event {
        if (rt.phase != .ready) rt.pending = true;
        out.ptr = @intFromPtr(words.ptr);
        out.count = @intCast(words.len);
        return .submit;
    }

    fn next(rt: *Runtime) !Event {
        if (rt.phase != .ready and rt.pending) return .wait;
        switch (rt.phase) {
            .setup => return rt.submit(rt.setup.batch()),
            .cold, .warm => return rt.submit(try rt.engine.calibration()),
            .ready => {},
        }
        if (!rt.running) return .idle;
        switch (try rt.engine.next()) {
            .submit => |words| return rt.submit(words),
            .wait => return .wait,
            .token => |t| {
                out.token = t.id;
                setText(t.text);
                return .token;
            },
            .done => |d| {
                rt.running = false;
                out.finish = @backingInt(d.finish);
                out.prompt_tokens = d.usage.prompt_tokens;
                out.cached_tokens = d.usage.cached_tokens;
                out.completion_tokens = d.usage.completion_tokens;
                out.prefill_steps = d.usage.prefill_steps;
                out.prefill_ms = ms(d.usage.prefill_ns);
                out.decode_ms = ms(d.usage.decode_ns);
                return .done;
            },
        }
    }

    fn complete(rt: *Runtime, elapsed_ns: u64) !void {
        if (rt.phase == .ready) return rt.engine.complete(elapsed_ns);
        if (!rt.pending) return error.NotSubmitted;
        rt.pending = false;
        switch (rt.phase) {
            .setup => {
                rt.setup.deinit(gpa);
                rt.setup = .{};
                rt.phase = .cold;
            },
            .cold => {
                rt.engine.calibrated(elapsed_ns, false);
                rt.phase = .warm;
            },
            .warm => {
                rt.engine.calibrated(elapsed_ns, true);
                rt.phase = .ready;
            },
            .ready => unreachable,
        }
    }

    fn start(rt: *Runtime, json: []const u8) !void {
        if (rt.phase != .ready) return error.Loading;
        if (rt.running) return error.Busy;
        _ = rt.request_arena.reset(.retain_capacity);
        const arena = rt.request_arena.allocator();
        // Numbers in tools and tool calls stay as written (zllm.chat).
        const r = try std.json.parseFromSliceLeaky(RequestJson, arena, json, .{ .ignore_unknown_fields = true, .allocate = .alloc_always, .parse_numbers = false });

        var prompt: std.ArrayList(u32) = .empty;
        var defaults: sampler.Params = .{};
        if (r.messages) |messages| {
            var text: std.ArrayList(u8) = .empty;
            try zllm.chat.render(arena, messages, .{ .enable_thinking = r.enable_thinking, .reasoning_effort = r.reasoning_effort, .tools = r.tools }, &text);
            try rt.tok.encode(arena, text.items, &prompt);
            defaults = if (r.enable_thinking) qwen_thinking else qwen_instruct;
        } else if (r.tokens) |ids| {
            try prompt.appendSlice(arena, ids);
        } else if (r.prompt) |text| {
            if (rt.tok.add_bos) if (rt.tok.bos) |b| try prompt.append(arena, b);
            try rt.tok.encode(arena, text, &prompt);
        } else return error.NoPrompt;

        var p = defaults;
        if (r.temperature) |v| p.temperature = v;
        if (r.top_k) |v| p.top_k = v;
        if (r.top_p) |v| p.top_p = v;
        if (r.min_p) |v| p.min_p = v;
        if (r.repetition_penalty) |v| p.repetition_penalty = v;
        if (r.presence_penalty) |v| p.presence_penalty = v;
        if (r.frequency_penalty) |v| p.frequency_penalty = v;
        p.seed = r.seed;
        p.logit_bias = r.logit_bias;

        try rt.engine.start(.{
            .prompt = prompt.items,
            .max_tokens = r.max_tokens,
            .sampling = p,
            .stop_tokens = rt.stop_tokens.items,
            .stop = r.stop,
        });
        rt.running = true;
    }
};

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}
