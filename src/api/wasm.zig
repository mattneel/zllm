//! zllm.wasm: the core for the browser, driven by the TypeScript shim
//! (web/src/zllm). The shim owns the GPU and the model file; the core sees
//! only the GGUF header and hands back command batches.
//!
//!     zllm_open(header)      parse the header (asks for a longer prefix)
//!     zllm_plan_key(options) the plan cache's key for this load
//!     zllm_load(options)     build the model, the step program, the engine
//!     zllm_plan()            after loading: the tuning to cache, if it swept
//!     zllm_next()            submit | token | done | idle | wait
//!     zllm_complete(ms)      the oldest batch the shim has executed
//!     zllm_start(request)    a generation request (raw text, ids or chat)
//!
//! Loading runs through `zllm_next` too: the setup batch (buffers, weights,
//! pipelines), the decode multiplies' tuning sweep (unless the options carry
//! the plan cache's tuning), the step program's batch, then the engine's
//! calibration batches, then idle; each is completed before the next. During a request the shim may keep up to two
//! batches in flight: after `submit` it calls `zllm_next` again, and on
//! `wait` it completes its oldest batch (see sched.zig). Results
//! land in one block (`Out`, at `zllm_out()`); any negative return leaves
//! an error message at `Out.ptr`/`Out.len`. Every export runs to completion:
//! there is no blocking and no Asyncify.

const std = @import("std");
const Allocator = std.mem.Allocator;
const zllm = @import("zllm");
const plan = zllm.plan;
const plan_common = zllm.plan.common;
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
    /// Entries at top_ptr, for a token with logprobs.
    top_count: u32 = 0,
    prefill_ms: f64 = 0,
    decode_ms: f64 = 0,
    /// The token's log-probability; NaN if the request didn't ask.
    logprob: f32 = std.math.nan(f32),
    /// The most likely tokens': top_count (id, logprob) pairs.
    top_ptr: u32 = 0,
};

/// Where a token's top logprobs go for the shim to read.
var top: [zllm.sched.max_top_logprobs]extern struct { id: u32, logprob: f32 } = undefined;

const Event = enum(i32) { submit = 0, token = 1, done = 2, idle = 3, wait = 4 };

var out: Out = .{};
var message: std.ArrayList(u8) = .empty;

/// The parsed header, alive until unload: the metadata slices into it.
var header: ?struct { bytes: []u8, file: zllm.gguf.File } = null;
var runtime: ?*Runtime = null;

/// Reports `err` with its context; returns the error result.
/// More about the error `fail` reports next (a chat template's
/// raise_exception message), alive until then.
var detail: ?[]const u8 = null;

fn fail(context: []const u8, err: anyerror) i32 {
    message.clearRetainingCapacity();
    message.print(gpa, "{s}: {t}", .{ context, err }) catch {};
    if (detail) |d| message.print(gpa, ": {s}", .{d}) catch {};
    detail = null;
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
    kv: plan_common.KvFormat = .f16,
    max_binding: u64,
    offset_align: u32,
    /// The adapter's vendor, architecture, device and description, and the
    /// browser (its user agent): what the plan cache's key covers besides
    /// the model and the build.
    adapter: []const []const u8 = &.{},
    runtime: []const u8 = "",
    /// Tune the decode multiplies at load, unless `plan` holds the tuning.
    tune: bool = true,
    /// The plan cache's file for this load's key (zllm_plan_key), if any.
    plan: ?[]const u8 = null,
};

fn planKey(h: anytype, o: LoadOptions) plan.cache.Key {
    return plan.cache.key(.{
        .adapter = o.adapter,
        .runtime = o.runtime,
        .limits = &.{ o.max_binding, o.offset_align },
        .model_header = h.bytes[0..@min(h.bytes.len, @as(usize, @intCast(h.file.data_offset)))],
    });
}

/// The plan cache's key for a load with these options, as hex.
export fn zllm_plan_key(ptr: [*]const u8, len: usize) i32 {
    const h = header orelse return fail("plan key", error.NotOpen);
    const opts = std.json.parseFromSlice(LoadOptions, gpa, ptr[0..len], .{ .ignore_unknown_fields = true }) catch |err| return fail("load options", err);
    defer opts.deinit();
    const k = planKey(h, opts.value);
    message.clearRetainingCapacity();
    message.print(gpa, "{x}", .{&k}) catch |err| return fail("plan key", err);
    setText(message.items);
    return 0;
}

/// After loading: the plan cache's file for this load, if it swept (empty
/// text if the tuning came from the cache or was skipped).
export fn zllm_plan() i32 {
    const rt = runtime orelse return fail("plan", error.NotLoaded);
    setText(rt.plan_file.items);
    return 0;
}

export fn zllm_load(ptr: [*]const u8, len: usize) i32 {
    if (runtime != null) return fail("load", error.AlreadyLoaded);
    const h = header orelse return fail("load", error.NotOpen);
    const opts = std.json.parseFromSlice(LoadOptions, gpa, ptr[0..len], .{ .ignore_unknown_fields = true }) catch |err| return fail("load options", err);
    defer opts.deinit();
    runtime = Runtime.create(h.file, opts.value, planKey(h, opts.value)) catch |err| return fail("load", err);
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
    if (runtime) |rt| if (rt.phase == .ready) rt.engine.cancel();
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
    if (rt.phase != .ready) return fail("info", error.Loading);
    const p = &rt.program;
    rt.text.clearRetainingCapacity();
    rt.text.print(gpa, "{f}", .{std.json.fmt(.{
        .architecture = p.architecture(),
        .n_layer = p.layers(),
        .n_embd = p.embedding(),
        .vocab = p.vocab(),
        .context = p.context(),
        .max_tokens = p.maxTokens(),
        .kv = p.kv(),
        .sampling = .{ .thinking = rt.defaults.thinking, .instruct = rt.defaults.instruct, .raw = rt.defaults.raw },
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
    /// Top logprobs per token (zllm.sched.Request.logprobs).
    logprobs: ?u32 = null,
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
    /// zllm.sched.Request.speculate.
    speculate: ?bool = null,
    /// More keyword arguments for a model's own chat template, as vLLM's.
    chat_template_kwargs: ?std.json.Value = null,
    /// Unix time in seconds, for a model's own template's strftime_now.
    now: ?i64 = null,
};

/// The model on the GPU and the engine over it. Heap-allocated: the engine
/// and the program must not move.
const Runtime = struct {
    arena_state: std.heap.ArenaAllocator,
    /// The current request's memory, alive until the next `start`.
    request_arena: std.heap.ArenaAllocator,
    tok: zllm.tokenizer.Tokenizer,
    /// What requests leave unset takes.
    defaults: sampler.Defaults,
    /// The model's own chat template, where zllm.chat's fixed one does not
    /// apply.
    template: ?zllm.chat.Template = null,
    set: zllm.kernels.Set,
    model: plan.Model,
    program: plan.Program,
    engine: sched.Engine,
    setup: zllm.gpu.cmd.Recorder = .{},
    phase: Phase = .setup,
    /// What the step program is built from, once tuning is settled.
    limits: plan_common.Limits,
    config: plan.Config,
    /// The tuning sweep, while it runs (`tune` phase). Its batches come from
    /// `setup` too: one recorder for every object the load creates, so ids
    /// never repeat.
    sweep: ?zllm.tune.Sweep = null,
    /// Sweep after the setup batch (no usable plan in the options).
    will_sweep: bool = false,
    plan_key: plan.cache.Key,
    /// The plan cache's file for what the sweep found (zllm_plan).
    plan_file: std.ArrayList(u8) = .empty,
    /// A setup or calibration batch is out for execution.
    pending: bool = false,
    /// A request is running.
    running: bool = false,
    /// <|im_end|> and the end-of-sequence token.
    stop_tokens: std.ArrayList(u32) = .empty,
    ids: std.ArrayList(u32) = .empty,
    text: std.ArrayList(u8) = .empty,

    /// The setup batch, the tuning sweep's batches, the step program's
    /// batch, the engine's calibration batches, then ready for requests.
    const Phase = enum { setup, tune, program, calibrate, ready };

    fn create(file: zllm.gguf.File, o: LoadOptions, key: plan.cache.Key) !*Runtime {
        const rt = try gpa.create(Runtime);
        errdefer gpa.destroy(rt);
        rt.* = .{
            .arena_state = .init(gpa),
            .request_arena = .init(gpa),
            .tok = try .load(gpa, file),
            .defaults = .fromGguf(file),
            .set = undefined,
            .model = undefined,
            .program = undefined,
            .engine = undefined,
            .limits = .{ .max_binding = o.max_binding, .offset_align = o.offset_align },
            .config = .{ .context = o.context, .kv = o.kv },
            .plan_key = key,
        };
        const arena = rt.arena_state.allocator();
        const arch = file.get("general.architecture");
        if (arch == null or arch.? != .string or !zllm.chat.supports(arch.?.string)) {
            rt.template = try zllm.chat.Template.fromGguf(arena, file, &rt.tok);
        }
        rt.set = try .create(gpa, arena, &rt.setup);
        rt.model = try plan.Model.upload(arena, gpa, &rt.setup, &rt.set, file, rt.limits);
        rt.config.mtp = rt.model.canSpeculate(rt.limits);
        try rt.setup.submit(gpa, 0);
        // The cached tuning, if it is this load's; else a sweep, if asked.
        const cached = if (o.plan) |bytes| plan.cache.parse(arena, bytes, key) else null;
        if (cached) |t| rt.config.tuning = t;
        rt.will_sweep = o.tune and cached == null;
        if (rt.tok.ids.get("<|im_end|>")) |id| try rt.stop_tokens.append(gpa, id);
        if (rt.tok.eos) |id| if (std.mem.indexOfScalar(u32, rt.stop_tokens.items, id) == null) try rt.stop_tokens.append(gpa, id);
        return rt;
    }

    fn destroy(rt: *Runtime) void {
        if (@backingInt(rt.phase) >= @backingInt(Phase.calibrate)) rt.engine.deinit();
        rt.setup.deinit(gpa);
        rt.plan_file.deinit(gpa);
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
            .tune => if (try rt.sweep.?.next()) |words| return rt.submit(words) else {
                // Done: the winners, and the plan cache's file for them.
                rt.config.tuning = rt.sweep.?.tuning();
                try plan.cache.serialize(gpa, rt.plan_key, rt.config.tuning, &rt.plan_file);
                rt.sweep = null;
                return rt.buildProgram();
            },
            .program => return rt.buildProgram(),
            .calibrate => if (try rt.engine.calibration()) |words| return rt.submit(words) else {
                rt.phase = .ready;
            },
            .ready => {},
        }
        if (!rt.running) return .idle;
        switch (try rt.engine.next()) {
            .submit => |words| return rt.submit(words),
            .wait => return .wait,
            .token => |t| {
                out.token = t.id;
                setText(t.text);
                out.logprob = std.math.nan(f32);
                out.top_count = 0;
                if (t.logprob) |lp| {
                    out.logprob = lp.logprob;
                    for (lp.top, top[0..lp.top.len]) |s, *d| d.* = .{ .id = s.id, .logprob = s.logprob };
                    out.top_count = @intCast(lp.top.len);
                    out.top_ptr = @intCast(@intFromPtr(&top));
                }
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
                rt.setup.clear();
                if (rt.will_sweep) {
                    // Decoding mostly runs steps of two where it speculates.
                    const tokens: u32 = if (rt.config.mtp and rt.model.speculationPays()) 2 else 1;
                    rt.sweep = try zllm.tune.Sweep.init(gpa, rt.arena_state.allocator(), &rt.setup, &rt.set, &rt.model, .{}, tokens);
                    rt.phase = .tune;
                } else {
                    rt.phase = .program;
                }
            },
            .tune => try rt.sweep.?.done(elapsed_ns),
            .program => {
                rt.setup.clear();
                rt.engine = try .init(gpa, &rt.program, &rt.tok);
                rt.phase = .calibrate;
            },
            .calibrate => rt.engine.calibrated(elapsed_ns),
            .ready => unreachable,
        }
    }

    /// Records the step program's batch, with the settled tuning.
    fn buildProgram(rt: *Runtime) !Event {
        rt.program = try plan.Program.build(rt.arena_state.allocator(), gpa, &rt.setup, &rt.set, &rt.model, rt.config, rt.limits);
        try rt.setup.submit(gpa, 0);
        rt.phase = .program;
        return rt.submit(rt.setup.batch());
    }

    fn start(rt: *Runtime, json: []const u8) !void {
        if (rt.phase != .ready) return error.Loading;
        if (rt.running) return error.Busy;
        _ = rt.request_arena.reset(.retain_capacity);
        const arena = rt.request_arena.allocator();
        // Numbers in tools and tool calls stay as written (zllm.chat).
        const r = try std.json.parseFromSliceLeaky(RequestJson, arena, json, .{ .ignore_unknown_fields = true, .allocate = .alloc_always, .parse_numbers = false });

        var prompt: std.ArrayList(u32) = .empty;
        var defaults = rt.defaults.raw;
        if (r.messages) |messages| {
            var text: std.ArrayList(u8) = .empty;
            if (zllm.chat.supports(rt.program.architecture())) {
                try zllm.chat.render(arena, messages, .{ .enable_thinking = r.enable_thinking, .reasoning_effort = r.reasoning_effort, .tools = r.tools }, &text);
            } else {
                const t = rt.template orelse return error.NoChatTemplate;
                // The tools as sent; the messages in the OpenAI form.
                const raw = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{ .parse_numbers = false });
                var kwargs: std.json.ObjectMap = if (r.chat_template_kwargs) |kw| switch (kw) {
                    .object => |o| try o.clone(arena),
                    else => return error.BadRequest,
                } else .empty;
                if (!kwargs.contains("enable_thinking")) try kwargs.put(arena, "enable_thinking", .{ .bool = r.enable_thinking });
                if (r.reasoning_effort) |e| if (!kwargs.contains("reasoning_effort")) try kwargs.put(arena, "reasoning_effort", .{ .string = e });
                var raised: ?[]const u8 = null;
                t.render(arena, .{
                    .messages = try zllm.chat.messagesJson(arena, messages),
                    .tools = if (r.tools.len > 0) raw.object.get("tools") else null,
                    .kwargs = kwargs,
                    .now = if (r.now) |s| .fromUnix(s) else null,
                }, &text, &raised) catch |err| {
                    detail = raised;
                    return err;
                };
            }
            try rt.tok.encode(arena, text.items, &prompt);
            defaults = rt.defaults.chat(r.enable_thinking);
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
            .logprobs = r.logprobs,
            .sampling = p,
            .stop_tokens = rt.stop_tokens.items,
            .stop = r.stop,
            .speculate = r.speculate,
        });
        rt.running = true;
    }
};

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}
