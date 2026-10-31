//! zllm command-line tool. Subcommands produce machine-readable output for
//! the parity harness in tools/.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const zllm = @import("zllm");
const gen = @import("generate.zig");
const engine_check = @import("engine_check.zig");
const bench = @import("bench.zig");
const profile = @import("profile.zig");
const soak = @import("soak.zig");
const benchmark = @import("benchmark.zig");

const usage =
    \\usage: zllm <command> [args]
    \\
    \\commands:
    \\  gguf-index <model.gguf>             print header, metadata and tensor index as JSON
    \\  tokenize <model.gguf> <texts.jsonl>  for each {"text": ...} line, print
    \\                                       {"ids": [...], "decoded": ...} as one JSON line
    \\  forward <model.gguf> <tokens.i32> <out.bin> [--llama-activations] [--mtp]
    \\                                       run the CPU f32 reference on little-endian int32
    \\                                       token ids; write activation records (format in
    \\                                       tools/llama-dump/dump.cpp). --llama-activations
    \\                                       rounds each multiply's input as llama.cpp's CPU
    \\                                       backend does (bf16, Q8_0 or Q8_K blocks). --mtp
    \\                                       also writes mtp_logits: the MTP block's draft of
    \\                                       token i + 2 at each position
    \\  dequant <type> <in.bin> <out.f32>  decode raw blocks of a GGML type (q8_0, q3_k, q4_k,
    \\                                       q5_k, q6_k, f16, bf16) to little-endian f32
    \\  chat-render <cases.jsonl> [<model.gguf> --now <unix seconds>]
    \\                                       for each {"name", "messages", "options"} line, print
    \\                                       {"name": ..., "text": ...}: the prompt in the fixed Qwen
    \\                                       chat template, or with the model's own template (options
    \\                                       as its keyword arguments; {"name", "error"} where it fails)
    \\  gpu-info                            open Dawn and print the adapter and limits as JSON
    \\  gpu-check                           check every GPU kernel against a CPU computation
    \\  bench <model.gguf> [tokens]          time decode steps after a context of `tokens` (default
    \\                                       1024), and each kernel's share of a step
    \\  profile <model.gguf> [tokens] [at]   GPU time per kernel (timestamp queries) over prefill
    \\                                       steps of `tokens` (default 32) from position `at`
    \\                                       (default 0), then decode steps
    \\  benchmark <model.gguf>              the per-device record as one JSON line: load phases,
    \\                                       prefill of 1,024 tokens, greedy decode of 128
    \\  soak <model.gguf> [minutes]          requests of every shape for `minutes` (default 10):
    \\                                       fails on a device error or a submission past 250 ms
    \\  engine-check <model.gguf>           check that reusing a resident prefix or checkpoint gives
    \\                                       the logits of a fresh run
    \\  gpu-forward <model.gguf> <tokens.i32> <out.bin> [--chunk N] [--kv f32|f16]
    \\              [--path prefill|decode] [--mtp] [--tune] [--no-fuse]
    \\                                       as `forward`, on the GPU through Dawn, in
    \\                                       steps of N tokens (default 512; f32 KV;
    \\                                       the decode kernels for steps of up to 4).
    \\                                       Qwen4Exp: decode kernels only, up to 2,048
    \\                                       tokens; writes each layer's streams and
    \\                                       experts (l_last, ffn_moe_topk) and logits
    \\
++ gen.usage ++
    \\environment:
    \\  ZLLM_THREADS   worker threads for `forward` (default: one per physical core)
    \\  ZLLM_DAWN      path to the Dawn library (default: libwebgpu_dawn.so / webgpu_dawn.dll)
    \\  ZLLM_BACKEND   GPU backend: webgpu (Dawn; default) or velocity (native D3D12; builds
    \\                 with -Dvelocity)
    \\  ZLLM_ADAPTER   Velocity's adapter index (default 0: the high-performance one)
    \\
;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    var stdout_buf: [64 * 1024]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), io, &stdout_buf);
    const out = &stdout.interface;

    if (args.len < 2) return fail(usage);
    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "gguf-index")) {
        if (args.len != 3) return fail(usage);
        var mapped = try FileBytes.open(io, args[2]);
        defer mapped.close(io);
        var file = try zllm.gguf.parse(gpa, mapped.bytes, true);
        defer file.deinit(gpa);
        try writeIndexJson(out, file);
    } else if (std.mem.eql(u8, cmd, "tokenize")) {
        if (args.len != 4) return fail(usage);
        var mapped = try FileBytes.open(io, args[2]);
        defer mapped.close(io);
        // The tokenizer needs only metadata, so a header-only prefix suffices.
        var file = try zllm.gguf.parse(gpa, mapped.bytes, false);
        defer file.deinit(gpa);
        var tok = try zllm.tokenizer.Tokenizer.load(gpa, file);
        defer tok.deinit(gpa);
        const texts = try Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .unlimited);
        defer gpa.free(texts);
        try tokenizeLines(gpa, &tok, texts, out);
    } else if (std.mem.eql(u8, cmd, "forward")) {
        if (args.len < 5 or args.len > 7) return fail(usage);
        var act: zllm.ref.matmul.Activations = .f32;
        var mtp = false;
        for (args[5..]) |flag| {
            if (std.mem.eql(u8, flag, "--llama-activations")) {
                act = .llama_cpp;
            } else if (std.mem.eql(u8, flag, "--mtp")) {
                mtp = true;
            } else return fail(usage);
        }
        try forward(gpa, io, defaultThreads(io, init.environ_map), args[2], args[3], args[4], act, mtp);
    } else if (std.mem.eql(u8, cmd, "generate") or std.mem.eql(u8, cmd, "chat") or std.mem.eql(u8, cmd, "engine-check") or std.mem.eql(u8, cmd, "bench") or std.mem.eql(u8, cmd, "profile") or std.mem.eql(u8, cmd, "soak") or std.mem.eql(u8, cmd, "benchmark")) {
        const is_chat = std.mem.eql(u8, cmd, "chat");
        const is_check = std.mem.eql(u8, cmd, "engine-check");
        const is_bench = std.mem.eql(u8, cmd, "bench");
        const is_profile = std.mem.eql(u8, cmd, "profile");
        const is_soak = std.mem.eql(u8, cmd, "soak");
        const is_benchmark = std.mem.eql(u8, cmd, "benchmark");
        var fixed: usize = if (is_chat or is_check or is_benchmark or ((is_bench or is_profile or is_soak) and args.len == 3)) 3 else 4;
        // profile takes a second positional argument.
        if (is_profile and args.len > 4 and !std.mem.startsWith(u8, args[4], "--")) fixed = 5;
        if (args.len < fixed) return fail(usage);
        var opts = gen.Options.parse(arena, args[fixed..]) catch |err| switch (err) {
            error.InvalidUsage => return fail(usage),
            else => return err,
        };
        if (is_check) opts.context = 2048;
        opts.plan_cache = try planCacheDir(arena, init.environ_map);
        var mapped = try FileBytes.open(io, args[2]);
        defer mapped.close(io);
        var file = try zllm.gguf.parse(gpa, mapped.bytes, true);
        defer file.deinit(gpa);
        const t0 = Io.Timestamp.now(io, .awake);
        const rt = try gen.Runtime.create(gpa, io, try gpuOpen(init.environ_map), file, mapped.bytes, args[2], opts);
        defer rt.destroy();
        const ld = rt.load;
        std.debug.print("[{s}; loaded in {d} ms (pipelines {d}, weights {d}, tuning {d}, program {d}, calibration {d}); context {d}, {t} KV", .{
            rt.adapter, t0.durationTo(Io.Timestamp.now(io, .awake)).toMilliseconds(), ld.pipelines / std.time.ns_per_ms, ld.weights / std.time.ns_per_ms, ld.tuning / std.time.ns_per_ms, ld.program / std.time.ns_per_ms, ld.calibration / std.time.ns_per_ms, rt.program.context(), rt.program.kv(),
        });
        if (rt.plan_cached) std.debug.print("; tuning from the plan cache", .{});
        if (rt.tuned[0] > 0) std.debug.print("; {d} of {d} decode-multiply shapes tuned to another variant", .{ rt.tuned[1], rt.tuned[0] });
        if (rt.program.mtp()) std.debug.print("; {s}", .{if (rt.engine.speculates()) "speculative" else "MTP block, not speculating by default"});
        std.debug.print("]\n", .{});
        if (is_benchmark) {
            try benchmark.run(rt, args[2], mapped.bytes.len, @intCast(t0.durationTo(Io.Timestamp.now(io, .awake)).toNanoseconds()), out);
        } else if (is_soak) {
            const minutes = if (args.len > 3) std.fmt.parseInt(u32, args[3], 10) catch return fail(usage) else 10;
            if (!try soak.run(rt, minutes, out)) return error.SoakFailed;
        } else if (is_profile) {
            const chunk = if (args.len > 3) std.fmt.parseInt(u32, args[3], 10) catch return fail(usage) else 32;
            const at = if (args.len > 4) std.fmt.parseInt(u32, args[4], 10) catch return fail(usage) else 0;
            if (chunk == 0) return fail(usage);
            try profile.run(rt, chunk, at, out);
        } else if (is_bench) {
            const fill = if (args.len > 3) std.fmt.parseInt(u32, args[3], 10) catch return fail(usage) else 1024;
            if (fill == 0 or fill >= opts.context) return fail(usage);
            try bench.run(rt, fill, out);
        } else if (is_check) {
            if (!try engine_check.run(rt, out)) return error.EngineCheckFailed;
        } else if (is_chat) {
            var in_buf: [64 * 1024]u8 = undefined;
            var stdin: Io.File.Reader = .init(.stdin(), io, &in_buf);
            try gen.chatLoop(rt, opts, &stdin.interface, out);
        } else {
            try gen.generate(rt, args[3], opts, out);
        }
    } else if (std.mem.eql(u8, cmd, "dequant")) {
        if (args.len != 5) return fail(usage);
        const t = std.meta.stringToEnum(zllm.gguf.TensorType, args[2]) orelse return fail(usage);
        const layout = t.layout() orelse return fail(usage);
        const in = try Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .unlimited);
        defer gpa.free(in);
        if (in.len % layout.block_bytes != 0) return fail(usage);
        const values = try gpa.alloc(f32, in.len / layout.block_bytes * layout.block_elems);
        defer gpa.free(values);
        try zllm.ref.tensor.dequantize(t, in, values);
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = args[4], .data = std.mem.sliceAsBytes(values) });
    } else if (std.mem.eql(u8, cmd, "chat-render")) {
        if (args.len != 3 and args.len != 6) return fail(usage);
        const cases = try Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .unlimited);
        defer gpa.free(cases);
        if (args.len == 3) {
            try chatRender(gpa, cases, out);
        } else {
            if (!std.mem.eql(u8, args[4], "--now")) return fail(usage);
            const now = std.fmt.parseInt(i64, args[5], 10) catch return fail(usage);
            try chatRenderTemplate(gpa, io, cases, args[3], zllm.chat.jinja.Date.fromUnix(now), out);
        }
    } else if (std.mem.eql(u8, cmd, "gpu-info")) {
        if (args.len != 2) return fail(usage);
        try gpuInfo(gpa, init.environ_map, out);
    } else if (std.mem.eql(u8, cmd, "gpu-check")) {
        if (args.len != 2) return fail(usage);
        try gpuCheck(gpa, init.environ_map, out);
    } else if (std.mem.eql(u8, cmd, "gpu-forward")) {
        if (args.len < 5) return fail(usage);
        try gpuForward(gpa, io, init.environ_map, args[2], args[3], args[4], try ForwardOptions.parse(args[5..]));
    } else {
        return fail(usage);
    }
    try out.flush();
}

fn tokenizeLines(gpa: std.mem.Allocator, tok: *const zllm.tokenizer.Tokenizer, texts: []const u8, out: *Io.Writer) !void {
    var ids: std.ArrayList(zllm.tokenizer.Id) = .empty;
    defer ids.deinit(gpa);
    var decoded: std.ArrayList(u8) = .empty;
    defer decoded.deinit(gpa);

    var lines = std.mem.splitScalar(u8, texts, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const parsed = try std.json.parseFromSlice(struct { text: []const u8 }, gpa, line, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        ids.clearRetainingCapacity();
        decoded.clearRetainingCapacity();
        try tok.encode(gpa, parsed.value.text, &ids);
        try tok.decode(gpa, ids.items, &decoded);

        var s: std.json.Stringify = .{ .writer = out };
        try s.beginObject();
        try s.objectField("ids");
        try s.write(ids.items);
        try s.objectField("decoded");
        try s.write(decoded.items);
        try s.endObject();
        try out.writeByte('\n');
    }
}

fn chatRender(gpa: std.mem.Allocator, cases: []const u8, out: *Io.Writer) !void {
    const Case = struct {
        name: []const u8,
        messages: []const zllm.chat.Message,
        options: zllm.chat.Options = .{},
    };
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    var lines = std.mem.splitScalar(u8, cases, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        // Numbers stay as written, as the template's tojson writes them.
        const parsed = try std.json.parseFromSlice(Case, gpa, line, .{ .ignore_unknown_fields = true, .parse_numbers = false });
        defer parsed.deinit();
        text.clearRetainingCapacity();
        try zllm.chat.render(gpa, parsed.value.messages, parsed.value.options, &text);
        var s: std.json.Stringify = .{ .writer = out };
        try s.write(.{ .name = parsed.value.name, .text = text.items });
        try out.writeByte('\n');
    }
}

/// chat-render with a model's own template: each case's messages, and its
/// options as the template's keyword arguments (tools among them), at
/// time `now`. A case the template refuses prints its error instead.
fn chatRenderTemplate(gpa: std.mem.Allocator, io: Io, cases: []const u8, model_path: []const u8, now: zllm.chat.jinja.Date, out: *Io.Writer) !void {
    const bytes = try Io.Dir.cwd().readFileAlloc(io, model_path, gpa, .unlimited);
    defer gpa.free(bytes);
    var file = try zllm.gguf.parse(gpa, bytes, false);
    defer file.deinit(gpa);
    var tok = try zllm.tokenizer.Tokenizer.load(gpa, file);
    defer tok.deinit(gpa);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const tmpl = try zllm.chat.Template.fromGguf(arena, file, &tok) orelse return error.NoChatTemplate;

    var lines = std.mem.splitScalar(u8, cases, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const parsed = try std.json.parseFromSlice(std.json.Value, arena, line, .{ .parse_numbers = false });
        const case = parsed.value.object;
        var kwargs = if (case.get("options")) |o| try o.object.clone(arena) else std.json.ObjectMap.empty;
        const tools = kwargs.get("tools");
        _ = kwargs.orderedRemove("tools");
        const gen_prompt = kwargs.get("add_generation_prompt");
        _ = kwargs.orderedRemove("add_generation_prompt");
        var text: std.ArrayList(u8) = .empty;
        var raised: ?[]const u8 = null;
        const name = case.get("name").?.string;
        var s: std.json.Stringify = .{ .writer = out };
        if (tmpl.render(arena, .{
            .messages = case.get("messages").?,
            .tools = tools,
            .add_generation_prompt = if (gen_prompt) |g| g.bool else true,
            .kwargs = kwargs,
            .now = now,
        }, &text, &raised)) |_| {
            try s.write(.{ .name = name, .text = text.items });
        } else |err| switch (err) {
            error.Raised => try s.write(.{ .name = name, .@"error" = raised.? }),
            error.TemplateError, error.TemplateSyntax => try s.write(.{ .name = name, .@"error" = @errorName(err) }),
            else => return err,
        }
        try out.writeByte('\n');
    }
}

fn forward(gpa: std.mem.Allocator, io: Io, threads: usize, model_path: []const u8, tokens_path: []const u8, out_path: []const u8, act: zllm.ref.matmul.Activations, mtp: bool) !void {
    var mapped = try FileBytes.open(io, model_path);
    defer mapped.close(io);
    var file = try zllm.gguf.parse(gpa, mapped.bytes, true);
    defer file.deinit(gpa);
    const pool = try zllm.ref.Pool.init(gpa, io, threads);
    defer pool.deinit();
    const tokens = try readTokens(gpa, io, tokens_path);
    defer gpa.free(tokens);

    const out_file = try Io.Dir.cwd().createFile(io, out_path, .{});
    defer out_file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw: Io.File.Writer = .init(out_file, io, &buf);
    var rec: RecordWriter = .{ .w = &fw.interface };

    const arch = file.get("general.architecture") orelse return error.MissingMetadata;
    const obs: zllm.ref.dense.Observer = .{ .ctx = &rec, .emit = RecordWriter.emit };
    if (arch == .string and std.mem.eql(u8, arch.string, zllm.ref.qwen4exp.model.arch)) {
        try forwardQwen4exp(gpa, pool, file, mapped.bytes, tokens, &rec);
    } else if (arch == .string and zllm.graph.dense.Arch.fromName(arch.string) != null) {
        if (mtp) return error.NoMtpBlock;
        var model = try zllm.ref.dense.Model.load(gpa, pool, file, mapped.bytes);
        defer model.deinit();
        try model.forward(gpa, tokens, pool, obs, act);
    } else {
        var model = try zllm.ref.qwen35.Model.load(gpa, pool, file, mapped.bytes);
        defer model.deinit();
        try model.forward(gpa, tokens, pool, obs, act, mtp);
    }
    try fw.interface.flush();
}

/// The Qwen4Exp reference over `tokens` in one step: its graph's
/// intermediate tensors as records named `<name>-<layer>` (or `<name>`),
/// as the reference runtime's dump tool names them, then the logits of
/// every position as `result_output`.
fn forwardQwen4exp(gpa: std.mem.Allocator, pool: *zllm.ref.Pool, file: zllm.gguf.File, bytes: []const u8, tokens: []const u32, rec: *RecordWriter) !void {
    const q = zllm.ref.qwen4exp;
    var model = try q.Model.load(gpa, file, bytes);
    defer model.deinit();
    var session = try q.Session.init(gpa, &model, pool, @intCast(tokens.len));
    defer session.deinit();
    const Dump = struct {
        rec: *RecordWriter,
        err: ?anyerror = null,
        fn emit(ctx: *anyopaque, name: []const u8, layer: i32, values: []const f32, n: usize) void {
            const d: *@This() = @ptrCast(@alignCast(ctx));
            if (d.err != null) return;
            var name_buf: [64]u8 = undefined;
            const full = if (layer < 0) name else std.fmt.bufPrint(&name_buf, "{s}-{d}", .{ name, layer }) catch name;
            RecordWriter.emit(d.rec, full, values, values.len / n, n) catch |e| {
                d.err = e;
            };
        }
    };
    var dump: Dump = .{ .rec = rec };
    session.observer = .{ .ctx = &dump, .func = Dump.emit };
    const logits = try gpa.alloc(f32, tokens.len * model.h.vocab);
    defer gpa.free(logits);
    try session.forward(tokens, logits, true);
    if (dump.err) |e| return e;
    try RecordWriter.emit(rec, "result_output", logits, model.h.vocab, tokens.len);
}

/// Little-endian int32 token ids.
fn readTokens(gpa: std.mem.Allocator, io: Io, path: []const u8) ![]u32 {
    const raw = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(raw);
    if (raw.len == 0 or raw.len % 4 != 0) return error.BadTokenFile;
    const tokens = try gpa.alloc(u32, raw.len / 4);
    for (tokens, 0..) |*t, i| t.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);
    return tokens;
}

/// Activation records shared with tools/llama-dump: u32 name length, name,
/// u32 n_dims (2), i64 ne[2] = {dim, n_tokens}, then f32 data.
const RecordWriter = struct {
    w: *Io.Writer,

    fn emit(ctx: *anyopaque, name: []const u8, data: []const f32, dim: usize, n_tokens: usize) anyerror!void {
        const self: *RecordWriter = @ptrCast(@alignCast(ctx));
        try self.w.writeInt(u32, @intCast(name.len), .little);
        try self.w.writeAll(name);
        try self.w.writeInt(u32, 2, .little);
        try self.w.writeInt(i64, @intCast(dim), .little);
        try self.w.writeInt(i64, @intCast(n_tokens), .little);
        try self.w.writeAll(std.mem.sliceAsBytes(data));
    }
};

/// `ZLLM_THREADS` if set; otherwise one thread per physical core, since the
/// reference is FMA-bound and SMT siblings share the FMA units.
fn defaultThreads(io: Io, env: *const std.process.Environ.Map) usize {
    if (env.get("ZLLM_THREADS")) |s| {
        if (std.fmt.parseInt(usize, s, 10)) |n| return @max(1, n) else |_| {}
    }
    const logical = std.Thread.getCpuCount() catch 1;
    var buf: [8]u8 = undefined;
    const smt = Io.Dir.cwd().readFile(io, "/sys/devices/system/cpu/smt/active", &buf) catch return logical;
    return if (std.mem.startsWith(u8, smt, "1")) @max(1, logical / 2) else logical;
}

/// The plan cache's directory: ZLLM_PLAN_CACHE if set (empty: no cache),
/// else zllm/plans under the user's cache directory (LOCALAPPDATA on
/// Windows, XDG_CACHE_HOME or ~/.cache elsewhere).
fn planCacheDir(arena: std.mem.Allocator, env: *const std.process.Environ.Map) !?[]const u8 {
    if (env.get("ZLLM_PLAN_CACHE")) |dir| return if (dir.len == 0) null else dir;
    const sep = std.fs.path.sep_str;
    if (builtin.os.tag == .windows) {
        const base = env.get("LOCALAPPDATA") orelse return null;
        return try std.mem.concat(arena, u8, &.{ base, sep, "zllm", sep, "plans" });
    }
    if (env.get("XDG_CACHE_HOME")) |base| return try std.mem.concat(arena, u8, &.{ base, "/zllm/plans" });
    const home = env.get("HOME") orelse return null;
    return try std.mem.concat(arena, u8, &.{ home, "/.cache/zllm/plans" });
}

fn fail(msg: []const u8) error{InvalidUsage} {
    std.debug.print("{s}", .{msg});
    return error.InvalidUsage;
}

/// A whole file mapped read-only; pages are read as they are touched, so a
/// model larger than memory (the 75 GB Gyro-Coder) still opens.
const FileBytes = struct {
    file: Io.File,
    map: Io.File.MemoryMap,
    bytes: []const u8,

    fn open(io: Io, path: []const u8) !FileBytes {
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const len = try file.length(io);
        const map = try Io.File.MemoryMap.create(io, file, .{ .len = @intCast(len), .protection = .{ .read = true }, .populate = false });
        return .{ .file = file, .map = map, .bytes = map.memory };
    }

    fn close(f: *FileBytes, io: Io) void {
        f.map.destroy(io);
        f.file.close(io);
    }
};

/// The GPU the environment selects (ZLLM_BACKEND, ZLLM_DAWN, ZLLM_ADAPTER).
fn gpuOpen(env: *const std.process.Environ.Map) !zllm.gpu.Open {
    var o: zllm.gpu.Open = .{};
    if (env.get("ZLLM_BACKEND")) |b| o.backend = std.meta.stringToEnum(zllm.gpu.Backend, b) orelse {
        std.log.err("ZLLM_BACKEND={s}: not a backend (webgpu, velocity)", .{b});
        return error.InvalidUsage;
    };
    if (env.get("ZLLM_DAWN")) |lib| o.dawn_lib = lib;
    if (env.get("ZLLM_ADAPTER")) |a| o.adapter = std.fmt.parseInt(u32, a, 10) catch return error.InvalidUsage;
    return o;
}

fn openGpu(gpa: std.mem.Allocator, env: *const std.process.Environ.Map) !zllm.gpu.Device {
    const o = try gpuOpen(env);
    return zllm.gpu.Device.open(gpa, o) catch |err| {
        switch (o.backend) {
            .webgpu => std.log.err("cannot open Dawn ({s}): {t}", .{ o.dawn_lib, err }),
            .velocity => std.log.err("cannot open Velocity (adapter {d}): {t}", .{ o.adapter, err }),
        }
        return err;
    };
}

fn gpuCheck(gpa: std.mem.Allocator, env: *const std.process.Environ.Map, out: *Io.Writer) !void {
    const d = try openGpu(gpa, env);
    defer d.deinit(gpa);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const info = try d.adapter(arena_state.allocator());
    try out.print("{s} ({s}, {t})\n", .{ info.device, info.api, d.backend() });

    var host = try zllm.gpu.Host.init(gpa, d);
    defer host.deinit();
    const ok = try zllm.kernel_check.run(gpa, d, &host, out);
    try out.flush();
    if (!ok) return error.KernelCheckFailed;
}

const ForwardOptions = struct {
    /// Tokens per step; 1 runs the sequence the way decoding does.
    chunk: u32 = 512,
    kv: zllm.plan.common.KvFormat = .f32,
    /// Force the prefill or decode kernels (decode: steps of at most 4).
    path: ?zllm.plan.qwen35.Path = null,
    /// Also run the MTP block and write mtp_logits.
    mtp: bool = false,
    /// Tune the decode multiplies first, as loading for generation does.
    tune: bool = false,
    /// Skip the fusion pass: the unfused reference for fused-path checks.
    no_fuse: bool = false,

    fn parse(args: []const []const u8) !ForwardOptions {
        var o: ForwardOptions = .{};
        var i: usize = 0;
        while (i < args.len) : (i += 2) {
            if (std.mem.eql(u8, args[i], "--mtp") or std.mem.eql(u8, args[i], "--tune") or std.mem.eql(u8, args[i], "--no-fuse")) {
                switch (args[i][2]) {
                    'm' => o.mtp = true,
                    't' => o.tune = true,
                    else => o.no_fuse = true,
                }
                i -= 1;
                continue;
            }
            if (i + 1 == args.len) return fail(usage);
            if (std.mem.eql(u8, args[i], "--chunk")) {
                o.chunk = std.fmt.parseInt(u32, args[i + 1], 10) catch return fail(usage);
                if (o.chunk == 0) return fail(usage);
            } else if (std.mem.eql(u8, args[i], "--kv")) {
                o.kv = std.meta.stringToEnum(zllm.plan.common.KvFormat, args[i + 1]) orelse return fail(usage);
            } else if (std.mem.eql(u8, args[i], "--path")) {
                o.path = std.meta.stringToEnum(zllm.plan.qwen35.Path, args[i + 1]) orelse return fail(usage);
            } else return fail(usage);
        }
        if (o.path == .decode and o.chunk > zllm.kernels.decode_max_rows) return fail(usage);
        return o;
    }
};

/// gpu-forward for Qwen4Exp: the decode kernels in steps of up to
/// min(chunk, decode_max_rows) tokens, the PLE rows read from the mapped
/// file. Writes the records the reference names alike:
/// each layer's streams (`l_last-<layer>`) and chosen experts
/// (`ffn_moe_topk-<layer>`), and every position's logits.
fn gpuForwardQwen4exp(gpa: std.mem.Allocator, io: Io, d: zllm.gpu.Device, host: *zllm.gpu.Host, file: zllm.gguf.File, bytes: []const u8, path: []const u8, tokens: []const u32, out_path: []const u8, opts: ForwardOptions) !void {
    const plan = zllm.plan.qwen4exp;
    if (opts.path == .prefill or opts.mtp or opts.tune) {
        std.log.err("gpu-forward runs Qwen4Exp with the decode kernels only, without --mtp or --tune", .{});
        return error.InvalidUsage;
    }
    const t_start = Io.Timestamp.now(io, .awake);
    const dl = d.limits();
    const limits: zllm.plan.common.Limits = .{ .max_binding = dl.max_binding, .offset_align = dl.offset_align };
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var rec: zllm.gpu.cmd.Recorder = .{};
    defer rec.deinit(gpa);
    var completions: std.ArrayList(zllm.gpu.cmd.Completion) = .empty;
    defer completions.deinit(gpa);

    const n: u32 = @intCast(tokens.len);
    if (n > plan.max_context) return error.ContextTooLong;
    const chunk: u32 = @min(opts.chunk, zllm.kernels.decode_max_rows, n);
    var set = try zllm.kernels.Set.create(gpa, arena, &rec, d.backend());
    try set.requireAll();
    try host.setFile(gpa, bytes, path);
    const model = try plan.Model.upload(arena, gpa, &rec, &set, file, limits);
    var program = try plan.Program.build(arena, gpa, &rec, &set, &model, .{ .context = @max(n, 1), .kv = opts.kv, .fuse = !opts.no_fuse }, limits);
    try program.recordReset(gpa, &rec);
    try rec.submit(gpa, 0);
    try runBatch(d, host, &rec, &completions);
    const t_loaded = Io.Timestamp.now(io, .awake);

    const vocab: usize = model.hp.vocab;
    const logits = try arena.alloc(f32, @as(usize, n) * vocab);
    // Every layer's streams and experts, for every position; each step
    // reads back into its rows.
    const hd: usize = model.hp.hcDim();
    const used: usize = model.hp.n_expert_used;
    const streams = try arena.alloc([]f32, model.hp.n_layer);
    const experts = try arena.alloc([]u32, model.hp.n_layer);
    for (streams, experts) |*s, *x| {
        s.* = try arena.alloc(f32, n * hd);
        x.* = try arena.alloc(u32, n * used);
    }
    const step_streams = try arena.alloc([]f32, model.hp.n_layer);
    const step_experts = try arena.alloc([]u32, model.hp.n_layer);
    var pos0: u32 = 0;
    var steps: u32 = 0;
    while (pos0 < n) : (steps += 1) {
        const m = @min(chunk, n - pos0);
        for (streams, experts, step_streams, step_experts) |s, x, *ss, *sx| {
            ss.* = s[pos0 * hd ..][0 .. m * hd];
            sx.* = x[pos0 * used ..][0 .. m * used];
        }
        const readback: plan.Program.Readback = .{ .streams = step_streams, .experts = step_experts };
        try program.recordStep(gpa, &rec, .{
            .pos0 = pos0,
            .tokens = tokens[pos0..][0..m],
            .logit_rows = m,
            .logits = logits[pos0 * vocab ..][0 .. m * vocab],
            .readback = &readback,
        });
        try rec.submit(gpa, steps + 1);
        try runBatch(d, host, &rec, &completions);
        pos0 += m;
    }
    const t_done = Io.Timestamp.now(io, .awake);
    std.debug.print("gpu-forward: {d} tokens in {d} steps of up to {d} ({t} KV); load {d} ms, forward {d} ms\n", .{
        n, steps, chunk, opts.kv, t_start.durationTo(t_loaded).toMilliseconds(), t_loaded.durationTo(t_done).toMilliseconds(),
    });

    const out_file = try Io.Dir.cwd().createFile(io, out_path, .{});
    defer out_file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw: Io.File.Writer = .init(out_file, io, &buf);
    var rw: RecordWriter = .{ .w = &fw.interface };
    var name_buf: [32]u8 = undefined;
    const ids_f = try arena.alloc(f32, n * used);
    for (streams, experts, 0..) |s, x, il| {
        try RecordWriter.emit(&rw, try std.fmt.bufPrint(&name_buf, "l_last-{d}", .{il}), s, hd, n);
        for (ids_f, x) |*f, id| f.* = @floatFromInt(id);
        try RecordWriter.emit(&rw, try std.fmt.bufPrint(&name_buf, "ffn_moe_topk-{d}", .{il}), ids_f, used, n);
    }
    try RecordWriter.emit(&rw, "result_output", logits, vocab, n);
    try fw.interface.flush();
}

/// The GPU forward pass, with the same arguments and activation records as
/// `forward`, run as steps of `chunk` tokens against the KV cache and the
/// recurrent state. Load and forward timings go to stderr.
fn gpuForward(gpa: std.mem.Allocator, io: Io, env: *const std.process.Environ.Map, model_path: []const u8, tokens_path: []const u8, out_path: []const u8, opts: ForwardOptions) !void {
    const plan = zllm.plan.qwen35;
    const plan_common = zllm.plan.common;
    const t_start = Io.Timestamp.now(io, .awake);
    var mapped = try FileBytes.open(io, model_path);
    defer mapped.close(io);
    var file = try zllm.gguf.parse(gpa, mapped.bytes, true);
    defer file.deinit(gpa);
    const tokens = try readTokens(gpa, io, tokens_path);
    defer gpa.free(tokens);

    const d = try openGpu(gpa, env);
    defer d.deinit(gpa);
    var host = try zllm.gpu.Host.init(gpa, d);
    defer host.deinit();
    const arch = file.get("general.architecture") orelse return error.MissingMetadata;
    if (arch == .string and std.mem.eql(u8, arch.string, zllm.graph.qwen4exp.arch)) {
        return gpuForwardQwen4exp(gpa, io, d, &host, file, mapped.bytes, model_path, tokens, out_path, opts);
    }
    const dl = d.limits();
    const limits: plan_common.Limits = .{ .max_binding = dl.max_binding, .offset_align = dl.offset_align };

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var rec: zllm.gpu.cmd.Recorder = .{};
    defer rec.deinit(gpa);
    var completions: std.ArrayList(zllm.gpu.cmd.Completion) = .empty;
    defer completions.deinit(gpa);

    const n: u32 = @intCast(tokens.len);
    const chunk = @min(opts.chunk, n);
    var set = try zllm.kernels.Set.create(gpa, arena, &rec, d.backend());
    try set.requireAll();
    try host.setFile(gpa, mapped.bytes, model_path);
    const model = try plan.Model.uploadAny(arena, gpa, &rec, &set, file, limits);
    // The upload runs before the program is built, which may read what it
    // read back (RoPE frequency factors).
    try rec.submit(gpa, 0);
    try runBatch(d, &host, &rec, &completions);
    var tuning: plan_common.Tuning = .{};
    if (opts.tune) {
        var sweep = try zllm.tune.Sweep.init(gpa, arena, &rec, &set, &model, .{}, @min(chunk, 2));
        while (try sweep.next()) |_| {
            const t0 = Io.Timestamp.now(io, .awake);
            try runBatch(d, &host, &rec, &completions);
            try sweep.done(@intCast(t0.durationTo(Io.Timestamp.now(io, .awake)).toNanoseconds()));
        }
        tuning = sweep.tuning();
    }
    var program = try plan.Program.build(arena, gpa, &rec, &set, &model, .{
        .max_tokens = chunk,
        .context = n,
        // An MTP program needs two rows (its verify step).
        .logit_rows = if (opts.mtp) @max(2, @min(chunk, 64)) else @min(chunk, 64),
        .kv = opts.kv,
        .checkpoints = 1,
        .mtp = opts.mtp,
        .tuning = tuning,
    }, limits);
    try program.recordReset(gpa, &rec);
    try rec.submit(gpa, 0);
    try runBatch(d, &host, &rec, &completions);
    const t_loaded = Io.Timestamp.now(io, .awake);

    const hp = model.hp;
    const e: usize = hp.n_embd;
    const layer_out = try arena.alloc([]f32, hp.n_layer);
    for (layer_out) |*o| o.* = try arena.alloc(f32, n * e);
    const result_norm = try arena.alloc(f32, n * e);
    const logits = try arena.alloc(f32, @as(usize, n) * hp.vocab);
    const step_layers = try arena.alloc([]f32, hp.n_layer);
    // Row i drafts token i + 2, so there are n - 1 rows.
    const mtp_logits: []f32 = if (opts.mtp) try arena.alloc(f32, @as(usize, n - 1) * hp.vocab) else &.{};
    var pos0: u32 = 0;
    var steps: u32 = 0;
    while (pos0 < n) : (steps += 1) {
        const m = @min(chunk, n - pos0);
        for (step_layers, layer_out) |*s, o| s.* = o[pos0 * e ..][0 .. m * e];
        try program.recordStep(gpa, &rec, .{
            .pos0 = pos0,
            .tokens = tokens[pos0..][0..m],
            .logit_rows = m,
            .path = opts.path,
            .readback = .{
                .layer_out = step_layers,
                .result_norm = result_norm[pos0 * e ..][0 .. m * e],
                .logits = logits[@as(usize, pos0) * hp.vocab ..][0 .. @as(usize, m) * hp.vocab],
            },
        });
        // The rows with a next token, over this step's final hidden states.
        const rows = @min(m, n - 1 - pos0);
        if (opts.mtp and rows > 0) try program.recordMtp(gpa, &rec, .{
            .pos0 = pos0,
            .tokens = tokens[pos0 + 1 ..][0..rows],
            .logit_rows = rows,
            .path = opts.path,
            .logits = mtp_logits[@as(usize, pos0) * hp.vocab ..][0 .. @as(usize, rows) * hp.vocab],
        });
        try rec.submit(gpa, steps + 1);
        try runBatch(d, &host, &rec, &completions);
        pos0 += m;
    }
    const t_done = Io.Timestamp.now(io, .awake);
    std.debug.print("gpu-forward: {d} tokens in {d} steps of up to {d} ({t} KV); load {d} ms, forward {d} ms\n", .{
        n,
        steps,
        chunk,
        opts.kv,
        t_start.durationTo(t_loaded).toMilliseconds(),
        t_loaded.durationTo(t_done).toMilliseconds(),
    });

    const out_file = try Io.Dir.cwd().createFile(io, out_path, .{});
    defer out_file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw: Io.File.Writer = .init(out_file, io, &buf);
    var rw: RecordWriter = .{ .w = &fw.interface };
    var name_buf: [32]u8 = undefined;
    for (layer_out, 0..) |o, il| {
        try RecordWriter.emit(&rw, std.fmt.bufPrint(&name_buf, "l_out-{d}", .{il}) catch unreachable, o, e, n);
    }
    try RecordWriter.emit(&rw, "result_norm", result_norm, e, n);
    try RecordWriter.emit(&rw, "result_output", logits, hp.vocab, n);
    if (opts.mtp) try RecordWriter.emit(&rw, "mtp_logits", mtp_logits, hp.vocab, n - 1);
    try fw.interface.flush();
}

fn runBatch(d: zllm.gpu.Device, host: *zllm.gpu.Host, rec: *zllm.gpu.cmd.Recorder, completions: *std.ArrayList(zllm.gpu.cmd.Completion)) !void {
    host.execute(rec.batch(), completions) catch |err| {
        if (err == error.GpuError or err == error.DeviceLost) std.log.err("{s}", .{d.errorMessage()});
        return err;
    };
    rec.clear();
}

fn gpuInfo(gpa: std.mem.Allocator, env: *const std.process.Environ.Map, out: *Io.Writer) !void {
    const d = try openGpu(gpa, env);
    defer d.deinit(gpa);
    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_1 } };
    switch (d) {
        .webgpu => |x| {
            var raw: zllm.gpu.dawn.c.WGPUAdapterInfo = undefined;
            const info = x.info(&raw);
            defer x.freeInfo(raw);
            try s.write(.{
                .backend = "webgpu",
                .vendor = info.vendor,
                .architecture = info.architecture,
                .device = info.device,
                .description = info.description,
                .api = info.backend,
                .adapter_type = info.adapter_type,
                .shader_f16 = x.has_f16,
                .limits = .{
                    .max_storage_buffer_binding_size = x.limits.maxStorageBufferBindingSize,
                    .max_buffer_size = x.limits.maxBufferSize,
                    .max_storage_buffers_per_shader_stage = x.limits.maxStorageBuffersPerShaderStage,
                    .max_compute_workgroup_storage_size = x.limits.maxComputeWorkgroupStorageSize,
                    .max_compute_invocations_per_workgroup = x.limits.maxComputeInvocationsPerWorkgroup,
                    .max_compute_workgroups_per_dimension = x.limits.maxComputeWorkgroupsPerDimension,
                },
            });
        },
        .velocity => {
            var arena_state: std.heap.ArenaAllocator = .init(gpa);
            defer arena_state.deinit();
            const info = try d.adapter(arena_state.allocator());
            const limits = d.limits();
            try s.write(.{
                .backend = "velocity",
                .vendor = info.vendor,
                .architecture = info.architecture,
                .device = info.device,
                .description = info.description,
                .api = info.api,
                .limits = .{ .max_binding = limits.max_binding, .offset_align = limits.offset_align },
            });
        },
    }
    try out.writeByte('\n');
}

/// JSON shape mirrored by tools/check_gguf.py. Arrays are summarized by type,
/// length and a SHA-256 of their encoded elements so vocab arrays stay small.
fn writeIndexJson(w: *Io.Writer, f: zllm.gguf.File) !void {
    var s: std.json.Stringify = .{ .writer = w, .options = .{ .whitespace = .indent_1 } };
    try s.beginObject();
    try s.objectField("version");
    try s.write(f.version);
    try s.objectField("alignment");
    try s.write(f.alignment);
    try s.objectField("data_offset");
    try s.write(f.data_offset);

    try s.objectField("metadata");
    try s.beginObject();
    for (f.metadata) |kv| {
        try s.objectField(kv.key);
        try s.beginObject();
        try s.objectField("type");
        try s.write(@tagName(kv.value));
        try s.objectField("value");
        switch (kv.value) {
            .array => |a| {
                try s.beginObject();
                try s.objectField("elem_type");
                try s.write(@tagName(a.elem_type));
                try s.objectField("len");
                try s.write(a.len);
                var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
                std.crypto.hash.sha2.Sha256.hash(a.bytes, &digest, .{});
                try s.objectField("sha256");
                try s.print("\"{x}\"", .{&digest});
                try s.endObject();
            },
            inline else => |v| try s.write(v),
        }
        try s.endObject();
    }
    try s.endObject();

    try s.objectField("tensors");
    try s.beginArray();
    for (f.tensors) |t| {
        try s.beginObject();
        try s.objectField("name");
        try s.write(t.name);
        try s.objectField("type");
        try s.write(@tagName(t.type));
        try s.objectField("shape");
        try s.write(t.shape());
        try s.objectField("offset");
        try s.write(f.data_offset + t.offset);
        try s.objectField("bytes");
        try s.write(t.byteSize());
        try s.endObject();
    }
    try s.endArray();
    try s.endObject();
    try w.writeByte('\n');
}
