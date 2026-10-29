//! zllm command-line tool. Subcommands produce machine-readable output for
//! the parity harness in tools/.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const zllm = @import("zllm");
const gen = @import("generate.zig");

const usage =
    \\usage: zllm <command> [args]
    \\
    \\commands:
    \\  gguf-index <model.gguf>             print header, metadata and tensor index as JSON
    \\  tokenize <model.gguf> <texts.jsonl>  for each {"text": ...} line, print
    \\                                       {"ids": [...], "decoded": ...} as one JSON line
    \\  forward <model.gguf> <tokens.i32> <out.bin>
    \\                                       run the CPU f32 reference on little-endian int32
    \\                                       token ids; write activation records (format in
    \\                                       tools/llama-dump/dump.cpp)
    \\  chat-render <cases.jsonl>           for each {"name", "messages", "options"} line, print
    \\                                       {"name": ..., "text": ...}: the prompt in the fixed Qwen
    \\                                       chat template
    \\  gpu-info                            open Dawn and print the adapter and limits as JSON
    \\  gpu-check                           check every GPU kernel against a CPU computation
    \\  gpu-forward <model.gguf> <tokens.i32> <out.bin> [--chunk N] [--kv f32|f16]
    \\                                       as `forward`, on the GPU through Dawn, in
    \\                                       steps of N tokens (default 512; f32 KV)
    \\
++ gen.usage ++
    \\environment:
    \\  ZLLM_THREADS   worker threads for `forward` (default: one per physical core)
    \\  ZLLM_DAWN      path to the Dawn library (default: libwebgpu_dawn.so / webgpu_dawn.dll)
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
        const mapped = try FileBytes.open(gpa, io, args[2]);
        defer mapped.close(gpa);
        var file = try zllm.gguf.parse(gpa, mapped.bytes, true);
        defer file.deinit(gpa);
        try writeIndexJson(out, file);
    } else if (std.mem.eql(u8, cmd, "tokenize")) {
        if (args.len != 4) return fail(usage);
        const mapped = try FileBytes.open(gpa, io, args[2]);
        defer mapped.close(gpa);
        // The tokenizer needs only metadata, so a header-only prefix suffices.
        var file = try zllm.gguf.parse(gpa, mapped.bytes, false);
        defer file.deinit(gpa);
        var tok = try zllm.tokenizer.Tokenizer.load(gpa, file);
        defer tok.deinit(gpa);
        const texts = try Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .unlimited);
        defer gpa.free(texts);
        try tokenizeLines(gpa, &tok, texts, out);
    } else if (std.mem.eql(u8, cmd, "forward")) {
        if (args.len != 5) return fail(usage);
        try forward(gpa, io, defaultThreads(io, init.environ_map), args[2], args[3], args[4]);
    } else if (std.mem.eql(u8, cmd, "generate") or std.mem.eql(u8, cmd, "chat")) {
        const is_chat = std.mem.eql(u8, cmd, "chat");
        const fixed: usize = if (is_chat) 3 else 4;
        if (args.len < fixed) return fail(usage);
        const opts = gen.Options.parse(arena, args[fixed..]) catch |err| switch (err) {
            error.InvalidUsage => return fail(usage),
            else => return err,
        };
        const mapped = try FileBytes.open(gpa, io, args[2]);
        defer mapped.close(gpa);
        var file = try zllm.gguf.parse(gpa, mapped.bytes, true);
        defer file.deinit(gpa);
        const d = try openDawn(gpa, init.environ_map);
        defer d.deinit(gpa);
        var host = zllm.gpu.Host.init(gpa, d);
        defer host.deinit();
        const t0 = Io.Timestamp.now(io, .awake);
        const rt = try gen.Runtime.create(gpa, io, &host, file, mapped.bytes, opts);
        defer rt.destroy();
        std.debug.print("[loaded in {d} ms; context {d}, {t} KV]\n", .{ t0.durationTo(Io.Timestamp.now(io, .awake)).toMilliseconds(), opts.context, opts.kv });
        if (is_chat) {
            var in_buf: [64 * 1024]u8 = undefined;
            var stdin: Io.File.Reader = .init(.stdin(), io, &in_buf);
            try gen.chatLoop(rt, opts, &stdin.interface, out);
        } else {
            try gen.generate(rt, args[3], opts, out);
        }
    } else if (std.mem.eql(u8, cmd, "chat-render")) {
        if (args.len != 3) return fail(usage);
        const cases = try Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .unlimited);
        defer gpa.free(cases);
        try chatRender(gpa, cases, out);
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
        const parsed = try std.json.parseFromSlice(Case, gpa, line, .{});
        defer parsed.deinit();
        text.clearRetainingCapacity();
        try zllm.chat.render(gpa, parsed.value.messages, parsed.value.options, &text);
        var s: std.json.Stringify = .{ .writer = out };
        try s.write(.{ .name = parsed.value.name, .text = text.items });
        try out.writeByte('\n');
    }
}

fn forward(gpa: std.mem.Allocator, io: Io, threads: usize, model_path: []const u8, tokens_path: []const u8, out_path: []const u8) !void {
    const mapped = try FileBytes.open(gpa, io, model_path);
    defer mapped.close(gpa);
    var file = try zllm.gguf.parse(gpa, mapped.bytes, true);
    defer file.deinit(gpa);
    const pool = try zllm.ref.Pool.init(gpa, io, threads);
    defer pool.deinit();
    var model = try zllm.ref.qwen35.Model.load(gpa, pool, file, mapped.bytes);
    defer model.deinit();

    const tokens = try readTokens(gpa, io, tokens_path);
    defer gpa.free(tokens);

    const out_file = try Io.Dir.cwd().createFile(io, out_path, .{});
    defer out_file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw: Io.File.Writer = .init(out_file, io, &buf);
    var rec: RecordWriter = .{ .w = &fw.interface };

    try model.forward(gpa, tokens, pool, .{ .ctx = &rec, .emit = RecordWriter.emit });
    try fw.interface.flush();
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

fn fail(msg: []const u8) error{InvalidUsage} {
    std.debug.print("{s}", .{msg});
    return error.InvalidUsage;
}

/// A whole file in memory: mapped read-only on POSIX, read into the heap on
/// Windows (where std has no mmap).
const FileBytes = struct {
    bytes: []align(std.heap.page_size_min) const u8,

    fn open(gpa: std.mem.Allocator, io: Io, path: []const u8) !FileBytes {
        if (builtin.os.tag == .windows) {
            const data = try Io.Dir.cwd().readFileAllocOptions(io, path, gpa, .unlimited, .fromByteUnits(std.heap.page_size_min), null);
            return .{ .bytes = data };
        }
        const file = try Io.Dir.cwd().openFile(io, path, .{});
        defer file.close(io);
        const len = try file.length(io);
        return .{ .bytes = try std.posix.mmap(null, @intCast(len), .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0) };
    }

    fn close(f: FileBytes, gpa: std.mem.Allocator) void {
        if (builtin.os.tag == .windows) gpa.free(f.bytes) else std.posix.munmap(f.bytes);
    }
};

fn openDawn(gpa: std.mem.Allocator, env: *const std.process.Environ.Map) !*zllm.gpu.dawn.Device {
    const lib = env.get("ZLLM_DAWN") orelse zllm.gpu.dawn.default_library;
    return zllm.gpu.dawn.Device.init(gpa, lib) catch |err| {
        std.log.err("cannot open Dawn ({s}): {t}", .{ lib, err });
        return err;
    };
}

fn gpuCheck(gpa: std.mem.Allocator, env: *const std.process.Environ.Map, out: *Io.Writer) !void {
    const d = try openDawn(gpa, env);
    defer d.deinit(gpa);
    var raw: zllm.gpu.dawn.c.WGPUAdapterInfo = undefined;
    const info = d.info(&raw);
    try out.print("{s} ({s})\n", .{ info.device, info.backend });
    d.freeInfo(raw);

    var host = zllm.gpu.Host.init(gpa, d);
    defer host.deinit();
    const ok = try zllm.kernel_check.run(gpa, &host, out);
    try out.flush();
    if (!ok) return error.KernelCheckFailed;
}

const ForwardOptions = struct {
    /// Tokens per step; 1 runs the sequence the way decoding does.
    chunk: u32 = 512,
    kv: zllm.plan.qwen35.KvFormat = .f32,

    fn parse(args: []const []const u8) !ForwardOptions {
        var o: ForwardOptions = .{};
        var i: usize = 0;
        while (i < args.len) : (i += 2) {
            if (i + 1 == args.len) return fail(usage);
            if (std.mem.eql(u8, args[i], "--chunk")) {
                o.chunk = std.fmt.parseInt(u32, args[i + 1], 10) catch return fail(usage);
                if (o.chunk == 0) return fail(usage);
            } else if (std.mem.eql(u8, args[i], "--kv")) {
                o.kv = std.meta.stringToEnum(zllm.plan.qwen35.KvFormat, args[i + 1]) orelse return fail(usage);
            } else return fail(usage);
        }
        return o;
    }
};

/// The GPU forward pass, with the same arguments and activation records as
/// `forward`, run as steps of `chunk` tokens against the KV cache and the
/// recurrent state. Load and forward timings go to stderr.
fn gpuForward(gpa: std.mem.Allocator, io: Io, env: *const std.process.Environ.Map, model_path: []const u8, tokens_path: []const u8, out_path: []const u8, opts: ForwardOptions) !void {
    const plan = zllm.plan.qwen35;
    const t_start = Io.Timestamp.now(io, .awake);
    const mapped = try FileBytes.open(gpa, io, model_path);
    defer mapped.close(gpa);
    var file = try zllm.gguf.parse(gpa, mapped.bytes, true);
    defer file.deinit(gpa);
    const tokens = try readTokens(gpa, io, tokens_path);
    defer gpa.free(tokens);

    const d = try openDawn(gpa, env);
    defer d.deinit(gpa);
    var host = zllm.gpu.Host.init(gpa, d);
    defer host.deinit();
    const limits: plan.Limits = .{
        .max_binding = d.limits.maxStorageBufferBindingSize,
        .offset_align = d.limits.minStorageBufferOffsetAlignment,
    };

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var rec: zllm.gpu.cmd.Recorder = .{};
    defer rec.deinit(gpa);
    var completions: std.ArrayList(zllm.gpu.cmd.Completion) = .empty;
    defer completions.deinit(gpa);

    const n: u32 = @intCast(tokens.len);
    const chunk = @min(opts.chunk, n);
    const set = try zllm.kernels.Set.create(gpa, &rec);
    const model = try plan.Model.upload(arena, gpa, &rec, file, mapped.bytes, limits);
    var program = try plan.Program.build(arena, gpa, &rec, &set, &model, .{
        .max_tokens = chunk,
        .context = n,
        .logit_rows = @min(chunk, 64),
        .kv = opts.kv,
        .checkpoints = 1,
    }, limits);
    try program.recordReset(gpa, &rec);
    try rec.submit(gpa, 0);
    try runBatch(&host, &rec, &completions);
    const t_loaded = Io.Timestamp.now(io, .awake);

    const hp = model.hp;
    const e: usize = hp.n_embd;
    const layer_out = try arena.alloc([]f32, hp.n_layer);
    for (layer_out) |*o| o.* = try arena.alloc(f32, n * e);
    const result_norm = try arena.alloc(f32, n * e);
    const logits = try arena.alloc(f32, @as(usize, n) * hp.vocab);
    const step_layers = try arena.alloc([]f32, hp.n_layer);
    var pos0: u32 = 0;
    var steps: u32 = 0;
    while (pos0 < n) : (steps += 1) {
        const m = @min(chunk, n - pos0);
        for (step_layers, layer_out) |*s, o| s.* = o[pos0 * e ..][0 .. m * e];
        try program.recordStep(gpa, &rec, .{
            .pos0 = pos0,
            .tokens = tokens[pos0..][0..m],
            .logit_rows = m,
            .readback = .{
                .layer_out = step_layers,
                .result_norm = result_norm[pos0 * e ..][0 .. m * e],
                .logits = logits[@as(usize, pos0) * hp.vocab ..][0 .. @as(usize, m) * hp.vocab],
            },
        });
        try rec.submit(gpa, steps + 1);
        try runBatch(&host, &rec, &completions);
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
    try fw.interface.flush();
}

fn runBatch(host: *zllm.gpu.Host, rec: *zllm.gpu.cmd.Recorder, completions: *std.ArrayList(zllm.gpu.cmd.Completion)) !void {
    host.execute(rec.batch(), completions) catch |err| {
        if (err == error.GpuError or err == error.DeviceLost) std.log.err("{s}", .{host.d.errorMessage()});
        return err;
    };
    rec.clear();
}

fn gpuInfo(gpa: std.mem.Allocator, env: *const std.process.Environ.Map, out: *Io.Writer) !void {
    const d = try openDawn(gpa, env);
    defer d.deinit(gpa);
    var raw: zllm.gpu.dawn.c.WGPUAdapterInfo = undefined;
    const info = d.info(&raw);
    defer d.freeInfo(raw);

    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_1 } };
    try s.write(.{
        .vendor = info.vendor,
        .architecture = info.architecture,
        .device = info.device,
        .description = info.description,
        .backend = info.backend,
        .adapter_type = info.adapter_type,
        .shader_f16 = d.has_f16,
        .limits = .{
            .max_storage_buffer_binding_size = d.limits.maxStorageBufferBindingSize,
            .max_buffer_size = d.limits.maxBufferSize,
            .max_storage_buffers_per_shader_stage = d.limits.maxStorageBuffersPerShaderStage,
            .max_compute_workgroup_storage_size = d.limits.maxComputeWorkgroupStorageSize,
            .max_compute_invocations_per_workgroup = d.limits.maxComputeInvocationsPerWorkgroup,
            .max_compute_workgroups_per_dimension = d.limits.maxComputeWorkgroupsPerDimension,
        },
    });
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
