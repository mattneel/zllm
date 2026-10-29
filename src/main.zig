//! zllm command-line tool. Subcommands produce machine-readable output for
//! the parity harness in tools/.

const std = @import("std");
const Io = std.Io;
const zllm = @import("zllm");

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
        const mapped = try mapFile(io, args[2]);
        defer std.posix.munmap(mapped);
        var file = try zllm.gguf.parse(gpa, mapped, true);
        defer file.deinit(gpa);
        try writeIndexJson(out, file);
    } else if (std.mem.eql(u8, cmd, "tokenize")) {
        if (args.len != 4) return fail(usage);
        const mapped = try mapFile(io, args[2]);
        defer std.posix.munmap(mapped);
        // The tokenizer needs only metadata, so a header-only prefix suffices.
        var file = try zllm.gguf.parse(gpa, mapped, false);
        defer file.deinit(gpa);
        var tok = try zllm.tokenizer.Tokenizer.load(gpa, file);
        defer tok.deinit(gpa);
        const texts = try Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .unlimited);
        defer gpa.free(texts);
        try tokenizeLines(gpa, &tok, texts, out);
    } else if (std.mem.eql(u8, cmd, "forward")) {
        if (args.len != 5) return fail(usage);
        try forward(gpa, io, args[2], args[3], args[4]);
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

fn forward(gpa: std.mem.Allocator, io: Io, model_path: []const u8, tokens_path: []const u8, out_path: []const u8) !void {
    const mapped = try mapFile(io, model_path);
    defer std.posix.munmap(mapped);
    var file = try zllm.gguf.parse(gpa, mapped, true);
    defer file.deinit(gpa);
    var model = try zllm.ref.qwen35.Model.load(gpa, file, mapped);
    defer model.deinit();

    const raw = try Io.Dir.cwd().readFileAlloc(io, tokens_path, gpa, .unlimited);
    defer gpa.free(raw);
    if (raw.len == 0 or raw.len % 4 != 0) return error.BadTokenFile;
    const tokens = try gpa.alloc(u32, raw.len / 4);
    defer gpa.free(tokens);
    for (tokens, 0..) |*t, i| t.* = std.mem.readInt(u32, raw[i * 4 ..][0..4], .little);

    const out_file = try Io.Dir.cwd().createFile(io, out_path, .{});
    defer out_file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw: Io.File.Writer = .init(out_file, io, &buf);
    var rec: RecordWriter = .{ .w = &fw.interface };

    const threads = std.Thread.getCpuCount() catch 1;
    try model.forward(gpa, tokens, threads, .{ .ctx = &rec, .emit = RecordWriter.emit });
    try fw.interface.flush();
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

fn fail(msg: []const u8) error{InvalidUsage} {
    std.debug.print("{s}", .{msg});
    return error.InvalidUsage;
}

fn mapFile(io: Io, path: []const u8) ![]align(std.heap.page_size_min) const u8 {
    const file = try Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const len = try file.length(io);
    return std.posix.mmap(null, @intCast(len), .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
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
