//! zllm command-line tool. Subcommands produce machine-readable output for
//! the parity harness in tools/.

const std = @import("std");
const Io = std.Io;
const zllm = @import("zllm");

const usage =
    \\usage: zllm <command> [args]
    \\
    \\commands:
    \\  gguf-index <model.gguf>   print header, metadata and tensor index as JSON
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
    } else {
        return fail(usage);
    }
    try out.flush();
}

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
