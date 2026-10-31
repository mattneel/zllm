//! The command stream: how the zllm core drives a GPU host.
//!
//! The core never calls WebGPU. It appends commands to a `Recorder`, hands
//! the words to a host (Dawn natively, the JS shim in the browser), and later
//! consumes `Completion`s. Hosts interpret one batch per call, so the browser
//! crosses the wasm/JS boundary once per batch rather than once per dispatch.
//!
//! Encoding: little-endian u32 words. Each command starts with a header word,
//! `opcode | (words << 8)`, where `words` counts the whole command including the
//! header, so a host can skip or bounds-check every command. Objects are named
//! by ids the core assigns, densely from 0 per kind. Data the host reads
//! (buffer contents, WGSL source) and readback destinations are addresses in
//! core memory, as two words; in wasm32 the high word is 0.
//!
//! Commands between two `submit`s are encoded into one fresh command buffer.
//! Consecutive dispatches share one compute pass.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Op = enum(u8) {
    /// id, size: u64, usage: u32
    create_buffer = 1,
    /// buffer, offset: u64, data: ptr, len: u64 — queued immediately (queue.writeBuffer),
    /// so it lands before any later `submit`.
    write_buffer = 2,
    /// id, kernel: u32, variant: u32, writes: u32, artifact: ptr, len: u32 —
    /// a compute pipeline for kernel `kernel` (kernels.Kernel) in variant
    /// `variant` (0: the default), whose contract (Kernel.contract) writes
    /// the bindings set in `writes` and only reads the rest, from the
    /// backend's artifact for it: WGSL with one
    /// entry point named "main" on WebGPU, HLSL on Velocity. The ids say
    /// which kernel a pipeline is (profiles, capability reports). An empty
    /// artifact marks a kernel the backend does not implement: dispatching
    /// its pipeline fails the batch with error.UnsupportedKernel.
    create_kernel = 3,
    /// id, pipeline, group, count, then count × (binding, buffer, offset: u64, size: u64)
    create_bind_group = 5,
    /// pipeline, x, y, z, count, then count bind groups for groups 0..count
    dispatch = 6,
    /// src, src_offset: u64, dst, dst_offset: u64, size: u64
    copy_buffer = 7,
    /// token, buffer, offset: u64, size: u64, dst: ptr — copies the range back
    /// into core memory; completes after the next `submit` finishes.
    read_buffer = 8,
    /// token — submits everything recorded since the previous submit.
    submit = 9,
    /// kind, id
    destroy = 10,
    /// buffer, offset: u64, size: u64 — zero-fills the range, ordered with
    /// dispatches and copies (GPUCommandEncoder.clearBuffer).
    clear_buffer = 11,
    /// buffer, offset: u64, file_offset: u64, size: u64 — copies a range of
    /// the model file into the buffer, queued like `write_buffer`. The host
    /// reads the file itself (natively a mapping; in the browser range reads
    /// of a URL or a local file), so weights never pass through core memory.
    write_file = 12,
    _,
};

/// WebGPU GPUBufferUsage bits (spec values, so hosts pass them through).
pub const Usage = packed struct(u32) {
    map_read: bool = false,
    map_write: bool = false,
    copy_src: bool = false,
    copy_dst: bool = false,
    index: bool = false,
    vertex: bool = false,
    uniform: bool = false,
    storage: bool = false,
    indirect: bool = false,
    query_resolve: bool = false,
    _: u22 = 0,
};

pub const Kind = enum(u32) { buffer, pipeline, bind_group };

pub const Buffer = enum(u32) { _ };
pub const Pipeline = enum(u32) { _ };
pub const BindGroup = enum(u32) { _ };

pub const Binding = struct {
    binding: u32,
    buffer: Buffer,
    offset: u64 = 0,
    size: u64,
};

pub const Completion = struct {
    token: u32,
    status: Status,

    pub const Status = enum(u32) { ok = 0, failed = 1, _ };
};

/// Appends commands. Ids come from per-kind counters, so a recorder's ids
/// are only meaningful to the host executing its batches.
pub const Recorder = struct {
    words: std.ArrayList(u32) = .empty,
    next: [3]u32 = @splat(0),

    pub fn deinit(r: *Recorder, gpa: Allocator) void {
        r.words.deinit(gpa);
        r.* = undefined;
    }

    /// The commands recorded since the last `clear`.
    pub fn batch(r: *const Recorder) []const u32 {
        return r.words.items;
    }

    /// Drops the recorded commands once a host has executed them. Ids keep
    /// counting, since the host still holds the objects.
    pub fn clear(r: *Recorder) void {
        r.words.clearRetainingCapacity();
    }

    /// How many objects of `kind` the recorder has created: ids below it
    /// are taken, so a span of ids marks what a stretch of recording made.
    pub fn created(r: *const Recorder, kind: Kind) u32 {
        return r.next[@backingInt(kind)];
    }

    /// Records the destruction of every object of `kind` with an id in
    /// [from, created(kind)).
    pub fn destroySince(r: *Recorder, gpa: Allocator, kind: Kind, from: u32) Allocator.Error!void {
        for (from..r.created(kind)) |id| try r.destroy(gpa, kind, @intCast(id));
    }

    fn alloc(r: *Recorder, kind: Kind) u32 {
        const id = r.next[@backingInt(kind)];
        r.next[@backingInt(kind)] += 1;
        return id;
    }

    fn put(r: *Recorder, gpa: Allocator, op: Op, args: []const u32) Allocator.Error!void {
        try r.words.ensureUnusedCapacity(gpa, args.len + 1);
        r.words.appendAssumeCapacity(header(op, args.len + 1));
        r.words.appendSliceAssumeCapacity(args);
    }

    pub fn createBuffer(r: *Recorder, gpa: Allocator, size: u64, usage: Usage) Allocator.Error!Buffer {
        const id = r.alloc(.buffer);
        try r.put(gpa, .create_buffer, &(.{id} ++ split(size) ++ .{@as(u32, @bitCast(usage))}));
        return @fromBackingInt(@intCast(id));
    }

    /// `data` must stay valid until the batch has been executed.
    pub fn writeBuffer(r: *Recorder, gpa: Allocator, buffer: Buffer, offset: u64, data: []const u8) Allocator.Error!void {
        try r.put(gpa, .write_buffer, &(.{@backingInt(buffer)} ++ split(offset) ++ addr(data.ptr) ++ split(data.len)));
    }

    pub fn writeFile(r: *Recorder, gpa: Allocator, buffer: Buffer, offset: u64, file_offset: u64, size: u64) Allocator.Error!void {
        try r.put(gpa, .write_file, &(.{@backingInt(buffer)} ++ split(offset) ++ split(file_offset) ++ split(size)));
    }

    /// `artifact` must stay valid until the batch has been executed.
    pub fn createKernel(r: *Recorder, gpa: Allocator, kernel: u32, variant: u32, writes: u32, artifact: []const u8) Allocator.Error!Pipeline {
        const id = r.alloc(.pipeline);
        try r.put(gpa, .create_kernel, &(.{ id, kernel, variant, writes } ++ addr(artifact.ptr) ++ .{@as(u32, @intCast(artifact.len))}));
        return @fromBackingInt(@intCast(id));
    }

    pub fn createBindGroup(r: *Recorder, gpa: Allocator, pipeline: Pipeline, group: u32, entries: []const Binding) Allocator.Error!BindGroup {
        const id = r.alloc(.bind_group);
        try r.words.ensureUnusedCapacity(gpa, 5 + entries.len * 6);
        r.words.appendAssumeCapacity(header(.create_bind_group, 5 + entries.len * 6));
        r.words.appendSliceAssumeCapacity(&.{ id, @backingInt(pipeline), group, @intCast(entries.len) });
        for (entries) |e| {
            r.words.appendSliceAssumeCapacity(&(.{ e.binding, @backingInt(e.buffer) } ++ split(e.offset) ++ split(e.size)));
        }
        return @fromBackingInt(@intCast(id));
    }

    pub fn dispatch(r: *Recorder, gpa: Allocator, pipeline: Pipeline, groups: [3]u32, bind_groups: []const BindGroup) Allocator.Error!void {
        try r.words.ensureUnusedCapacity(gpa, 6 + bind_groups.len);
        r.words.appendAssumeCapacity(header(.dispatch, 6 + bind_groups.len));
        r.words.appendSliceAssumeCapacity(&.{ @backingInt(pipeline), groups[0], groups[1], groups[2], @intCast(bind_groups.len) });
        for (bind_groups) |g| r.words.appendAssumeCapacity(@backingInt(g));
    }

    pub fn copyBuffer(r: *Recorder, gpa: Allocator, src: Buffer, src_offset: u64, dst: Buffer, dst_offset: u64, size: u64) Allocator.Error!void {
        try r.put(gpa, .copy_buffer, &(.{@backingInt(src)} ++ split(src_offset) ++ .{@backingInt(dst)} ++ split(dst_offset) ++ split(size)));
    }

    /// `offset` and `size` must be multiples of 4.
    pub fn clearBuffer(r: *Recorder, gpa: Allocator, buffer: Buffer, offset: u64, size: u64) Allocator.Error!void {
        try r.put(gpa, .clear_buffer, &(.{@backingInt(buffer)} ++ split(offset) ++ split(size)));
    }

    /// `dst` must stay valid until the read's completion arrives.
    pub fn readBuffer(r: *Recorder, gpa: Allocator, token: u32, buffer: Buffer, offset: u64, dst: []u8) Allocator.Error!void {
        try r.put(gpa, .read_buffer, &(.{ token, @backingInt(buffer) } ++ split(offset) ++ split(dst.len) ++ addr(dst.ptr)));
    }

    pub fn submit(r: *Recorder, gpa: Allocator, token: u32) Allocator.Error!void {
        try r.put(gpa, .submit, &.{token});
    }

    pub fn destroy(r: *Recorder, gpa: Allocator, kind: Kind, id: u32) Allocator.Error!void {
        try r.put(gpa, .destroy, &.{ @backingInt(kind), id });
    }
};

fn split(v: u64) [2]u32 {
    return .{ @truncate(v), @truncate(v >> 32) };
}

fn addr(p: anytype) [2]u32 {
    return split(@intFromPtr(p));
}

pub fn join(lo: u32, hi: u32) u64 {
    return (@as(u64, hi) << 32) | lo;
}

fn header(op: Op, words: usize) u32 {
    return @backingInt(op) | (@as(u32, @intCast(words)) << 8);
}

/// One decoded command: its opcode and argument words (header excluded).
pub const Command = struct {
    op: Op,
    args: []const u32,

    pub fn u64At(c: Command, i: usize) u64 {
        return join(c.args[i], c.args[i + 1]);
    }

    pub fn ptrAt(c: Command, comptime T: type, i: usize) T {
        return @ptrFromInt(@as(usize, @intCast(c.u64At(i))));
    }
};

pub const DecodeError = error{ Truncated, BadLength };

/// Walks a batch command by command, checking every header.
pub const Iterator = struct {
    words: []const u32,
    pos: usize = 0,

    pub fn next(it: *Iterator) DecodeError!?Command {
        if (it.pos == it.words.len) return null;
        const word = it.words[it.pos];
        const n: usize = word >> 8;
        if (n == 0) return error.BadLength;
        if (n > it.words.len - it.pos) return error.Truncated;
        defer it.pos += n;
        return .{ .op = @fromBackingInt(@intCast(@as(u8, @truncate(word)))), .args = it.words[it.pos + 1 ..][0 .. n - 1] };
    }
};

test "recorder output decodes back to the same commands" {
    const gpa = std.testing.allocator;
    var r: Recorder = .{};
    defer r.deinit(gpa);

    const data = [_]u8{ 1, 2, 3, 4 };
    var dst: [8]u8 = undefined;
    const buf = try r.createBuffer(gpa, 1 << 33, .{ .storage = true, .copy_dst = true });
    try r.writeBuffer(gpa, buf, 16, &data);
    const pl = try r.createKernel(gpa, 5, 2, 0b1000, "wgsl");
    const bg = try r.createBindGroup(gpa, pl, 0, &.{
        .{ .binding = 0, .buffer = buf, .size = 64 },
        .{ .binding = 3, .buffer = buf, .offset = 256, .size = 1 << 32 },
    });
    try r.dispatch(gpa, pl, .{ 4, 2, 1 }, &.{bg});
    try r.readBuffer(gpa, 7, buf, 8, &dst);
    try r.clearBuffer(gpa, buf, 4, 1 << 32);
    try r.writeFile(gpa, buf, 1 << 32, (5 << 32) + 12, 3 << 31);
    try r.submit(gpa, 9);
    const second = try r.createBuffer(gpa, 4, .{ .uniform = true });
    try std.testing.expectEqual(@as(u32, 1), @backingInt(second));

    var it: Iterator = .{ .words = r.batch() };
    var c = (try it.next()).?;
    try std.testing.expectEqual(Op.create_buffer, c.op);
    try std.testing.expectEqual(@as(u64, 1 << 33), c.u64At(1));
    try std.testing.expectEqual(@as(u32, 0x88), c.args[3]); // copy_dst | storage

    c = (try it.next()).?;
    try std.testing.expectEqual(Op.write_buffer, c.op);
    try std.testing.expectEqual(@as(u64, 16), c.u64At(1));
    try std.testing.expectEqualSlices(u8, &data, c.ptrAt([*]const u8, 3)[0..@intCast(c.u64At(5))]);

    c = (try it.next()).?;
    try std.testing.expectEqual(Op.create_kernel, c.op);
    try std.testing.expectEqualSlices(u32, &.{ 0, 5, 2, 0b1000 }, c.args[0..4]);
    try std.testing.expectEqualStrings("wgsl", c.ptrAt([*]const u8, 4)[0..c.args[6]]);

    c = (try it.next()).?;
    try std.testing.expectEqual(Op.create_bind_group, c.op);
    try std.testing.expectEqual(@as(u32, 2), c.args[3]);
    try std.testing.expectEqual(@as(u32, 3), c.args[4 + 6]);
    try std.testing.expectEqual(@as(u64, 1 << 32), (Command{ .op = c.op, .args = c.args[4 + 6 ..] }).u64At(4));

    c = (try it.next()).?;
    try std.testing.expectEqual(Op.dispatch, c.op);
    try std.testing.expectEqualSlices(u32, &.{ 0, 4, 2, 1, 1, 0 }, c.args);

    c = (try it.next()).?;
    try std.testing.expectEqual(Op.read_buffer, c.op);
    try std.testing.expectEqual(@as(*[8]u8, &dst), c.ptrAt(*[8]u8, 6));

    c = (try it.next()).?;
    try std.testing.expectEqual(Op.clear_buffer, c.op);
    try std.testing.expectEqual(@as(u64, 4), c.u64At(1));
    try std.testing.expectEqual(@as(u64, 1 << 32), c.u64At(3));

    c = (try it.next()).?;
    try std.testing.expectEqual(Op.write_file, c.op);
    try std.testing.expectEqual(@as(u32, 0), c.args[0]);
    try std.testing.expectEqual(@as(u64, 1 << 32), c.u64At(1));
    try std.testing.expectEqual(@as(u64, (5 << 32) + 12), c.u64At(3));
    try std.testing.expectEqual(@as(u64, 3 << 31), c.u64At(5));

    c = (try it.next()).?;
    try std.testing.expectEqual(Op.submit, c.op);
    try std.testing.expectEqual(@as(u32, 9), c.args[0]);

    try std.testing.expectEqual(Op.create_buffer, (try it.next()).?.op);
    try std.testing.expectEqual(null, try it.next());
}

test "iterator rejects zero-length and truncated commands" {
    var it: Iterator = .{ .words = &.{@backingInt(Op.submit)} };
    try std.testing.expectError(error.BadLength, it.next());
    it = .{ .words = &.{ header(.submit, 3), 1 } };
    try std.testing.expectError(error.Truncated, it.next());
}
