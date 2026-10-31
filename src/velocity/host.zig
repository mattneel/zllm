//! Velocity's host: executes command batches (gpu/cmd.zig) on the native
//! Direct3D 12 layer (velocity.h, host.cpp).
//!
//! The same semantics as the Dawn host (gpu/dawn_host.zig), blocking: a
//! `submit` returns once the GPU has finished it, with one completion per
//! submit and per read. As on WebGPU, `write_buffer` and `write_file` land
//! before the next submit's work, even work recorded ahead of them: writes
//! are staged until the submit, and the submit runs them first.
//!
//! Every GPU access but DirectStorage's is a compute UAV access, so a
//! submission never mixes resource states: copies, clears, uploads (through
//! mapped host memory) and readbacks are the host's own kernels
//! (kernels/copy.hlsl, kernels/fill.hlsl). Dispatches are separated by UAV
//! barriers, the ordering WebGPU gives dispatches in one pass.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cmd = @import("../gpu/cmd.zig");
const Timings = @import("../gpu/timings.zig").Timings;
const Kernel = @import("../kernels/kernels.zig").Kernel;
pub const c = @import("velocity_c");

pub const timeout_ms: u32 = 60_000;

pub const Error = error{ GpuError, DeviceLost, BadCommand, BadId, NoTimestamps, UnsupportedKernel } || cmd.DecodeError || Allocator.Error;

/// A Velocity device: one D3D12 adapter with its compute and DirectStorage
/// queues.
pub const Device = struct {
    ctx: *c.vel_context,
    caps: c.vel_caps,
    /// The last error, for errorMessage.
    message: [4096]u8 = undefined,
    message_len: usize = 0,
    lost: bool = false,

    pub fn init(gpa: Allocator, adapter: u32) Error!*Device {
        var options = std.mem.zeroes(c.vel_options);
        options.adapter_index = adapter;
        var ctx: ?*c.vel_context = null;
        const r = c.vel_create(&options, &ctx);
        if (r < 0) {
            std.log.err("velocity: cannot create a device on adapter {d}: HRESULT 0x{x:0>8}", .{ adapter, @as(u32, @bitCast(r)) });
            return error.GpuError;
        }
        const d = try gpa.create(Device);
        d.* = .{ .ctx = ctx.?, .caps = undefined };
        errdefer d.deinit(gpa);
        try d.check(c.vel_get_caps(d.ctx, &d.caps));
        return d;
    }

    pub fn deinit(d: *Device, gpa: Allocator) void {
        _ = c.vel_destroy(d.ctx, timeout_ms);
        gpa.destroy(d);
    }

    /// Maps a result to an error, keeping its message for errorMessage.
    pub fn check(d: *Device, r: c.vel_result) Error!void {
        if (r >= 0) return;
        const code: u32 = @bitCast(r);
        var w: std.Io.Writer = .fixed(&d.message);
        w.print("HRESULT 0x{x:0>8}", .{code}) catch {};
        const len = w.end;
        if (len + 2 < d.message.len) {
            d.message[len] = ':';
            d.message[len + 1] = ' ';
            const n = c.vel_last_error(d.ctx, d.message[len + 2 ..].ptr, d.message.len - len - 2);
            d.message_len = if (n > 0) len + 2 + n else len;
        } else d.message_len = len;
        switch (code) {
            // DXGI_ERROR_DEVICE_REMOVED, _HUNG, _RESET, _DRIVER_INTERNAL_ERROR
            0x887A0005, 0x887A0006, 0x887A0007, 0x887A0020 => {
                d.lost = true;
                return error.DeviceLost;
            },
            else => return error.GpuError,
        }
    }

    /// Fails a dispatch of kernel `k` (kernels.Kernel), which Velocity
    /// does not implement.
    fn missing(d: *Device, k: u32) Error {
        const name = if (std.enums.fromInt(Kernel, k)) |kernel| @tagName(kernel) else "?";
        const msg = std.fmt.bufPrint(&d.message, "kernel {s} has no Velocity implementation", .{name}) catch &.{};
        d.message_len = msg.len;
        return error.UnsupportedKernel;
    }

    pub fn errorMessage(d: *const Device) []const u8 {
        return d.message[0..d.message_len];
    }

    pub fn adapterName(d: *const Device) []const u8 {
        return std.mem.sliceTo(&d.caps.adapter_name, 0);
    }

    /// Removes the device as a driver fault would; later calls fail with
    /// error.DeviceLost.
    pub fn forceLoss(d: *Device) void {
        _ = c.vel_remove_device(d.ctx);
    }
};

/// A kernel's pipeline, or `missing` where Velocity has no implementation
/// of the kernel (an empty artifact).
const Pipeline = union(enum) { ready: *c.vel_pipeline, missing: u32 };

const Buffer = struct {
    b: *c.vel_buffer,
    info: c.vel_buffer_info,
    size: u64,
};

/// A bind group as root constants: (SRV, UAV, byte offset) per binding.
const Group = struct {
    words: [max_words]u32,
    len: u32,
};
const max_words = 60;

const Op = union(enum) {
    dispatch: struct { pipeline: *c.vel_pipeline, id: u32, grid: [3]u32, words: [max_words]u32, len: u32 },
    copy: Copy,
    fill: struct { dst: u32, dst_offset: u32, words: u32 },
};

const Copy = struct { src: u32, src_offset: u32, dst: u32, dst_offset: u32, words: u32 };

const Write = struct { dst: u32, dst_offset: u32, start: usize, len: usize };
const Read = struct { token: u32, dst: []u8, offset: u64 };

pub const Host = struct {
    gpa: Allocator,
    d: *Device,
    /// The model file `write_file` reads from (setFile).
    file: ?*c.vel_file = null,
    file_size: u64 = 0,
    buffers: std.ArrayList(?Buffer) = .empty,
    pipelines: std.ArrayList(?Pipeline) = .empty,
    groups: std.ArrayList(?Group) = .empty,
    copy: *c.vel_pipeline,
    fill: *c.vel_pipeline,

    /// What the next submit runs: staged writes first, then `ops`.
    writes: std.ArrayList(Write) = .empty,
    write_data: std.ArrayList(u8) = .empty,
    ops: std.ArrayList(Op) = .empty,
    reads: std.ArrayList(Read) = .empty,
    read_bytes: u64 = 0,
    /// Buffers destroyed while ops that may use them wait for the submit.
    released: std.ArrayList(u32) = .empty,
    /// Host-memory staging for writes and reads, grown to fit a submit.
    upload: ?Buffer = null,
    download: ?Buffer = null,
    /// The newest DirectStorage ticket, and the newest the compute queue
    /// has been made to wait for.
    storage: u64 = 0,
    storage_waited: u64 = 0,
    /// The newest compute ticket, which DirectStorage writes wait for.
    compute: u64 = 0,
    /// Per-dispatch GPU times, while profiling (startProfile).
    timings: ?Timings = null,
    pending_times: std.ArrayList(Timings.Key) = .empty,

    /// Timestamp pairs per submit: the 4096 timestamps of the query heap.
    pub const timing_capacity = 2048;

    const copy_source = @embedFile("kernels/copy.hlsl");
    const fill_source = @embedFile("kernels/fill.hlsl");

    pub fn init(gpa: Allocator, d: *Device) Error!Host {
        var copy: ?*c.vel_pipeline = null;
        try d.check(c.vel_compile_pipeline(d.ctx, copy_source, copy_source.len, "cs_6_6", &copy));
        var fill: ?*c.vel_pipeline = null;
        try d.check(c.vel_compile_pipeline(d.ctx, fill_source, fill_source.len, "cs_6_6", &fill));
        return .{ .gpa = gpa, .d = d, .copy = copy.?, .fill = fill.? };
    }

    pub fn deinit(h: *Host) void {
        const ctx = h.d.ctx;
        for (h.buffers.items) |o| if (o) |b| {
            _ = c.vel_release_buffer(ctx, b.b, timeout_ms);
        };
        for ([_]?Buffer{ h.upload, h.download }) |o| if (o) |b| {
            _ = c.vel_release_buffer(ctx, b.b, timeout_ms);
        };
        for (h.pipelines.items) |o| if (o) |p| switch (p) {
            .ready => |x| _ = c.vel_release_pipeline(ctx, x, timeout_ms),
            .missing => {},
        };
        _ = c.vel_release_pipeline(ctx, h.copy, timeout_ms);
        _ = c.vel_release_pipeline(ctx, h.fill, timeout_ms);
        if (h.file) |f| _ = c.vel_release_file(ctx, f, timeout_ms);
        h.buffers.deinit(h.gpa);
        h.pipelines.deinit(h.gpa);
        h.groups.deinit(h.gpa);
        h.writes.deinit(h.gpa);
        h.write_data.deinit(h.gpa);
        h.ops.deinit(h.gpa);
        h.reads.deinit(h.gpa);
        h.released.deinit(h.gpa);
        h.pending_times.deinit(h.gpa);
        if (h.timings) |*t| t.deinit(h.gpa);
        h.* = undefined;
    }

    /// Opens the model file `write_file` reads from, through DirectStorage.
    pub fn setFile(h: *Host, gpa: Allocator, path: []const u8) Error!void {
        const z = try gpa.dupeSentinel(u8, path, 0);
        defer gpa.free(z);
        if (h.file) |f| {
            try h.d.check(c.vel_release_file(h.d.ctx, f, timeout_ms));
            h.file = null;
        }
        var f: ?*c.vel_file = null;
        try h.d.check(c.vel_open_file(h.d.ctx, z.ptr, &f));
        h.file = f;
        try h.d.check(c.vel_file_size(f, &h.file_size));
    }

    pub fn startProfile(h: *Host) Error!void {
        if (h.timings == null) h.timings = .{ .capacity = timing_capacity };
    }

    pub fn stopProfile(h: *Host) void {
        if (h.timings) |*t| t.deinit(h.gpa);
        h.timings = null;
        h.pending_times.clearRetainingCapacity();
    }

    /// Runs one batch, appending completions. On `error.GpuError`,
    /// `d.errorMessage()` holds the failure.
    /// error.UnsupportedKernel drops the submission in progress.
    pub fn execute(h: *Host, words: []const u32, completions: *std.ArrayList(cmd.Completion)) Error!void {
        if (h.d.lost) return error.DeviceLost;
        var it: cmd.Iterator = .{ .words = words };
        while (try it.next()) |command| h.run(command, completions) catch |err| {
            if (err == error.UnsupportedKernel) h.drop();
            return err;
        };
    }

    fn drop(h: *Host) void {
        h.writes.clearRetainingCapacity();
        h.write_data.clearRetainingCapacity();
        h.ops.clearRetainingCapacity();
        h.reads.clearRetainingCapacity();
        h.read_bytes = 0;
    }

    fn run(h: *Host, command: cmd.Command, completions: *std.ArrayList(cmd.Completion)) Error!void {
        const a = command.args;
        switch (command.op) {
            .create_buffer => {
                try need(a, 4);
                const size = command.u64At(1);
                var b: ?*c.vel_buffer = null;
                var info: c.vel_buffer_info = undefined;
                try h.d.check(c.vel_create_resource(h.d.ctx, @max(size, 4), c.VEL_RESOURCE_DEVICE, &b, &info));
                try put(Buffer, h.gpa, &h.buffers, a[0], .{ .b = b.?, .info = info, .size = size });
            },
            .write_buffer => {
                try need(a, 7);
                const len: usize = @intCast(command.u64At(5));
                const offset = command.u64At(1);
                const dst = try get(Buffer, h.buffers, a[0]);
                if (len % 4 != 0 or offset % 4 != 0 or offset + len > dst.size) return error.BadCommand;
                if (len == 0) return;
                const data = command.ptrAt([*]const u8, 3)[0..len];
                // 16-byte aligned starts keep the upload copy's words aligned.
                const start = std.mem.alignForward(usize, h.write_data.items.len, 16);
                try h.write_data.resize(h.gpa, start + len);
                @memcpy(h.write_data.items[start..][0..len], data);
                try h.writes.append(h.gpa, .{ .dst = dst.info.uav_index, .dst_offset = @intCast(offset), .start = start, .len = len });
            },
            .write_file => {
                try need(a, 7);
                try h.writeFile(try get(Buffer, h.buffers, a[0]), command.u64At(1), command.u64At(3), command.u64At(5));
            },
            .create_kernel => {
                try need(a, 6);
                if (a[5] == 0) return put(Pipeline, h.gpa, &h.pipelines, a[0], .{ .missing = a[1] });
                const source = command.ptrAt([*]const u8, 3)[0..a[5]];
                var p: ?*c.vel_pipeline = null;
                h.d.check(c.vel_compile_pipeline(h.d.ctx, source.ptr, source.len, target(source), &p)) catch |err| {
                    std.log.err("velocity: kernel {d} (variant {d}) does not compile: {s}", .{ a[1], a[2], h.d.errorMessage() });
                    return err;
                };
                try put(Pipeline, h.gpa, &h.pipelines, a[0], .{ .ready = p.? });
            },
            .create_bind_group => {
                try need(a, 4);
                const count = a[3];
                if (a.len != 4 + count * 6 or a[2] != 0) return error.BadCommand;
                var g: Group = .{ .words = @splat(0), .len = 0 };
                for (0..count) |i| {
                    const e: cmd.Command = .{ .op = command.op, .args = a[4 + i * 6 ..][0..6] };
                    const binding = e.args[0];
                    if (3 * binding + 3 > max_words) return error.BadCommand;
                    const b = try get(Buffer, h.buffers, e.args[1]);
                    const offset = e.u64At(2);
                    if (offset > std.math.maxInt(u32)) return error.BadCommand;
                    g.words[3 * binding ..][0..3].* = .{ b.info.srv_index, b.info.uav_index, @intCast(offset) };
                    g.len = @max(g.len, 3 * binding + 3);
                }
                try put(Group, h.gpa, &h.groups, a[0], g);
            },
            .dispatch => {
                try need(a, 5);
                if (a.len != 5 + a[4] or a[4] > 1) return error.BadCommand;
                const g: Group = if (a[4] == 1) try get(Group, h.groups, a[5]) else .{ .words = @splat(0), .len = 0 };
                const pipeline = switch (try get(Pipeline, h.pipelines, a[0])) {
                    .ready => |p| p,
                    .missing => |k| return h.d.missing(k),
                };
                try h.ops.append(h.gpa, .{ .dispatch = .{
                    .pipeline = pipeline,
                    .id = a[0],
                    .grid = a[1..4].*,
                    .words = g.words,
                    .len = g.len,
                } });
            },
            .copy_buffer => {
                try need(a, 8);
                const src = try get(Buffer, h.buffers, a[0]);
                const dst = try get(Buffer, h.buffers, a[3]);
                const so = command.u64At(1);
                const do = command.u64At(4);
                const size = command.u64At(6);
                if ((so | do | size) % 4 != 0 or so + size > src.size or do + size > dst.size) return error.BadCommand;
                if (size > 0) try h.ops.append(h.gpa, .{ .copy = .{
                    .src = src.info.uav_index,
                    .src_offset = @intCast(so),
                    .dst = dst.info.uav_index,
                    .dst_offset = @intCast(do),
                    .words = @intCast(size / 4),
                } });
            },
            .clear_buffer => {
                try need(a, 5);
                const dst = try get(Buffer, h.buffers, a[0]);
                const offset = command.u64At(1);
                const size = command.u64At(3);
                if ((offset | size) % 4 != 0 or offset + size > dst.size) return error.BadCommand;
                if (size > 0) try h.ops.append(h.gpa, .{ .fill = .{ .dst = dst.info.uav_index, .dst_offset = @intCast(offset), .words = @intCast(size / 4) } });
            },
            .read_buffer => {
                try need(a, 8);
                const src = try get(Buffer, h.buffers, a[1]);
                const offset = command.u64At(2);
                const size = command.u64At(4);
                if ((offset | size) % 4 != 0 or offset + size > src.size) return error.BadCommand;
                const at = std.mem.alignForward(u64, h.read_bytes, 256);
                h.read_bytes = at + size;
                try h.reads.append(h.gpa, .{ .token = a[0], .dst = command.ptrAt([*]u8, 6)[0..@intCast(size)], .offset = at });
                // The destination is the download buffer, known at submit;
                // dst = maxInt marks it.
                if (size > 0) try h.ops.append(h.gpa, .{ .copy = .{
                    .src = src.info.uav_index,
                    .src_offset = @intCast(offset),
                    .dst = std.math.maxInt(u32),
                    .dst_offset = @intCast(at),
                    .words = @intCast(size / 4),
                } });
            },
            .submit => {
                try need(a, 1);
                try h.submit();
                for (h.reads.items) |r| try completions.append(h.gpa, .{ .token = r.token, .status = .ok });
                h.reads.clearRetainingCapacity();
                h.read_bytes = 0;
                try completions.append(h.gpa, .{ .token = a[0], .status = .ok });
            },
            .destroy => {
                try need(a, 2);
                const kind = std.enums.fromInt(cmd.Kind, a[0]) orelse return error.BadCommand;
                switch (kind) {
                    .buffer => {
                        _ = try get(Buffer, h.buffers, a[1]);
                        if (h.ops.items.len + h.writes.items.len > 0) {
                            try h.released.append(h.gpa, a[1]);
                        } else try h.release(a[1]);
                    },
                    .pipeline => switch (try take(Pipeline, h.pipelines, a[1])) {
                        .ready => |p| try h.d.check(c.vel_release_pipeline(h.d.ctx, p, timeout_ms)),
                        .missing => {},
                    },
                    .bind_group => _ = try take(Group, h.groups, a[1]),
                }
            },
            _ => return error.BadCommand,
        }
    }

    fn release(h: *Host, id: u32) Error!void {
        const b = try take(Buffer, h.buffers, id);
        try h.d.check(c.vel_release_buffer(h.d.ctx, b.b, timeout_ms));
    }

    /// Enqueues DirectStorage reads of the file range into `dst`, after the
    /// newest submitted compute work; the next submit waits for them.
    fn writeFile(h: *Host, dst: Buffer, offset: u64, file_offset: u64, size: u64) Error!void {
        const f = h.file orelse return error.BadCommand;
        if (file_offset > h.file_size or size > h.file_size - file_offset or offset + size > dst.size) return error.BadCommand;
        // Requests stay within DirectStorage's staging buffer (host.cpp's
        // default 32 MiB), and within half the queue per call so two calls
        // in flight fit it.
        const chunk: u64 = 32 << 20;
        const per_call = 510;
        var ranges: [per_call]c.vel_range = undefined;
        var done: u64 = 0;
        while (done < size) {
            var n: usize = 0;
            while (n < per_call and done < size) : (n += 1) {
                const len = @min(chunk, size - done);
                ranges[n] = .{ .file_offset = file_offset + done, .buffer_offset = offset + done, .bytes = @intCast(len), .reserved = 0 };
                done += len;
            }
            const previous = h.storage;
            try h.d.check(c.vel_read_ranges_after(h.d.ctx, f, dst.b, &ranges, @intCast(n), h.compute, &h.storage));
            // At most two calls in flight.
            if (previous > 0) try h.d.check(c.vel_wait_storage(h.d.ctx, previous, timeout_ms));
        }
    }

    /// Grows host staging buffer `slot` to at least `bytes`.
    fn staging(h: *Host, slot: *?Buffer, bytes: u64) Error!Buffer {
        if (slot.*) |b| {
            if (b.size >= bytes) return b;
            try h.d.check(c.vel_release_buffer(h.d.ctx, b.b, timeout_ms));
            slot.* = null;
        }
        const size = @max(bytes, 1 << 20);
        var b: ?*c.vel_buffer = null;
        var info: c.vel_buffer_info = undefined;
        try h.d.check(c.vel_create_resource(h.d.ctx, size, c.VEL_RESOURCE_HOST, &b, &info));
        slot.* = .{ .b = b.?, .info = info, .size = size };
        return slot.*.?;
    }

    /// Runs the staged writes, then the ops, as one submission, and waits
    /// for it: reads are in place and timings collected on return.
    fn submit(h: *Host) Error!void {
        const ctx = h.d.ctx;
        const upload = if (h.writes.items.len > 0) try h.staging(&h.upload, h.write_data.items.len) else null;
        const download = if (h.read_bytes > 0) try h.staging(&h.download, h.read_bytes) else null;
        if (upload) |u| {
            const mapped: [*]u8 = @ptrCast(u.info.mapped.?);
            @memcpy(mapped[0..h.write_data.items.len], h.write_data.items);
        }
        if (h.storage > h.storage_waited) {
            try h.d.check(c.vel_queue_wait_storage(ctx, h.storage));
            h.storage_waited = h.storage;
        }
        // The writes run as their own command list: buffers return to the
        // common state between lists, so kernels may read uploaded data
        // through SRVs (common.hlsli's ro) in the next.
        if (h.writes.items.len > 0) {
            try h.d.check(c.vel_begin(ctx, timeout_ms));
            for (h.writes.items) |w| {
                try h.runCopy(.{ .src = upload.?.info.uav_index, .src_offset = @intCast(w.start), .dst = w.dst, .dst_offset = w.dst_offset, .words = @intCast(w.len / 4) });
            }
            var ticket: u64 = 0;
            try h.d.check(c.vel_submit(ctx, &ticket));
        }
        try h.d.check(c.vel_begin(ctx, timeout_ms));
        var first = true;
        var times: [2 * timing_capacity]u32 = undefined;
        h.pending_times.clearRetainingCapacity();
        for (h.ops.items) |op| {
            if (!first) try h.d.check(c.vel_barrier(ctx));
            first = false;
            switch (op) {
                .dispatch => |dp| {
                    const timed = if (h.timings) |*t| blk: {
                        if (h.pending_times.items.len < timing_capacity) break :blk true;
                        t.untimed += 1;
                        break :blk false;
                    } else false;
                    const i = h.pending_times.items.len;
                    if (timed) try h.d.check(c.vel_timestamp(ctx, &times[2 * i]));
                    try h.d.check(c.vel_dispatch(ctx, dp.pipeline, &dp.words, dp.len, dp.grid[0], dp.grid[1], dp.grid[2]));
                    if (timed) {
                        try h.d.check(c.vel_timestamp(ctx, &times[2 * i + 1]));
                        try h.pending_times.append(h.gpa, .{ .pipeline = dp.id, .grid = dp.grid });
                    }
                },
                .copy => |cp| {
                    var x = cp;
                    if (x.dst == std.math.maxInt(u32)) x.dst = download.?.info.uav_index;
                    try h.runCopy(x);
                },
                .fill => |f| {
                    const words = [_]u32{ f.dst, f.dst_offset, f.words };
                    try h.d.check(c.vel_dispatch(ctx, h.fill, &words, words.len, groupsFor(f.words), 1, 1));
                },
            }
        }
        var ticket: u64 = 0;
        try h.d.check(c.vel_submit(ctx, &ticket));
        h.compute = ticket;
        try h.d.check(c.vel_wait_compute(ctx, ticket, timeout_ms));

        if (download) |dl| {
            const mapped: [*]const u8 = @ptrCast(dl.info.mapped.?);
            for (h.reads.items) |r| @memcpy(r.dst, mapped[@intCast(r.offset)..][0..r.dst.len]);
        }
        if (h.pending_times.items.len > 0) {
            const n = 2 * h.pending_times.items.len;
            var ticks: [2 * timing_capacity]u64 = undefined;
            var frequency: u64 = 0;
            try h.d.check(c.vel_read_timestamps(ctx, @intCast(n), &ticks, &frequency));
            for (h.pending_times.items, 0..) |key, i| {
                const dt = ticks[2 * i + 1] -| ticks[2 * i];
                try h.timings.?.add(h.gpa, key, @intCast(@as(u128, dt) * std.time.ns_per_s / @max(frequency, 1)));
            }
            h.pending_times.clearRetainingCapacity();
        }
        h.writes.clearRetainingCapacity();
        h.write_data.clearRetainingCapacity();
        h.ops.clearRetainingCapacity();
        for (h.released.items) |id| try h.release(id);
        h.released.clearRetainingCapacity();
    }

    fn runCopy(h: *Host, x: Copy) Error!void {
        const words = [_]u32{ x.src, x.src_offset, x.dst, x.dst_offset, x.words };
        try h.d.check(c.vel_dispatch(h.d.ctx, h.copy, &words, words.len, groupsFor(x.words), 1, 1));
    }
};

/// Groups for the grid-strided copy and fill kernels (256 words a group).
fn groupsFor(words: u32) u32 {
    return @max(1, @min(65535, (words + 255) / 256));
}

/// The shader target a kernel's source asks for with a line
/// "// velocity: cs_6_10" (LinAlg); cs_6_6 (bindless) otherwise.
fn target(source: []const u8) [*:0]const u8 {
    return if (std.mem.indexOf(u8, source, "\n// velocity: cs_6_10\n") != null) "cs_6_10" else "cs_6_6";
}

fn need(args: []const u32, n: usize) error{BadCommand}!void {
    if (args.len < n) return error.BadCommand;
}

fn put(comptime T: type, gpa: Allocator, list: *std.ArrayList(?T), id: u32, obj: T) Error!void {
    if (id >= list.items.len) try list.appendNTimes(gpa, null, id + 1 - list.items.len);
    if (list.items[id] != null) return error.BadId;
    list.items[id] = obj;
}

fn get(comptime T: type, list: std.ArrayList(?T), id: u32) error{BadId}!T {
    if (id >= list.items.len) return error.BadId;
    return list.items[id] orelse error.BadId;
}

fn take(comptime T: type, list: std.ArrayList(?T), id: u32) error{BadId}!T {
    const obj = try get(T, list, id);
    list.items[id] = null;
    return obj;
}
