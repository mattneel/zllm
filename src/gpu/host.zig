//! Native host: executes command batches (cmd.zig) on a Dawn device.
//!
//! Blocking, as the native host is allowed to be: `execute` returns once
//! every submit in the batch has finished on the GPU and every read has been
//! copied back, with one completion per submit and per read.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cmd = @import("cmd.zig");
const dawn = @import("dawn.zig");
const c = dawn.c;

pub const Host = struct {
    gpa: Allocator,
    d: *dawn.Device,
    /// The model file `write_file` copies from.
    file: []const u8 = &.{},
    buffers: std.ArrayList(c.WGPUBuffer) = .empty,
    shaders: std.ArrayList(c.WGPUShaderModule) = .empty,
    pipelines: std.ArrayList(c.WGPUComputePipeline) = .empty,
    bind_groups: std.ArrayList(c.WGPUBindGroup) = .empty,

    encoder: c.WGPUCommandEncoder = null,
    pass: c.WGPUComputePassEncoder = null,
    reads: std.ArrayList(Read) = .empty,
    /// Per-dispatch GPU time, while profiling (startProfile).
    profile: ?Profile = null,

    const Read = struct { token: u32, staging: c.WGPUBuffer, dst: []u8 };

    /// While profiling, each dispatch runs in its own compute pass with
    /// timestamps at its start and end, which serializes dispatches a
    /// shared pass might overlap: totals are an upper bound on a normal
    /// step. Times add up per pipeline.
    pub const Profile = struct {
        query_set: c.WGPUQuerySet,
        resolve: c.WGPUBuffer,
        readback: c.WGPUBuffer,
        /// Each timed dispatch in the open submit.
        pending: std.ArrayList(Key) = .empty,
        /// Nanoseconds and dispatches per pipeline and grid.
        totals: std.AutoArrayHashMapUnmanaged(Key, Total) = .empty,
        /// Dispatches past `capacity` in one submit, not timed.
        untimed: u64 = 0,

        /// Timed dispatches per submit: two of WebGPU's at most 4096
        /// queries per set each.
        pub const capacity = 2048;

        pub const Key = struct { pipeline: u32, grid: [3]u32 };
        pub const Total = struct { ns: u64 = 0, count: u64 = 0 };
    };

    pub const Error = dawn.Error || cmd.DecodeError || Allocator.Error || error{ BadCommand, BadId, NoTimestamps };

    pub fn init(gpa: Allocator, d: *dawn.Device) Host {
        return .{ .gpa = gpa, .d = d };
    }

    pub fn deinit(h: *Host) void {
        h.stopProfile();
        const p = &h.d.p;
        if (h.pass) |pass| p.ComputePassEncoderRelease(pass);
        if (h.encoder) |enc| p.CommandEncoderRelease(enc);
        for (h.reads.items) |r| p.BufferRelease(r.staging);
        for (h.bind_groups.items) |o| if (o) |x| p.BindGroupRelease(x);
        for (h.pipelines.items) |o| if (o) |x| p.ComputePipelineRelease(x);
        for (h.shaders.items) |o| if (o) |x| p.ShaderModuleRelease(x);
        for (h.buffers.items) |o| if (o) |x| {
            p.BufferDestroy(x);
            p.BufferRelease(x);
        };
        h.reads.deinit(h.gpa);
        h.bind_groups.deinit(h.gpa);
        h.pipelines.deinit(h.gpa);
        h.shaders.deinit(h.gpa);
        h.buffers.deinit(h.gpa);
        h.* = undefined;
    }

    /// Starts timing every dispatch (Profile). Needs the device's
    /// timestamp queries.
    pub fn startProfile(h: *Host) Error!void {
        if (!h.d.has_timestamps) return error.NoTimestamps;
        if (h.profile != null) return;
        const p = &h.d.p;
        var qdesc = std.mem.zeroes(c.WGPUQuerySetDescriptor);
        qdesc.type = c.WGPUQueryType_Timestamp;
        qdesc.count = 2 * Profile.capacity;
        const qs = p.DeviceCreateQuerySet(h.d.device, &qdesc) orelse return error.GpuError;
        var bdesc = std.mem.zeroes(c.WGPUBufferDescriptor);
        bdesc.size = 16 * Profile.capacity;
        bdesc.usage = c.WGPUBufferUsage_QueryResolve | c.WGPUBufferUsage_CopySrc;
        const resolve = p.DeviceCreateBuffer(h.d.device, &bdesc) orelse return error.GpuError;
        bdesc.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
        const readback = p.DeviceCreateBuffer(h.d.device, &bdesc) orelse return error.GpuError;
        h.profile = .{ .query_set = qs, .resolve = resolve, .readback = readback };
    }

    /// Stops timing and frees the profile.
    pub fn stopProfile(h: *Host) void {
        var pr = h.profile orelse return;
        const p = &h.d.p;
        p.QuerySetDestroy(pr.query_set);
        p.QuerySetRelease(pr.query_set);
        for ([_]c.WGPUBuffer{ pr.resolve, pr.readback }) |b| {
            p.BufferDestroy(b);
            p.BufferRelease(b);
        }
        pr.pending.deinit(h.gpa);
        pr.totals.deinit(h.gpa);
        h.profile = null;
    }

    /// Runs one batch, appending completions. On `error.GpuError`,
    /// `d.errorMessage()` holds the first WebGPU error message.
    pub fn execute(h: *Host, words: []const u32, completions: *std.ArrayList(cmd.Completion)) Error!void {
        var it: cmd.Iterator = .{ .words = words };
        while (try it.next()) |command| {
            try h.run(command, completions);
            try h.d.takeError();
        }
    }

    fn run(h: *Host, command: cmd.Command, completions: *std.ArrayList(cmd.Completion)) Error!void {
        const p = &h.d.p;
        const a = command.args;
        switch (command.op) {
            .create_buffer => {
                try need(a, 4);
                var desc = std.mem.zeroes(c.WGPUBufferDescriptor);
                desc.size = command.u64At(1);
                desc.usage = a[3];
                const buf = p.DeviceCreateBuffer(h.d.device, &desc);
                try put(c.WGPUBuffer, h.gpa, &h.buffers, a[0], buf);
            },
            .write_buffer => {
                try need(a, 7);
                const data = command.ptrAt([*]const u8, 3);
                const len: usize = @intCast(command.u64At(5));
                p.QueueWriteBuffer(h.d.queue, try get(c.WGPUBuffer, h.buffers, a[0]), command.u64At(1), data, len);
            },
            .write_file => {
                try need(a, 7);
                const start = command.u64At(3);
                const len = command.u64At(5);
                if (start > h.file.len or len > h.file.len - start) return error.BadCommand;
                const data = h.file[@intCast(start)..][0..@intCast(len)];
                p.QueueWriteBuffer(h.d.queue, try get(c.WGPUBuffer, h.buffers, a[0]), command.u64At(1), data.ptr, data.len);
            },
            .create_shader => {
                try need(a, 4);
                const code = command.ptrAt([*]const u8, 1)[0..a[3]];
                var src = std.mem.zeroes(c.WGPUShaderSourceWGSL);
                src.chain.sType = c.WGPUSType_ShaderSourceWGSL;
                src.code = dawn.str(code);
                var desc = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
                desc.nextInChain = &src.chain;
                try put(c.WGPUShaderModule, h.gpa, &h.shaders, a[0], p.DeviceCreateShaderModule(h.d.device, &desc));
            },
            .create_pipeline => {
                try need(a, 2);
                var desc = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
                desc.compute.module = try get(c.WGPUShaderModule, h.shaders, a[1]);
                desc.compute.entryPoint = dawn.str("main");
                try put(c.WGPUComputePipeline, h.gpa, &h.pipelines, a[0], p.DeviceCreateComputePipeline(h.d.device, &desc));
            },
            .create_bind_group => {
                try need(a, 4);
                const count = a[3];
                if (a.len != 4 + count * 6) return error.BadCommand;
                var entries: [16]c.WGPUBindGroupEntry = undefined;
                if (count > entries.len) return error.BadCommand;
                for (entries[0..count], 0..) |*e, i| {
                    const ea: cmd.Command = .{ .op = command.op, .args = a[4 + i * 6 ..][0..6] };
                    e.* = std.mem.zeroes(c.WGPUBindGroupEntry);
                    e.binding = ea.args[0];
                    e.buffer = try get(c.WGPUBuffer, h.buffers, ea.args[1]);
                    e.offset = ea.u64At(2);
                    e.size = ea.u64At(4);
                }
                const layout = p.ComputePipelineGetBindGroupLayout(try get(c.WGPUComputePipeline, h.pipelines, a[1]), a[2]);
                defer p.BindGroupLayoutRelease(layout);
                var desc = std.mem.zeroes(c.WGPUBindGroupDescriptor);
                desc.layout = layout;
                desc.entryCount = count;
                desc.entries = &entries;
                try put(c.WGPUBindGroup, h.gpa, &h.bind_groups, a[0], p.DeviceCreateBindGroup(h.d.device, &desc));
            },
            .dispatch => {
                try need(a, 5);
                if (a.len != 5 + a[4]) return error.BadCommand;
                const timed = if (h.profile) |*pr| pr.pending.items.len < Profile.capacity else false;
                if (h.profile) |*pr| if (!timed) {
                    pr.untimed += 1;
                };
                const pass = if (timed) try h.timedPass() else try h.computePass();
                p.ComputePassEncoderSetPipeline(pass, try get(c.WGPUComputePipeline, h.pipelines, a[0]));
                for (a[5..], 0..) |g, i| {
                    p.ComputePassEncoderSetBindGroup(pass, @intCast(i), try get(c.WGPUBindGroup, h.bind_groups, g), 0, null);
                }
                p.ComputePassEncoderDispatchWorkgroups(pass, a[1], a[2], a[3]);
                if (timed) {
                    h.endPass();
                    try h.profile.?.pending.append(h.gpa, .{ .pipeline = a[0], .grid = a[1..4].* });
                }
            },
            .copy_buffer => {
                try need(a, 8);
                const enc = try h.copyEncoder();
                const src = try get(c.WGPUBuffer, h.buffers, a[0]);
                const dst = try get(c.WGPUBuffer, h.buffers, a[3]);
                p.CommandEncoderCopyBufferToBuffer(enc, src, command.u64At(1), dst, command.u64At(4), command.u64At(6));
            },
            .clear_buffer => {
                try need(a, 5);
                const enc = try h.copyEncoder();
                p.CommandEncoderClearBuffer(enc, try get(c.WGPUBuffer, h.buffers, a[0]), command.u64At(1), command.u64At(3));
            },
            .read_buffer => {
                try need(a, 8);
                const size = command.u64At(4);
                var desc = std.mem.zeroes(c.WGPUBufferDescriptor);
                desc.size = size;
                desc.usage = c.WGPUBufferUsage_MapRead | c.WGPUBufferUsage_CopyDst;
                const staging = p.DeviceCreateBuffer(h.d.device, &desc) orelse return error.GpuError;
                try h.reads.append(h.gpa, .{
                    .token = a[0],
                    .staging = staging,
                    .dst = command.ptrAt([*]u8, 6)[0..@intCast(size)],
                });
                const enc = try h.copyEncoder();
                p.CommandEncoderCopyBufferToBuffer(enc, try get(c.WGPUBuffer, h.buffers, a[1]), command.u64At(2), staging, 0, size);
            },
            .submit => {
                try need(a, 1);
                try h.submit();
                for (h.reads.items) |r| try completions.append(h.gpa, .{ .token = r.token, .status = .ok });
                h.reads.clearRetainingCapacity();
                try completions.append(h.gpa, .{ .token = a[0], .status = .ok });
            },
            .destroy => {
                try need(a, 2);
                const kind = std.enums.fromInt(cmd.Kind, a[0]) orelse return error.BadCommand;
                switch (kind) {
                    .buffer => if (try take(c.WGPUBuffer, h.buffers, a[1])) |x| {
                        p.BufferDestroy(x);
                        p.BufferRelease(x);
                    },
                    .shader => if (try take(c.WGPUShaderModule, h.shaders, a[1])) |x| p.ShaderModuleRelease(x),
                    .pipeline => if (try take(c.WGPUComputePipeline, h.pipelines, a[1])) |x| p.ComputePipelineRelease(x),
                    .bind_group => if (try take(c.WGPUBindGroup, h.bind_groups, a[1])) |x| p.BindGroupRelease(x),
                }
            },
            _ => return error.BadCommand,
        }
    }

    fn encoderOrNew(h: *Host) Error!c.WGPUCommandEncoder {
        if (h.encoder) |e| return e;
        h.encoder = h.d.p.DeviceCreateCommandEncoder(h.d.device, null) orelse return error.GpuError;
        return h.encoder;
    }

    fn computePass(h: *Host) Error!c.WGPUComputePassEncoder {
        if (h.pass) |pass| return pass;
        h.pass = h.d.p.CommandEncoderBeginComputePass(try h.encoderOrNew(), null) orelse return error.GpuError;
        return h.pass;
    }

    /// A new compute pass that timestamps its start and end into the
    /// profile's next two queries.
    fn timedPass(h: *Host) Error!c.WGPUComputePassEncoder {
        h.endPass();
        const pr = &h.profile.?;
        const i: u32 = @intCast(pr.pending.items.len);
        var writes = std.mem.zeroes(c.WGPUPassTimestampWrites);
        writes.querySet = pr.query_set;
        writes.beginningOfPassWriteIndex = 2 * i;
        writes.endOfPassWriteIndex = 2 * i + 1;
        var desc = std.mem.zeroes(c.WGPUComputePassDescriptor);
        desc.timestampWrites = &writes;
        h.pass = h.d.p.CommandEncoderBeginComputePass(try h.encoderOrNew(), &desc) orelse return error.GpuError;
        return h.pass;
    }

    /// The encoder, with any open compute pass ended so copies can follow.
    fn copyEncoder(h: *Host) Error!c.WGPUCommandEncoder {
        h.endPass();
        return h.encoderOrNew();
    }

    fn endPass(h: *Host) void {
        const pass = h.pass orelse return;
        h.d.p.ComputePassEncoderEnd(pass);
        h.d.p.ComputePassEncoderRelease(pass);
        h.pass = null;
    }

    /// Submits the open encoder (if any), waits for the queue, then maps and
    /// copies out every pending read.
    fn submit(h: *Host) Error!void {
        const p = &h.d.p;
        h.endPass();
        const timed: usize = if (h.profile) |pr| pr.pending.items.len else 0;
        if (timed > 0) {
            const pr = &h.profile.?;
            const enc = try h.encoderOrNew();
            p.CommandEncoderResolveQuerySet(enc, pr.query_set, 0, @intCast(2 * timed), pr.resolve, 0);
            p.CommandEncoderCopyBufferToBuffer(enc, pr.resolve, 0, pr.readback, 0, 16 * timed);
        }
        if (h.encoder) |enc| {
            const buf = p.CommandEncoderFinish(enc, null);
            p.CommandEncoderRelease(enc);
            h.encoder = null;
            p.QueueSubmit(h.d.queue, 1, &buf);
            p.CommandBufferRelease(buf);
        }
        var done = std.mem.zeroes(c.WGPUQueueWorkDoneCallbackInfo);
        done.mode = c.WGPUCallbackMode_WaitAnyOnly;
        done.callback = onWorkDone;
        h.d.wait(p.QueueOnSubmittedWorkDone(h.d.queue, done));
        try h.d.takeError();

        for (h.reads.items) |r| {
            var status: c.WGPUMapAsyncStatus = c.WGPUMapAsyncStatus_Error;
            var info = std.mem.zeroes(c.WGPUBufferMapCallbackInfo);
            info.mode = c.WGPUCallbackMode_WaitAnyOnly;
            info.callback = onMapped;
            info.userdata1 = &status;
            h.d.wait(p.BufferMapAsync(r.staging, c.WGPUMapMode_Read, 0, r.dst.len, info));
            if (status != c.WGPUMapAsyncStatus_Success) return error.GpuError;
            const src: [*]const u8 = @ptrCast(p.BufferGetConstMappedRange(r.staging, 0, r.dst.len) orelse return error.GpuError);
            @memcpy(r.dst, src[0..r.dst.len]);
            p.BufferUnmap(r.staging);
            p.BufferRelease(r.staging);
        }
        if (timed > 0) try h.collectProfile(timed);
    }

    /// Adds the open submit's `timed` dispatches to the profile's totals.
    fn collectProfile(h: *Host, timed: usize) Error!void {
        const p = &h.d.p;
        const pr = &h.profile.?;
        var status: c.WGPUMapAsyncStatus = c.WGPUMapAsyncStatus_Error;
        var info = std.mem.zeroes(c.WGPUBufferMapCallbackInfo);
        info.mode = c.WGPUCallbackMode_WaitAnyOnly;
        info.callback = onMapped;
        info.userdata1 = &status;
        h.d.wait(p.BufferMapAsync(pr.readback, c.WGPUMapMode_Read, 0, 16 * timed, info));
        if (status != c.WGPUMapAsyncStatus_Success) return error.GpuError;
        const raw: [*]const u64 = @ptrCast(@alignCast(p.BufferGetConstMappedRange(pr.readback, 0, 16 * timed) orelse return error.GpuError));
        for (pr.pending.items, 0..) |key, i| {
            const t = try pr.totals.getOrPutValue(h.gpa, key, .{});
            t.value_ptr.ns += raw[2 * i + 1] -| raw[2 * i];
            t.value_ptr.count += 1;
        }
        p.BufferUnmap(pr.readback);
        pr.pending.clearRetainingCapacity();
    }

    fn onWorkDone(_: c.WGPUQueueWorkDoneStatus, _: c.WGPUStringView, _: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {}

    fn onMapped(status: c.WGPUMapAsyncStatus, _: c.WGPUStringView, userdata1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
        const out: *c.WGPUMapAsyncStatus = @ptrCast(@alignCast(userdata1.?));
        out.* = status;
    }
};

fn need(args: []const u32, n: usize) error{BadCommand}!void {
    if (args.len < n) return error.BadCommand;
}

fn put(comptime T: type, gpa: Allocator, list: *std.ArrayList(T), id: u32, obj: T) Host.Error!void {
    if (obj == null) return error.GpuError;
    if (id >= list.items.len) try list.appendNTimes(gpa, null, id + 1 - list.items.len);
    if (list.items[id] != null) return error.BadId;
    list.items[id] = obj;
}

fn get(comptime T: type, list: std.ArrayList(T), id: u32) error{BadId}!T {
    if (id >= list.items.len) return error.BadId;
    return list.items[id] orelse error.BadId;
}

fn take(comptime T: type, list: std.ArrayList(T), id: u32) error{BadId}!T {
    const obj = try get(T, list, id);
    list.items[id] = null;
    return obj;
}
