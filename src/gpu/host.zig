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
    buffers: std.ArrayList(c.WGPUBuffer) = .empty,
    shaders: std.ArrayList(c.WGPUShaderModule) = .empty,
    pipelines: std.ArrayList(c.WGPUComputePipeline) = .empty,
    bind_groups: std.ArrayList(c.WGPUBindGroup) = .empty,

    encoder: c.WGPUCommandEncoder = null,
    pass: c.WGPUComputePassEncoder = null,
    reads: std.ArrayList(Read) = .empty,

    const Read = struct { token: u32, staging: c.WGPUBuffer, dst: []u8 };

    pub const Error = dawn.Error || cmd.DecodeError || Allocator.Error || error{ BadCommand, BadId };

    pub fn init(gpa: Allocator, d: *dawn.Device) Host {
        return .{ .gpa = gpa, .d = d };
    }

    pub fn deinit(h: *Host) void {
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
                const pass = try h.computePass();
                p.ComputePassEncoderSetPipeline(pass, try get(c.WGPUComputePipeline, h.pipelines, a[0]));
                for (a[5..], 0..) |g, i| {
                    p.ComputePassEncoderSetBindGroup(pass, @intCast(i), try get(c.WGPUBindGroup, h.bind_groups, g), 0, null);
                }
                p.ComputePassEncoderDispatchWorkgroups(pass, a[1], a[2], a[3]);
            },
            .copy_buffer => {
                try need(a, 8);
                const enc = try h.copyEncoder();
                const src = try get(c.WGPUBuffer, h.buffers, a[0]);
                const dst = try get(c.WGPUBuffer, h.buffers, a[3]);
                p.CommandEncoderCopyBufferToBuffer(enc, src, command.u64At(1), dst, command.u64At(4), command.u64At(6));
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
