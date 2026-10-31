//! The platform layer: a GPU device and the host that executes the command
//! stream (cmd.zig) on it, for every backend (backend/backend.zig). The core
//! only talks to these; each backend's own device and host live behind them.

const std = @import("std");
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");
const cmd = @import("cmd.zig");
const dawn = @import("dawn.zig");
const dawn_host = @import("dawn_host.zig");
const Timings = @import("timings.zig").Timings;
pub const Backend = @import("../backend/backend.zig").Backend;
const velocity = if (build_options.velocity) @import("../velocity/host.zig") else @import("../velocity/absent.zig");

/// error.UnsupportedKernel: a kernel the backend lacks was dispatched
/// (kernels.Set.missing); the submission in progress is dropped.
pub const Error = dawn_host.Host.Error || velocity.Error || error{ VelocityNotBuilt, UnsupportedKernel };

/// Which device to open.
pub const Open = struct {
    backend: Backend = .webgpu,
    /// The Dawn library (WebGPU).
    dawn_lib: []const u8 = dawn.default_library,
    /// The adapter index (Velocity: DXGI's high-performance order).
    adapter: u32 = 0,
};

/// What the planner needs to know about buffer bindings.
pub const Limits = struct {
    /// The largest buffer binding (WebGPU's maxStorageBufferBindingSize).
    max_binding: u64,
    /// Binding offsets' alignment (minStorageBufferOffsetAlignment).
    offset_align: u64,
};

/// The adapter, as the plan cache keys it and the CLI shows it.
pub const Adapter = struct {
    vendor: []const u8,
    architecture: []const u8,
    device: []const u8,
    description: []const u8,
    /// The API underneath: "d3d12", "vulkan", "metal", ... for WebGPU;
    /// "d3d12" for Velocity.
    api: []const u8,
};

pub const Device = union(Backend) {
    webgpu: *dawn.Device,
    velocity: *velocity.Device,

    pub fn open(gpa: Allocator, o: Open) !Device {
        return switch (o.backend) {
            .webgpu => .{ .webgpu = try dawn.Device.init(gpa, o.dawn_lib) },
            .velocity => if (!build_options.velocity) error.VelocityNotBuilt else .{ .velocity = try velocity.Device.init(gpa, o.adapter) },
        };
    }

    pub fn deinit(d: Device, gpa: Allocator) void {
        switch (d) {
            inline else => |x| x.deinit(gpa),
        }
    }

    pub fn backend(d: Device) Backend {
        return d;
    }

    pub fn limits(d: Device) Limits {
        return switch (d) {
            .webgpu => |x| .{ .max_binding = x.limits.maxStorageBufferBindingSize, .offset_align = x.limits.minStorageBufferOffsetAlignment },
            // Velocity's raw views address bytes with 32 bits and its
            // offsets need 4, but Dawn's D3D12 limits (2 GiB, 256) keep the
            // plans the same as WebGPU's on the same machine.
            .velocity => .{ .max_binding = 1 << 31, .offset_align = 256 },
        };
    }

    /// The adapter's description, copied into `arena`.
    pub fn adapter(d: Device, arena: Allocator) Allocator.Error!Adapter {
        switch (d) {
            .webgpu => |x| {
                var raw: dawn.c.WGPUAdapterInfo = undefined;
                const info = x.info(&raw);
                defer x.freeInfo(raw);
                return .{
                    .vendor = try arena.dupe(u8, info.vendor),
                    .architecture = try arena.dupe(u8, info.architecture),
                    .device = try arena.dupe(u8, info.device),
                    .description = try arena.dupe(u8, info.description),
                    .api = try arena.dupe(u8, info.backend),
                };
            },
            .velocity => |x| return .{
                .vendor = try std.fmt.allocPrint(arena, "0x{x:0>4}", .{x.caps.vendor_id}),
                .architecture = try std.fmt.allocPrint(arena, "sm_{d}_{d} linalg {d}", .{ x.caps.shader_model >> 4, x.caps.shader_model & 15, x.caps.linalg_tier }),
                .device = try arena.dupe(u8, x.adapterName()),
                .description = try std.fmt.allocPrint(arena, "0x{x:0>4}", .{x.caps.device_id}),
                .api = "d3d12",
            },
        }
    }

    /// The first error a backend reported, after error.GpuError or
    /// error.DeviceLost.
    pub fn errorMessage(d: Device) []const u8 {
        return switch (d) {
            inline else => |x| x.errorMessage(),
        };
    }

    /// Loses the device as a driver fault would (recovery tests).
    pub fn forceLoss(d: Device) void {
        switch (d) {
            inline else => |x| x.forceLoss(),
        }
    }
};

pub const Host = union(Backend) {
    webgpu: dawn_host.Host,
    velocity: velocity.Host,

    pub fn init(gpa: Allocator, d: Device) !Host {
        return switch (d) {
            .webgpu => |x| .{ .webgpu = .init(gpa, x) },
            .velocity => |x| .{ .velocity = try .init(gpa, x) },
        };
    }

    pub fn deinit(h: *Host) void {
        switch (h.*) {
            inline else => |*x| x.deinit(),
        }
    }

    /// Runs one batch, appending one completion per submit and per read.
    pub fn execute(h: *Host, words: []const u32, completions: *std.ArrayList(cmd.Completion)) Error!void {
        return switch (h.*) {
            inline else => |*x| x.execute(words, completions),
        };
    }

    /// The model file `write_file` reads: its mapping (`bytes`, Dawn), or
    /// its path (Velocity reads it with DirectStorage).
    pub fn setFile(h: *Host, gpa: Allocator, bytes: []const u8, path: []const u8) Error!void {
        switch (h.*) {
            .webgpu => |*x| x.file = bytes,
            .velocity => |*x| try x.setFile(gpa, path),
        }
    }

    /// Starts timing every dispatch (timings).
    pub fn startProfile(h: *Host) Error!void {
        return switch (h.*) {
            inline else => |*x| x.startProfile(),
        };
    }

    pub fn stopProfile(h: *Host) void {
        switch (h.*) {
            inline else => |*x| x.stopProfile(),
        }
    }

    /// Dispatch times since startProfile.
    pub fn timings(h: *Host) ?*const Timings {
        return switch (h.*) {
            .webgpu => |*x| if (x.profile) |*p| &p.timings else null,
            .velocity => |*x| if (x.timings) |*t| t else null,
        };
    }
};
