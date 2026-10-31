//! Stands in for velocity/host.zig in builds without Velocity (-Dvelocity
//! off, or not Windows): its types exist for gpu/host.zig, but no device
//! ever opens (gpu.Device.open returns error.VelocityNotBuilt).

const std = @import("std");
const cmd = @import("../gpu/cmd.zig");
const Timings = @import("../gpu/timings.zig").Timings;

pub const Error = error{};

pub const Device = struct {
    caps: struct { vendor_id: u32, device_id: u32, shader_model: u32, linalg_tier: u32 },

    pub fn init(_: std.mem.Allocator, _: u32) error{VelocityNotBuilt}!*Device {
        return error.VelocityNotBuilt;
    }
    pub fn deinit(_: *Device, _: std.mem.Allocator) void {
        unreachable;
    }
    pub fn adapterName(_: *const Device) []const u8 {
        unreachable;
    }
    pub fn errorMessage(_: *const Device) []const u8 {
        unreachable;
    }
    pub fn forceLoss(_: *Device) void {
        unreachable;
    }
};

pub const Host = struct {
    timings: ?Timings,

    pub fn init(_: std.mem.Allocator, _: *Device) Error!Host {
        unreachable;
    }
    pub fn deinit(_: *Host) void {
        unreachable;
    }
    pub fn execute(_: *Host, _: []const u32, _: *std.ArrayList(cmd.Completion)) Error!void {
        unreachable;
    }
    pub fn setFile(_: *Host, _: std.mem.Allocator, _: []const u8) Error!void {
        unreachable;
    }
    pub fn startProfile(_: *Host) Error!void {
        unreachable;
    }
    pub fn stopProfile(_: *Host) void {
        unreachable;
    }
};
