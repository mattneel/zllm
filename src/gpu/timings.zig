//! Per-dispatch GPU times while a host profiles (Host.startProfile), the
//! same on every backend: totals per pipeline and grid.

const std = @import("std");

pub const Timings = struct {
    /// Nanoseconds and dispatches per pipeline and grid.
    totals: std.AutoArrayHashMapUnmanaged(Key, Total) = .empty,
    /// Dispatches past `capacity` in one submit, not timed.
    untimed: u64 = 0,
    /// Timed dispatches per submit.
    capacity: u64,

    pub const Key = struct { pipeline: u32, grid: [3]u32 };
    pub const Total = struct { ns: u64 = 0, count: u64 = 0 };

    pub fn deinit(t: *Timings, gpa: std.mem.Allocator) void {
        t.totals.deinit(gpa);
        t.* = undefined;
    }

    pub fn add(t: *Timings, gpa: std.mem.Allocator, key: Key, ns: u64) std.mem.Allocator.Error!void {
        const e = try t.totals.getOrPutValue(gpa, key, .{});
        e.value_ptr.ns += ns;
        e.value_ptr.count += 1;
    }
};
