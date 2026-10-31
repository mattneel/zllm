//! The ordering Agility's host keeps between operations in a command list:
//! a UAV barrier only where an operation touches a buffer range one since
//! the last barrier wrote, or writes a range one of them read.

const std = @import("std");

/// A buffer range an operation reads or writes.
pub const Access = struct {
    buffer: u32,
    lo: u64,
    hi: u64,
    write: bool,

    pub fn conflicts(a: Access, b: Access) bool {
        return a.buffer == b.buffer and a.lo < b.hi and b.lo < a.hi and (a.write or b.write);
    }
};

/// The accesses since the last UAV barrier.
pub const Hazards = struct {
    since: std.ArrayList(Access) = .empty,

    /// Whether `ops` must wait for what ran since the last barrier.
    pub fn conflict(h: *const Hazards, ops: []const Access) bool {
        for (ops) |a| for (h.since.items) |b| if (a.conflicts(b)) return true;
        return false;
    }

    pub fn reset(h: *Hazards) void {
        h.since.clearRetainingCapacity();
    }
};

test "operations conflict only on overlapping ranges one of them writes" {
    const a: Access = .{ .buffer = 1, .lo = 0, .hi = 64, .write = true };
    try std.testing.expect(a.conflicts(.{ .buffer = 1, .lo = 32, .hi = 96, .write = false }));
    try std.testing.expect(!a.conflicts(.{ .buffer = 1, .lo = 64, .hi = 96, .write = true }));
    try std.testing.expect(!a.conflicts(.{ .buffer = 2, .lo = 0, .hi = 64, .write = true }));
    const r: Access = .{ .buffer = 1, .lo = 0, .hi = 64, .write = false };
    try std.testing.expect(!r.conflicts(.{ .buffer = 1, .lo = 0, .hi = 64, .write = false }));
}
