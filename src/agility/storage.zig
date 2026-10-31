//! Consecutive disjoint file reads for one GPU buffer share a DirectStorage
//! submission. Keep half the 1,024-entry queue free for the previous batch,
//! its status and fence requests. No allocation or sorting on the step path.
const std = @import("std");

pub const capacity = 510;

/// The C ABI's vel_range, checked against it by the native host.
pub const Range = extern struct {
    file_offset: u64,
    buffer_offset: u64,
    bytes: u32,
    reserved: u32 = 0,
};

pub const Reads = struct {
    buffer: u32 = 0,
    ranges: [capacity]Range = undefined,
    count: usize = 0,

    /// A reversal or overlap starts a new batch, so later writes can wait
    /// for the previous storage ticket before they overwrite its bytes.
    pub fn orderedAfter(r: *const Reads, buffer: u32, offset: u64) bool {
        if (r.count == 0 or buffer != r.buffer) return true;
        const last = r.ranges[r.count - 1];
        return offset >= last.buffer_offset + last.bytes;
    }

    pub fn fits(r: *const Reads, buffer: u32, range: Range) bool {
        return r.count == 0 or (r.buffer == buffer and r.count < capacity and r.orderedAfter(buffer, range.buffer_offset));
    }

    pub fn append(r: *Reads, buffer: u32, range: Range) void {
        std.debug.assert(r.fits(buffer, range));
        r.buffer = buffer;
        r.ranges[r.count] = range;
        r.count += 1;
    }

    pub fn clear(r: *Reads) void {
        r.count = 0;
    }
};

test "file read batches preserve bytes and stop at the queue capacity" {
    var reads: Reads = .{};
    for (0..capacity) |i| {
        const range: Range = .{ .file_offset = (capacity - i) * 512, .buffer_offset = i * 272, .bytes = 272 };
        try std.testing.expect(reads.fits(7, range));
        reads.append(7, range);
        try std.testing.expectEqual(range, reads.ranges[i]);
    }
    const next: Range = .{ .file_offset = 0, .buffer_offset = capacity * 272, .bytes = 272 };
    try std.testing.expect(!reads.fits(7, next));
    reads.clear();
    try std.testing.expect(reads.fits(9, next));
    reads.append(9, next);
    try std.testing.expectEqual(@as(usize, 1), reads.count);
    try std.testing.expectEqual(@as(u32, 9), reads.buffer);
}

test "file read batching rejects buffer, overlap and reversal mutations" {
    var reads: Reads = .{};
    const first: Range = .{ .file_offset = 80, .buffer_offset = 16, .bytes = 32 };
    reads.append(3, first);
    const adjacent: Range = .{ .file_offset = 0, .buffer_offset = 48, .bytes = 16 };
    try std.testing.expect(reads.fits(3, adjacent));
    try std.testing.expect(!reads.fits(4, adjacent));
    var mutated = adjacent;
    mutated.buffer_offset = 47;
    try std.testing.expect(!reads.fits(3, mutated));
    try std.testing.expect(!reads.orderedAfter(3, mutated.buffer_offset));
    mutated.buffer_offset = 0;
    try std.testing.expect(!reads.fits(3, mutated));
    try std.testing.expect(!reads.orderedAfter(3, mutated.buffer_offset));
    reads.append(3, adjacent);
    try std.testing.expectEqual(first, reads.ranges[0]);
    try std.testing.expectEqual(adjacent, reads.ranges[1]);
}

test "a 64-token PLE step uses three submissions without dropping rows" {
    var reads: Reads = .{};
    var batches: usize = 0;
    var rows: usize = 0;
    for (0..64 * 16) |i| {
        const range: Range = .{ .file_offset = ((i * 997) % 4096) * 272, .buffer_offset = i * 272, .bytes = 272 };
        if (!reads.fits(1, range)) {
            batches += 1;
            rows += reads.count;
            reads.clear();
        }
        reads.append(1, range);
    }
    batches += 1;
    rows += reads.count;
    try std.testing.expectEqual(@as(usize, 3), batches);
    try std.testing.expectEqual(@as(usize, 1024), rows);
}
