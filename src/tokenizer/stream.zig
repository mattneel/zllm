//! Incremental detokenization for streaming. Decoded bytes are released
//! only once they form complete UTF-8 and cannot be the start of a stop
//! string; a matched stop string ends the stream and is not part of the
//! output (vLLM's default, include_stop_str_in_output = false).

const std = @import("std");
const Allocator = std.mem.Allocator;
const Tokenizer = @import("tokenizer.zig").Tokenizer;

pub const Stream = struct {
    /// Decoded bytes not yet released.
    pending: std.ArrayList(u8) = .empty,
    stops: []const []const u8,
    stopped: bool = false,
    /// HF ByteFallback validates an entire consecutive byte-token run.
    fallback: std.ArrayList(u8) = .empty,
    decoded_started: bool = false,
    strip_initial: bool = false,

    pub fn deinit(s: *Stream, gpa: Allocator) void {
        s.pending.deinit(gpa);
        s.fallback.deinit(gpa);
        s.* = undefined;
    }

    /// Decodes `id` and appends the text it releases to `out`. Returns
    /// whether a stop string was completed (nothing more is released).
    pub fn push(s: *Stream, gpa: Allocator, tok: *const Tokenizer, id: u32, out: *std.ArrayList(u8)) Tokenizer.DecodeError!bool {
        s.strip_initial = tok.stripsInitialSpace();
        if (try tok.fallbackByte(id)) |byte| {
            try s.fallback.append(gpa, byte);
            return false;
        }
        try s.finishFallback(gpa);
        try tok.decodeRaw(gpa, &.{id}, &s.pending);
        s.stripInitial();
        return s.release(gpa, out);
    }

    /// As `push`, for bytes already decoded.
    pub fn pushBytes(s: *Stream, gpa: Allocator, bytes: []const u8, out: *std.ArrayList(u8)) Allocator.Error!bool {
        try s.pending.appendSlice(gpa, bytes);
        return s.release(gpa, out);
    }

    /// Releases everything still held (the generation ended without a stop).
    pub fn flush(s: *Stream, gpa: Allocator, out: *std.ArrayList(u8)) Allocator.Error!void {
        if (!s.stopped) {
            try s.finishFallback(gpa);
            _ = try s.release(gpa, out);
        }
        if (!s.stopped) try out.appendSlice(gpa, s.pending.items);
        s.pending.clearRetainingCapacity();
    }

    fn finishFallback(s: *Stream, gpa: Allocator) Allocator.Error!void {
        if (s.fallback.items.len == 0) return;
        if (std.unicode.utf8ValidateSlice(s.fallback.items)) {
            try s.pending.appendSlice(gpa, s.fallback.items);
        } else {
            const needed = std.math.mul(usize, s.fallback.items.len, 3) catch return error.OutOfMemory;
            try s.pending.ensureUnusedCapacity(gpa, needed);
            for (s.fallback.items) |_| s.pending.appendSliceAssumeCapacity("�");
        }
        s.fallback.clearRetainingCapacity();
        s.stripInitial();
    }

    fn stripInitial(s: *Stream) void {
        if (s.decoded_started or s.pending.items.len == 0) return;
        s.decoded_started = true;
        if (s.strip_initial and s.pending.items[0] == ' ') {
            std.mem.copyForwards(u8, s.pending.items[0 .. s.pending.items.len - 1], s.pending.items[1..]);
            s.pending.items.len -= 1;
        }
    }

    fn release(s: *Stream, gpa: Allocator, out: *std.ArrayList(u8)) Allocator.Error!bool {
        std.debug.assert(!s.stopped);
        const p = s.pending.items;
        var first: ?usize = null;
        for (s.stops) |stop| if (std.mem.indexOf(u8, p, stop)) |i| {
            if (first == null or i < first.?) first = i;
        };
        if (first) |i| {
            try out.appendSlice(gpa, p[0..i]);
            s.pending.clearRetainingCapacity();
            s.stopped = true;
            return true;
        }
        var cut = p.len - s.stopPrefix();
        cut = utf8Boundary(p, cut);
        try out.appendSlice(gpa, p[0..cut]);
        std.mem.copyForwards(u8, p[0 .. p.len - cut], p[cut..]);
        s.pending.shrinkRetainingCapacity(p.len - cut);
        return false;
    }

    /// Length of the longest suffix of the pending bytes that is a proper
    /// prefix of some stop string.
    fn stopPrefix(s: *const Stream) usize {
        const p = s.pending.items;
        var best: usize = 0;
        for (s.stops) |stop| {
            var n = @min(stop.len - 1, p.len);
            while (n > best) : (n -= 1) {
                if (std.mem.eql(u8, p[p.len - n ..], stop[0..n])) {
                    best = n;
                    break;
                }
            }
        }
        return best;
    }
};

/// The largest cut ≤ `cut` that does not split a UTF-8 sequence. Invalid
/// bytes are released as they are.
fn utf8Boundary(p: []const u8, cut: usize) usize {
    var j = cut;
    var back: usize = 0;
    while (j > 0 and back < 4) {
        j -= 1;
        back += 1;
        const b = p[j];
        if (b & 0xc0 == 0x80) continue;
        const len = std.unicode.utf8ByteSequenceLength(b) catch return cut;
        return if (j + len > cut) j else cut;
    }
    return cut;
}

test "incomplete UTF-8 is held until it completes" {
    const gpa = std.testing.allocator;
    var s: Stream = .{ .stops = &.{} };
    defer s.deinit(gpa);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const euro = "€"; // e2 82 ac
    _ = try s.pushBytes(gpa, "a" ++ euro[0..2], &out);
    try std.testing.expectEqualStrings("a", out.items);
    _ = try s.pushBytes(gpa, euro[2..], &out);
    try std.testing.expectEqualStrings("a€", out.items);
}

test "a possible stop prefix is held, then released or cut" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);

    var s: Stream = .{ .stops = &.{ "</end>", "STOP" } };
    defer s.deinit(gpa);
    try std.testing.expect(!try s.pushBytes(gpa, "hello </e", &out));
    try std.testing.expectEqualStrings("hello ", out.items);
    // Not a stop after all: released.
    try std.testing.expect(!try s.pushBytes(gpa, "x> ST", &out));
    try std.testing.expectEqualStrings("hello </ex> ", out.items);
    // The stop completes: the text before it is released, the stop is not.
    try std.testing.expect(try s.pushBytes(gpa, "OP tail", &out));
    try std.testing.expectEqualStrings("hello </ex> ", out.items);
    try s.flush(gpa, &out);
    try std.testing.expectEqualStrings("hello </ex> ", out.items);
}

test "flush releases held bytes when the generation ends" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var s: Stream = .{ .stops = &.{"STOP"} };
    defer s.deinit(gpa);
    _ = try s.pushBytes(gpa, "end ST", &out);
    try std.testing.expectEqualStrings("end ", out.items);
    try s.flush(gpa, &out);
    try std.testing.expectEqualStrings("end ST", out.items);
}
