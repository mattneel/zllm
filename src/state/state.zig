//! Which tokens the GPU state holds, where the DeltaNet checkpoints are, and
//! how to reuse the longest possible prefix for a new prompt.
//!
//! The resident sequence is the tokens the GPU has processed: the KV cache
//! holds their rows and the recurrent state is the state after the last one.
//! Attention rows can be rewound by truncating the sequence (later rows are
//! never read), but the DeltaNet state cannot, so the engine saves it to
//! checkpoint slots: just before each prompt's last token (so the same
//! prompt can be resumed one token short of its end, to get its logits) and
//! every `checkpoint_interval` tokens.
//! A new prompt keeps the longest common prefix by either continuing from
//! the current state (the resident sequence is a prefix of the prompt) or
//! restoring the latest checkpoint inside the common prefix.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Positions the engine checkpoints at, besides prompt ends.
pub const checkpoint_interval = 512;

pub const Checkpoint = struct {
    /// Tokens the saved state covers; null while the slot is empty.
    pos: ?u32 = null,
    /// When it was saved, to evict the oldest.
    serial: u64 = 0,
};

pub const Sequence = struct {
    tokens: std.ArrayList(u32) = .empty,
    slots: []Checkpoint,
    serial: u64 = 0,

    pub fn init(gpa: Allocator, slots: u32) Allocator.Error!Sequence {
        const s = try gpa.alloc(Checkpoint, slots);
        @memset(s, .{});
        return .{ .slots = s };
    }

    pub fn deinit(s: *Sequence, gpa: Allocator) void {
        s.tokens.deinit(gpa);
        gpa.free(s.slots);
        s.* = undefined;
    }

    pub const Resume = struct {
        /// The state to start from.
        from: union(enum) { current, checkpoint: u32, reset },
        /// Tokens of the prompt already covered; prefill the rest.
        start: u32,
    };

    /// How to bring the state to the longest reusable prefix of `prompt`,
    /// leaving at least one token to process (its logits seed sampling).
    pub fn plan(s: *const Sequence, prompt: []const u32) Resume {
        std.debug.assert(prompt.len > 0);
        // An empty sequence says nothing about what the state buffers hold.
        if (s.tokens.items.len == 0) return .{ .from = .reset, .start = 0 };
        const common = std.mem.indexOfDiff(u32, s.tokens.items, prompt) orelse prompt.len;
        const limit: u32 = @intCast(@min(common, prompt.len - 1));
        if (s.tokens.items.len <= limit) return .{ .from = .current, .start = @intCast(s.tokens.items.len) };
        var best: ?u32 = null;
        for (s.slots, 0..) |cp, i| {
            const pos = cp.pos orelse continue;
            if (pos > limit) continue;
            if (best == null or pos > s.slots[best.?].pos.?) best = @intCast(i);
        }
        if (best) |slot| return .{ .from = .{ .checkpoint = slot }, .start = s.slots[slot].pos.? };
        return .{ .from = .reset, .start = 0 };
    }

    /// Records that the GPU state now covers the first `r.start` tokens
    /// (after the engine has restored or reset it as `r` says). Checkpoints
    /// past that point describe a sequence that is being replaced.
    pub fn rewind(s: *Sequence, r: Resume) void {
        s.tokens.shrinkRetainingCapacity(r.start);
        for (s.slots) |*cp| if (cp.pos) |pos| if (pos > r.start) {
            cp.* = .{};
        };
    }

    pub fn append(s: *Sequence, gpa: Allocator, tokens: []const u32) Allocator.Error!void {
        try s.tokens.appendSlice(gpa, tokens);
    }

    pub fn len(s: *const Sequence) u32 {
        return @intCast(s.tokens.items.len);
    }

    /// The slot to save the current state to, or null if a checkpoint at
    /// this position already exists. Takes an empty slot, else the oldest.
    pub fn checkpointSlot(s: *Sequence) ?u32 {
        const pos = s.len();
        var pick: u32 = 0;
        for (s.slots, 0..) |cp, i| {
            if (cp.pos == pos) return null;
            const p = s.slots[pick];
            if (p.pos == null) continue;
            if (cp.pos == null or cp.serial < p.serial) pick = @intCast(i);
        }
        s.serial += 1;
        s.slots[pick] = .{ .pos = pos, .serial = s.serial };
        return pick;
    }
};

test "a prompt that extends the sequence continues from the current state" {
    const gpa = std.testing.allocator;
    var s: Sequence = try .init(gpa, 2);
    defer s.deinit(gpa);
    try s.append(gpa, &.{ 1, 2, 3 });
    const r = s.plan(&.{ 1, 2, 3, 4, 5 });
    try std.testing.expectEqual(Sequence.Resume{ .from = .current, .start = 3 }, r);
}

test "an identical prompt replays its last token from a checkpoint" {
    const gpa = std.testing.allocator;
    var s: Sequence = try .init(gpa, 2);
    defer s.deinit(gpa);
    try s.append(gpa, &.{ 1, 2 });
    try std.testing.expectEqual(@as(?u32, 0), s.checkpointSlot());
    try s.append(gpa, &.{3});
    try std.testing.expectEqual(@as(?u32, 1), s.checkpointSlot());
    // The whole prompt is resident, but one token must be processed for its
    // logits: the checkpoint at 3 is past the limit of 2, so restore 2.
    const r = s.plan(&.{ 1, 2, 3 });
    try std.testing.expectEqual(Sequence.Resume{ .from = .{ .checkpoint = 0 }, .start = 2 }, r);
    s.rewind(r);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, s.tokens.items);
    try std.testing.expectEqual(@as(?u32, null), s.slots[1].pos);
}

test "a diverging prompt restores the latest checkpoint inside the common prefix" {
    const gpa = std.testing.allocator;
    var s: Sequence = try .init(gpa, 3);
    defer s.deinit(gpa);
    try s.append(gpa, &.{ 1, 2 });
    _ = s.checkpointSlot();
    try s.append(gpa, &.{ 3, 4 });
    _ = s.checkpointSlot();
    try s.append(gpa, &.{ 5, 6 });
    _ = s.checkpointSlot();
    // Common prefix 5 tokens: checkpoints at 2 and 4 qualify, 6 does not.
    const r = s.plan(&.{ 1, 2, 3, 4, 5, 9 });
    try std.testing.expectEqual(Sequence.Resume{ .from = .{ .checkpoint = 1 }, .start = 4 }, r);
    // Nothing in common: start over.
    try std.testing.expectEqual(Sequence.Resume{ .from = .reset, .start = 0 }, s.plan(&.{ 7, 8 }));
}

test "an empty sequence always resets the state" {
    const gpa = std.testing.allocator;
    var s: Sequence = try .init(gpa, 1);
    defer s.deinit(gpa);
    try std.testing.expectEqual(Sequence.Resume{ .from = .reset, .start = 0 }, s.plan(&.{ 1, 2 }));
}

test "checkpoint slots fill empty ones first, then evict the oldest" {
    const gpa = std.testing.allocator;
    var s: Sequence = try .init(gpa, 2);
    defer s.deinit(gpa);
    try s.append(gpa, &.{1});
    try std.testing.expectEqual(@as(?u32, 0), s.checkpointSlot());
    try std.testing.expectEqual(@as(?u32, null), s.checkpointSlot());
    try s.append(gpa, &.{2});
    try std.testing.expectEqual(@as(?u32, 1), s.checkpointSlot());
    try s.append(gpa, &.{3});
    try std.testing.expectEqual(@as(?u32, 0), s.checkpointSlot());
    try std.testing.expectEqual(@as(?u32, 3), s.slots[0].pos);
    try std.testing.expectEqual(@as(?u32, 2), s.slots[1].pos);
}
