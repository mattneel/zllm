//! Portable rank-pair BPE: flat immutable merge lookup, linked symbols and
//! cached candidates. Short pieces scan cached ranks; long pieces use a heap.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Id = u32;
pub const Merge = struct { rank: u32, id: Id };
const absent = std.math.maxInt(u32);
const end = std.math.maxInt(usize);

pub const MergeTable = struct {
    entries: []Entry,
    const Entry = struct { key: u64 = 0, rank: u32 = absent, id: Id = 0 };

    pub fn init(gpa: Allocator, count: usize) Allocator.Error!MergeTable {
        const doubled = std.math.mul(usize, count, 2) catch return error.OutOfMemory;
        const capacity = std.math.ceilPowerOfTwo(usize, @max(2, doubled)) catch return error.OutOfMemory;
        const entries = try gpa.alloc(Entry, capacity);
        @memset(entries, .{});
        return .{ .entries = entries };
    }

    pub fn deinit(table: *MergeTable, gpa: Allocator) void {
        gpa.free(table.entries);
        table.* = undefined;
    }

    fn slot(table: *const MergeTable, key: u64) usize {
        // Integer avalanche; independent of pointers, host endian and SIMD ISA.
        var hash = key;
        hash = (hash ^ (hash >> 30)) *% 0xbf58476d1ce4e5b9;
        hash = (hash ^ (hash >> 27)) *% 0x94d049bb133111eb;
        hash ^= hash >> 31;
        return @as(usize, @truncate(hash)) & (table.entries.len - 1);
    }

    pub fn putFirst(table: *MergeTable, left: Id, right: Id, merge: Merge) void {
        const key = pairKey(left, right);
        var index = table.slot(key);
        while (table.entries[index].rank != absent) : (index = (index + 1) & (table.entries.len - 1)) {
            if (table.entries[index].key == key) return;
        }
        table.entries[index] = .{ .key = key, .rank = merge.rank, .id = merge.id };
    }

    pub fn get(table: *const MergeTable, left: Id, right: Id) ?Merge {
        const key = pairKey(left, right);
        var index = table.slot(key);
        while (table.entries[index].rank != absent) : (index = (index + 1) & (table.entries.len - 1)) {
            const entry = table.entries[index];
            if (entry.key == key) return .{ .rank = entry.rank, .id = entry.id };
        }
        return null;
    }
};

fn pairKey(left: Id, right: Id) u64 {
    return (@as(u64, left) << 32) | right;
}

pub const Scratch = struct {
    nodes: std.ArrayList(Node) = .empty,
    heap: std.ArrayList(Candidate) = .empty,
    // Most regex pieces are short. A scan avoids heap maintenance for these,
    // while still only hashing pairs when an adjacent symbol changes.
    const heap_threshold = 32;
    const Node = struct {
        id: Id,
        prev: usize,
        next: usize,
        merge: Merge = .{ .rank = absent, .id = 0 },
        generation: usize = 0,
        active: bool = true,
    };
    const Candidate = struct {
        rank: u32,
        index: usize,
        generation: usize,

        fn less(left: Candidate, right: Candidate) bool {
            return left.rank < right.rank or (left.rank == right.rank and left.index < right.index);
        }
    };

    pub fn deinit(scratch: *Scratch, gpa: Allocator) void {
        scratch.nodes.deinit(gpa);
        scratch.heap.deinit(gpa);
        scratch.* = undefined;
    }

    pub fn encode(scratch: *Scratch, gpa: Allocator, table: *const MergeTable, byte_tokens: *const [256]Id, piece: []const u8, out: *std.ArrayList(Id)) Allocator.Error!void {
        if (piece.len == 0) return;
        if (piece.len == 1) return out.append(gpa, byte_tokens[piece[0]]);
        const use_heap = piece.len > heap_threshold;
        var local: [heap_threshold]Node = undefined;
        const nodes = if (use_heap) blk: {
            try scratch.nodes.resize(gpa, piece.len);
            break :blk scratch.nodes.items;
        } else local[0..piece.len];
        scratch.heap.clearRetainingCapacity();
        for (piece, 0..) |byte, index| nodes[index] = .{
            .id = byte_tokens[byte],
            .prev = if (index == 0) end else index - 1,
            .next = if (index + 1 == piece.len) end else index + 1,
        };
        if (use_heap) try scratch.heap.ensureTotalCapacity(gpa, piece.len);
        for (0..piece.len - 1) |index| {
            try scratch.refresh(gpa, table, nodes, index, false);
            if (use_heap and nodes[index].merge.rank != absent) scratch.heap.appendAssumeCapacity(.{
                .rank = nodes[index].merge.rank,
                .index = index,
                .generation = nodes[index].generation,
            });
        }
        var parent = scratch.heap.items.len / 2;
        while (parent > 0) {
            parent -= 1;
            scratch.siftDown(parent, scratch.heap.items[parent]);
        }
        var remaining = piece.len;
        while (if (use_heap) scratch.popValid(nodes) else minimum(nodes)) |index| {
            const right = nodes[index].next;
            const following = nodes[right].next;
            nodes[index].id = nodes[index].merge.id;
            nodes[index].next = following;
            nodes[right].active = false;
            remaining -= 1;
            if (following != end) nodes[following].prev = index;
            try scratch.refresh(gpa, table, nodes, index, use_heap);
            if (nodes[index].prev != end) try scratch.refresh(gpa, table, nodes, nodes[index].prev, use_heap);
        }
        try out.ensureUnusedCapacity(gpa, remaining);
        var index: usize = 0;
        while (index != end) : (index = nodes[index].next) out.appendAssumeCapacity(nodes[index].id);
    }

    fn refresh(scratch: *Scratch, gpa: Allocator, table: *const MergeTable, nodes: []Node, index: usize, use_heap: bool) Allocator.Error!void {
        const node = &nodes[index];
        node.generation += 1;
        node.merge = if (node.next == end) .{ .rank = absent, .id = 0 } else table.get(node.id, nodes[node.next].id) orelse .{ .rank = absent, .id = 0 };
        if (use_heap and node.merge.rank != absent) try scratch.push(gpa, .{ .rank = node.merge.rank, .index = index, .generation = node.generation });
    }

    fn minimum(nodes: []const Node) ?usize {
        var best: ?usize = null;
        var rank: u32 = absent;
        for (nodes, 0..) |node, index| {
            if (node.active and node.merge.rank < rank) {
                best = index;
                rank = node.merge.rank;
            }
        }
        return best;
    }

    fn push(scratch: *Scratch, gpa: Allocator, candidate: Candidate) Allocator.Error!void {
        try scratch.heap.append(gpa, candidate);
        var index = scratch.heap.items.len - 1;
        while (index > 0) {
            const parent = (index - 1) / 2;
            if (!Candidate.less(candidate, scratch.heap.items[parent])) break;
            scratch.heap.items[index] = scratch.heap.items[parent];
            index = parent;
        }
        scratch.heap.items[index] = candidate;
    }

    fn siftDown(scratch: *Scratch, start: usize, candidate: Candidate) void {
        var index = start;
        while (index * 2 + 1 < scratch.heap.items.len) {
            var child = index * 2 + 1;
            if (child + 1 < scratch.heap.items.len and Candidate.less(scratch.heap.items[child + 1], scratch.heap.items[child])) child += 1;
            if (!Candidate.less(scratch.heap.items[child], candidate)) break;
            scratch.heap.items[index] = scratch.heap.items[child];
            index = child;
        }
        scratch.heap.items[index] = candidate;
    }

    fn popValid(scratch: *Scratch, nodes: []const Node) ?usize {
        while (scratch.heap.items.len != 0) {
            const candidate = scratch.heap.items[0];
            const last = scratch.heap.pop().?;
            if (scratch.heap.items.len != 0) scratch.siftDown(0, last);
            const node = nodes[candidate.index];
            if (node.active and node.generation == candidate.generation) return candidate.index;
        }
        return null;
    }
};

test "flat pair table retains first rank and handles collisions" {
    const gpa = std.testing.allocator;
    var table = try MergeTable.init(gpa, 256);
    defer table.deinit(gpa);
    for (0..256) |index| table.putFirst(@intCast(index), @intCast(index + 1), .{ .rank = @intCast(index), .id = @intCast(index + 256) });
    table.putFirst(0, 1, .{ .rank = 999, .id = 999 });
    for (0..256) |index| try std.testing.expectEqual(Merge{ .rank = @intCast(index), .id = @intCast(index + 256) }, table.get(@intCast(index), @intCast(index + 1)).?);
    try std.testing.expectEqual(@as(?Merge, null), table.get(900, 901));
    try std.testing.expectError(error.OutOfMemory, MergeTable.init(gpa, std.math.maxInt(usize)));
}

test "cached merge schedulers match exhaustive rank leftmost oracle" {
    const gpa = std.testing.allocator;
    var table = try MergeTable.init(gpa, 128);
    defer table.deinit(gpa);
    // Include repeated ranks deliberately: ties are leftmost, not token-id order.
    for (0..8) |left| for (0..8) |right| {
        table.putFirst(@intCast(left), @intCast(right), .{ .rank = @intCast((left * 7 + right * 3) % 11), .id = @intCast(8 + (left + right) % 8) });
    };
    for (0..8) |left| for (0..8) |right| {
        table.putFirst(@intCast(left + 8), @intCast(right), .{ .rank = @intCast((left + right) % 5), .id = @intCast(16 + left) });
    };
    var byte_tokens: [256]Id = undefined;
    for (&byte_tokens, 0..) |*id, index| id.* = @intCast(index % 8);
    var scratch: Scratch = .{};
    defer scratch.deinit(gpa);
    var actual: std.ArrayList(Id) = .empty;
    defer actual.deinit(gpa);
    var expected: std.ArrayList(Id) = .empty;
    defer expected.deinit(gpa);
    var state: u64 = 42;
    var bytes: [257]u8 = undefined;
    for (0..1000) |trial| {
        const len = trial % bytes.len;
        for (bytes[0..len]) |*byte| {
            state = state *% 6364136223846793005 +% 1;
            byte.* = @truncate(state >> 32);
        }
        actual.clearRetainingCapacity();
        expected.clearRetainingCapacity();
        for (bytes[0..len]) |byte| try expected.append(gpa, byte_tokens[byte]);
        while (expected.items.len > 1) {
            var best: ?usize = null;
            var merge: Merge = .{ .rank = absent, .id = 0 };
            for (0..expected.items.len - 1) |index| {
                const next = table.get(expected.items[index], expected.items[index + 1]) orelse continue;
                if (next.rank < merge.rank) {
                    best = index;
                    merge = next;
                }
            }
            const index = best orelse break;
            expected.items[index] = merge.id;
            _ = expected.orderedRemove(index + 1);
        }
        try scratch.encode(gpa, &table, &byte_tokens, bytes[0..len], &actual);
        try std.testing.expectEqualSlices(Id, expected.items, actual.items);
    }
}
