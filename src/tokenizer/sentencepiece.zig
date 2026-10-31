//! SentencePiece BPE expressed as the exact Hugging Face rank-pair graph.
//! A scored GGUF vocabulary contains enough information to reconstruct the
//! converter's merge order: descending score, left scalar count, right scalar
//! count, then stable vocabulary/pair-id order. Runtime merges use bpe.zig.
const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const bpe = @import("bpe.zig");
const metaspace = @import("gemma.zig");

pub const Id = u32;
pub const LoadError = metaspace.LoadError;
pub const Model = struct {
    base: metaspace.Model,
    add_prefix: bool,

    pub fn deinit(model: *Model, gpa: Allocator) void {
        model.base.deinit(gpa);
        model.* = undefined;
    }

    /// Metaspace's `first` scheme refers to the original input offset, not the
    /// first ordinary fragment remaining after raw added-token extraction.
    pub fn encode(model: *const Model, gpa: Allocator, raw: []const u8, first: bool, out: *std.ArrayList(Id)) metaspace.EncodeError!void {
        try model.base.encodeWithPrefix(gpa, raw, first and model.add_prefix, out);
    }

    pub fn decode(model: *const Model, gpa: Allocator, token_ids: []const Id, out: *std.ArrayList(u8)) metaspace.DecodeError!void {
        const start = out.items.len;
        try model.base.decode(gpa, token_ids, out);
        // HF Strip(start=1) follows Fuse, so this applies once to the whole
        // decoded stream. A leading BOS literal prevents subsequent stripping.
        if (out.items.len > start and out.items[start] == ' ') {
            std.mem.copyForwards(u8, out.items[start .. out.items.len - 1], out.items[start + 1 ..]);
            out.items.len -= 1;
        }
    }

    pub fn decodeRaw(model: *const Model, gpa: Allocator, token_ids: []const Id, out: *std.ArrayList(u8)) metaspace.DecodeError!void {
        try model.base.decodeRaw(gpa, token_ids, out);
    }
};

pub fn init(gpa: Allocator, file: gguf.File, tokens: []const []const u8, ids: *const std.StringHashMapUnmanaged(Id)) LoadError!Model {
    const scores_value = file.get("tokenizer.ggml.scores") orelse return error.MissingMetadata;
    const types_value = file.get("tokenizer.ggml.token_type") orelse return error.MissingMetadata;
    if (scores_value != .array or scores_value.array.elem_type != .f32 or types_value != .array or types_value.array.elem_type != .i32) return error.BadMetadata;
    if (scores_value.array.len != tokens.len or types_value.array.len != tokens.len) return error.BadMetadata;
    var prefix = true;
    if (file.get("tokenizer.ggml.add_space_prefix")) |value| {
        if (value != .bool) return error.BadMetadata;
        prefix = value.bool;
    }
    if (file.get("tokenizer.ggml.remove_extra_whitespaces")) |value| {
        if (value != .bool) return error.BadMetadata;
        if (value.bool) return error.UnsupportedTokenizer;
    }
    const merges = try buildMerges(gpa, tokens, ids, scores_value.array, types_value.array);
    errdefer {
        var owned = merges;
        owned.deinit(gpa);
    }
    var byte_tokens: [256]Id = undefined;
    for (&byte_tokens, 0..) |*id, byte| {
        var name: [6]u8 = undefined;
        const text = std.fmt.bufPrint(&name, "<0x{X:0>2}>", .{byte}) catch unreachable;
        id.* = ids.get(text) orelse return error.BadMetadata;
    }
    const space = ids.get("▁") orelse return error.BadMetadata;
    var ascii_tokens: [128]Id = undefined;
    for (&ascii_tokens, 0..) |*id, byte| {
        const name = [1]u8{@intCast(byte)};
        id.* = if (byte == ' ') space else ids.get(&name) orelse byte_tokens[byte];
    }
    return .{ .base = .{
        .tokens = tokens,
        .ids = ids.*,
        .merges = merges,
        .byte_token = byte_tokens,
        .ascii_token = ascii_tokens,
    }, .add_prefix = prefix };
}

const Candidate = struct {
    score: f32,
    left_len: usize,
    right_len: usize,
    output: Id,
    left: Id,
    right: Id,

    fn less(_: void, left: Candidate, right: Candidate) bool {
        if (left.score != right.score) return left.score > right.score;
        if (left.left_len != right.left_len) return left.left_len > right.left_len;
        if (left.right_len != right.right_len) return left.right_len > right.right_len;
        // Equivalent to stable global sorting after per-token (left,right)
        // id sorting; no dependence on hash iteration or byte/UTF-8 length.
        if (left.output != right.output) return left.output < right.output;
        if (left.left != right.left) return left.left < right.left;
        return left.right < right.right;
    }
};

fn buildMerges(gpa: Allocator, tokens: []const []const u8, ids: *const std.StringHashMapUnmanaged(Id), scores: gguf.Array, types: gguf.Array) LoadError!bpe.MergeTable {
    var candidates: std.ArrayList(Candidate) = .empty;
    defer candidates.deinit(gpa);
    for (tokens, 0..) |token, output| {
        if (types.get(u32, output) == 5) continue; // GGUF vocabulary padding.
        const score: f32 = @bitCast(scores.get(u32, output));
        if (!std.math.isFinite(score)) return error.BadMetadata;
        const chars = std.unicode.utf8CountCodepoints(token) catch return error.BadMetadata;
        if (chars < 2) continue;
        var split: usize = std.unicode.utf8ByteSequenceLength(token[0]) catch return error.BadMetadata;
        var left_len: usize = 1;
        while (split < token.len) {
            if (ids.get(token[0..split])) |left| {
                if (ids.get(token[split..])) |right| {
                    if (types.get(u32, left) != 5 and types.get(u32, right) != 5) try candidates.append(gpa, .{
                        .score = score,
                        .left_len = left_len,
                        .right_len = chars - left_len,
                        .output = @intCast(output),
                        .left = left,
                        .right = right,
                    });
                }
            }
            split += std.unicode.utf8ByteSequenceLength(token[split]) catch return error.BadMetadata;
            left_len += 1;
        }
    }
    if (candidates.items.len >= std.math.maxInt(u32)) return error.BadMetadata;
    std.mem.sort(Candidate, candidates.items, {}, Candidate.less);
    var table = try bpe.MergeTable.init(gpa, candidates.items.len);
    for (candidates.items, 0..) |candidate, rank| table.putFirst(candidate.left, candidate.right, .{ .rank = @intCast(rank), .id = candidate.output });
    return table;
}

test "score conversion preserves scalar-length and stable tie order" {
    const gpa = std.testing.allocator;
    // Unicode length must win, not UTF-8 byte length: 'aa' has two scalars,
    // while '中' has one despite occupying more bytes.
    const tokens = [_][]const u8{ "a", "b", "aa", "ab", "aab", "中", "中b", "unused" };
    const scores = [_]f32{ -10, -10, -5, -5, -5, -10, -5, 100 };
    const types = [_]u32{ 1, 1, 1, 1, 1, 1, 1, 5 };
    var ids: std.StringHashMapUnmanaged(Id) = .empty;
    defer ids.deinit(gpa);
    for (tokens, 0..) |text, index| try ids.put(gpa, text, @intCast(index));
    var table = try buildMerges(gpa, &tokens, &ids, .{ .elem_type = .f32, .len = scores.len, .bytes = std.mem.sliceAsBytes(&scores) }, .{ .elem_type = .i32, .len = types.len, .bytes = std.mem.sliceAsBytes(&types) });
    defer table.deinit(gpa);
    try std.testing.expectEqual(bpe.Merge{ .rank = 0, .id = 4 }, table.get(2, 1).?); // aa + b
    try std.testing.expectEqual(bpe.Merge{ .rank = 1, .id = 4 }, table.get(0, 3).?); // a + ab
    try std.testing.expectEqual(bpe.Merge{ .rank = 2, .id = 2 }, table.get(0, 0).?);
    try std.testing.expectEqual(bpe.Merge{ .rank = 3, .id = 3 }, table.get(0, 1).?);
    try std.testing.expectEqual(bpe.Merge{ .rank = 4, .id = 6 }, table.get(5, 1).?);
    // Mutation: score changes must change ranks despite identical vocabulary.
    var changed = scores;
    changed[6] = 0;
    var mutated = try buildMerges(gpa, &tokens, &ids, .{ .elem_type = .f32, .len = changed.len, .bytes = std.mem.sliceAsBytes(&changed) }, .{ .elem_type = .i32, .len = types.len, .bytes = std.mem.sliceAsBytes(&types) });
    defer mutated.deinit(gpa);
    try std.testing.expectEqual(bpe.Merge{ .rank = 0, .id = 6 }, mutated.get(5, 1).?);
}

test "score conversion rejects nonfinite metadata" {
    const gpa = std.testing.allocator;
    const tokens = [_][]const u8{"a"};
    const scores = [_]f32{std.math.nan(f32)};
    const types = [_]u32{1};
    var ids: std.StringHashMapUnmanaged(Id) = .empty;
    defer ids.deinit(gpa);
    try ids.put(gpa, "a", 0);
    try std.testing.expectError(error.BadMetadata, buildMerges(gpa, &tokens, &ids, .{ .elem_type = .f32, .len = 1, .bytes = std.mem.sliceAsBytes(&scores) }, .{ .elem_type = .i32, .len = 1, .bytes = std.mem.sliceAsBytes(&types) }));
}
