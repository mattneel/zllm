//! Gemma 4's Hugging Face BPE, not SentencePiece's score-based BPE.
//!
//! Normalization replaces ASCII space with U+2581 only. The following HF
//! literal-space pre-tokenizer is therefore inert: each ordinary raw segment
//! is one BPE word. There is no NFC, dummy prefix, whitespace collapse, or
//! initial-space stripping, including at boundaries around raw added tokens.
//! Explicit GGUF merges supply ranks; scores and whole-word vocab hits do not.
//!
//! Decoder order matters: Replace(U+2581, space), ByteFallback, Fuse. A byte
//! run producing U+2581 must stay U+2581, and an invalid run yields one U+FFFD
//! per byte. `decodeRaw` instead leaves fallback bytes unvalidated for streaming
//! UTF-8 assembly; exact HF decoding of malformed runs requires the whole run.
const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const bpe = @import("bpe.zig");

pub const Id = u32;
pub const LoadError = error{ MissingMetadata, BadMetadata, UnknownMergeToken, UnsupportedTokenizer } || Allocator.Error;
pub const EncodeError = error{InvalidUtf8} || Allocator.Error;
pub const DecodeError = error{UnknownToken} || Allocator.Error;
const space_marker = "▁";

pub const Model = struct {
    tokens: []const []const u8,
    // A borrowed immutable map handle, not a pointer to the caller's movable
    // Tokenizer struct. Only the caller frees its entries and token strings.
    ids: std.StringHashMapUnmanaged(Id),
    merges: bpe.MergeTable,
    byte_token: [256]Id,
    ascii_token: [128]Id,

    pub fn deinit(model: *Model, gpa: Allocator) void {
        model.merges.deinit(gpa);
        model.* = undefined;
    }

    pub fn encode(model: *const Model, gpa: Allocator, raw: []const u8, out: *std.ArrayList(Id)) EncodeError!void {
        return model.encodeWithPrefix(gpa, raw, false, out);
    }

    /// Shared scalar/byte BPE input for SentencePiece wrappers. Gemma itself
    /// always passes false. Prefix insertion does not materialize normalized
    /// text, and an existing initial ASCII space or U+2581 already supplies it.
    pub fn encodeWithPrefix(model: *const Model, gpa: Allocator, raw: []const u8, prepend: bool, out: *std.ArrayList(Id)) EncodeError!void {
        if (raw.len == 0) return;
        if (!std.unicode.utf8ValidateSlice(raw)) return error.InvalidUtf8;
        const prefix = prepend and raw[0] != ' ' and !std.mem.startsWith(u8, raw, space_marker);
        // One symbol per scalar or fallback byte, plus an optional prefix.
        const capacity = std.math.add(usize, raw.len, @intFromBool(prefix)) catch return error.OutOfMemory;
        var local: [128]Id = undefined;
        const symbols = if (capacity <= local.len) local[0..capacity] else try gpa.alloc(Id, capacity);
        defer if (capacity > local.len) gpa.free(symbols);
        var count: usize = 0;
        if (prefix) {
            symbols[0] = model.ascii_token[' '];
            count = 1;
        }
        var pos: usize = 0;
        while (pos < raw.len) {
            const byte = raw[pos];
            if (byte < 128) {
                symbols[count] = model.ascii_token[byte];
                count += 1;
                pos += 1;
                continue;
            }
            const len = std.unicode.utf8ByteSequenceLength(byte) catch unreachable;
            const scalar = raw[pos..][0..len];
            if (model.ids.get(scalar)) |id| {
                symbols[count] = id;
                count += 1;
            } else {
                for (scalar) |fallback| {
                    symbols[count] = model.byte_token[fallback];
                    count += 1;
                }
            }
            pos += len;
        }
        var scratch: bpe.Scratch = .{};
        defer scratch.deinit(gpa);
        try scratch.encodeIds(gpa, &model.merges, symbols[0..count], out);
    }

    /// Appends the HF decoder result, retaining special-token text. Each call
    /// must include complete consecutive fallback runs; there is no prefix state.
    pub fn decode(model: *const Model, gpa: Allocator, token_ids: []const Id, out: *std.ArrayList(u8)) DecodeError!void {
        var pos: usize = 0;
        while (pos < token_ids.len) {
            const text = try model.token(token_ids[pos]);
            if (fallbackByte(text) == null) {
                try appendReplaced(gpa, text, out);
                pos += 1;
                continue;
            }
            const start = out.items.len;
            while (pos < token_ids.len) {
                const byte = fallbackByte(try model.token(token_ids[pos])) orelse break;
                try out.append(gpa, byte);
                pos += 1;
            }
            const count = out.items.len - start;
            if (!std.unicode.utf8ValidateSlice(out.items[start..])) {
                out.shrinkRetainingCapacity(start);
                const needed = std.math.mul(usize, count, 3) catch return error.OutOfMemory;
                try out.ensureUnusedCapacity(gpa, needed);
                for (0..count) |_| out.appendSliceAssumeCapacity("�");
            }
        }
    }

    /// Stateless raw bytes for a UTF-8 streaming assembler, not HF's replacement
    /// policy for malformed byte runs. Ordinary pieces still replace U+2581.
    pub fn decodeRaw(model: *const Model, gpa: Allocator, token_ids: []const Id, out: *std.ArrayList(u8)) DecodeError!void {
        for (token_ids) |id| {
            const text = try model.token(id);
            if (fallbackByte(text)) |byte| {
                try out.append(gpa, byte);
            } else try appendReplaced(gpa, text, out);
        }
    }

    fn token(model: *const Model, id: Id) error{UnknownToken}![]const u8 {
        if (id >= model.tokens.len) return error.UnknownToken;
        return model.tokens[id];
    }
};

/// Borrows tokens and the immutable ids map. GGUF storage and map entries must
/// outlive Model. Raw added-token extraction and any BOS policy belong upstream.
pub fn init(gpa: Allocator, file: gguf.File, tokens: []const []const u8, ids: *const std.StringHashMapUnmanaged(Id)) LoadError!Model {
    if (tokens.len > std.math.maxInt(Id)) return error.BadMetadata;
    if (file.get("tokenizer.ggml.add_space_prefix")) |value| {
        if (value != .bool) return error.BadMetadata;
        if (value.bool) return error.UnsupportedTokenizer;
    }
    const value = file.get("tokenizer.ggml.merges") orelse return error.MissingMetadata;
    if (value != .array or value.array.elem_type != .string) return error.BadMetadata;
    const merge_array = value.array;
    if (merge_array.len > std.math.maxInt(u32)) return error.BadMetadata;
    var byte_token: [256]Id = undefined;
    for (&byte_token, 0..) |*id, byte| {
        const name = byteName(@intCast(byte));
        id.* = ids.get(&name) orelse return error.BadMetadata;
    }
    var ascii_token: [128]Id = undefined;
    const space = ids.get(space_marker) orelse return error.BadMetadata;
    for (&ascii_token, 0..) |*id, byte| {
        const name = [1]u8{@intCast(byte)};
        id.* = if (byte == ' ') space else ids.get(&name) orelse byte_token[byte];
    }
    var merges = try bpe.MergeTable.init(gpa, @intCast(merge_array.len));
    errdefer merges.deinit(gpa);
    var concat: std.ArrayList(u8) = .empty;
    defer concat.deinit(gpa);
    var it = merge_array.iterator();
    var rank: u32 = 0;
    while (it.nextString()) |merge| : (rank += 1) {
        const split = std.mem.indexOfScalar(u8, merge, ' ') orelse return error.BadMetadata;
        const left = ids.get(merge[0..split]) orelse return error.UnknownMergeToken;
        const right = ids.get(merge[split + 1 ..]) orelse return error.UnknownMergeToken;
        concat.clearRetainingCapacity();
        try concat.appendSlice(gpa, merge[0..split]);
        try concat.appendSlice(gpa, merge[split + 1 ..]);
        const joined = ids.get(concat.items) orelse return error.UnknownMergeToken;
        merges.putFirst(left, right, .{ .rank = rank, .id = joined });
    }
    return .{ .tokens = tokens, .ids = ids.*, .merges = merges, .byte_token = byte_token, .ascii_token = ascii_token };
}

fn byteName(byte: u8) [6]u8 {
    const hex = "0123456789ABCDEF";
    return .{ '<', '0', 'x', hex[byte >> 4], hex[byte & 15], '>' };
}

pub fn fallbackByte(token: []const u8) ?u8 {
    if (token.len != 6 or !std.mem.startsWith(u8, token, "<0x") or token[5] != '>') return null;
    const high = std.fmt.charToDigit(token[3], 16) catch return null;
    const low = std.fmt.charToDigit(token[4], 16) catch return null;
    return high * 16 + low;
}

fn appendReplaced(gpa: Allocator, token: []const u8, out: *std.ArrayList(u8)) Allocator.Error!void {
    var pos: usize = 0;
    while (std.mem.indexOf(u8, token[pos..], space_marker)) |offset| {
        try out.appendSlice(gpa, token[pos..][0..offset]);
        try out.append(gpa, ' ');
        pos += offset + space_marker.len;
    }
    try out.appendSlice(gpa, token[pos..]);
}

const test_byte_names = blk: {
    @setEvalBranchQuota(10_000);
    var names: [256][6]u8 = undefined;
    for (&names, 0..) |*name, byte| name.* = byteName(@intCast(byte));
    break :blk names;
};
const test_tokens = blk: {
    @setEvalBranchQuota(10_000);
    const ordinary = [_][]const u8{ "▁", "a", "b", "c", "ab", "bc", "abc", "▁a", "é", "e", "́", "<eos>", "▁▁", "\n", "\n\n", "\t", "\t\t" };
    var tokens: [256 + ordinary.len][]const u8 = undefined;
    for (&test_byte_names, 0..) |*name, index| tokens[index] = name;
    @memcpy(tokens[256..], &ordinary);
    break :blk tokens;
};

const TestModel = struct {
    ids: std.StringHashMapUnmanaged(Id) = .empty,
    model: Model = undefined,

    fn create(gpa: Allocator, merge_names: []const []const u8) !TestModel {
        var result: TestModel = .{};
        errdefer result.ids.deinit(gpa);
        for (test_tokens, 0..) |name, index| try result.ids.put(gpa, name, @intCast(index));
        var encoded: std.ArrayList(u8) = .empty;
        defer encoded.deinit(gpa);
        for (merge_names) |merge| {
            var length: [8]u8 = undefined;
            std.mem.writeInt(u64, &length, @intCast(merge.len), .little);
            try encoded.appendSlice(gpa, &length);
            try encoded.appendSlice(gpa, merge);
        }
        var metadata = [_]gguf.KeyValue{.{ .key = "tokenizer.ggml.merges", .value = .{ .array = .{ .elem_type = .string, .len = merge_names.len, .bytes = encoded.items } } }};
        const file: gguf.File = .{ .version = 3, .alignment = 32, .metadata = &metadata, .tensors = &.{}, .data_offset = 0 };
        result.model = try init(gpa, file, &test_tokens, &result.ids);
        return result;
    }

    fn deinit(test_model: *TestModel, gpa: Allocator) void {
        test_model.model.deinit(gpa);
        test_model.ids.deinit(gpa);
    }
};

test "Gemma normalization is space-only with no prefix and no word splitting" {
    const gpa = std.testing.allocator;
    var fixture = try TestModel.create(gpa, &.{ "a b", "ab c", "▁ a", "▁ ▁", "\n \n", "\t \t" });
    defer fixture.deinit(gpa);
    var out: std.ArrayList(Id) = .empty;
    defer out.deinit(gpa);
    const cases = [_]struct { text: []const u8, ids: []const Id }{
        .{ .text = "", .ids = &.{} },
        .{ .text = "abc", .ids = &.{262} },
        .{ .text = "a", .ids = &.{257} },
        .{ .text = " a", .ids = &.{263} },
        .{ .text = "▁a", .ids = &.{263} },
        .{ .text = "  a", .ids = &.{ 256, 263 } },
        .{ .text = "\n\n\t\t", .ids = &.{ 270, 272 } },
        .{ .text = "é", .ids = &.{ 265, 266 } },
        .{ .text = "é", .ids = &.{264} },
        .{ .text = "\x00🫩", .ids = &.{ 0, 0xF0, 0x9F, 0xAB, 0xA9 } },
    };
    for (cases) |case| {
        out.clearRetainingCapacity();
        try fixture.model.encode(gpa, case.text, &out);
        try std.testing.expectEqualSlices(Id, case.ids, out.items);
    }
    try std.testing.expectError(error.InvalidUtf8, fixture.model.encode(gpa, "\xFF", &out));
}

test "Gemma uses explicit ranks not scores or greedy whole-token matches" {
    const gpa = std.testing.allocator;
    var fixture = try TestModel.create(gpa, &.{ "b c", "a b", "ab c" });
    defer fixture.deinit(gpa);
    var out: std.ArrayList(Id) = .empty;
    defer out.deinit(gpa);
    try fixture.model.encode(gpa, "abc", &out);
    try std.testing.expectEqualSlices(Id, &.{ 257, 261 }, out.items);
    // Mutating the merge graph while keeping the vocabulary identical must
    // change the encoding, despite "abc" being a complete vocabulary token.
    var mutated = try TestModel.create(gpa, &.{ "a b", "b c", "ab c" });
    defer mutated.deinit(gpa);
    out.clearRetainingCapacity();
    try mutated.model.encode(gpa, "abc", &out);
    try std.testing.expectEqualSlices(Id, &.{262}, out.items);
}

test "Gemma decoder matches HF byte-run oracle and replacement ordering" {
    const gpa = std.testing.allocator;
    var fixture = try TestModel.create(gpa, &.{});
    defer fixture.deinit(gpa);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    // Observed with google/gemma-4-31B-it tokenizer.json revision
    // 842da3794eaa0b77d5f08bae87a17459d91ff475, tokenizers ByteFallback.
    const cases = [_]struct { ids: []const Id, text: []const u8 }{
        .{ .ids = &.{ 0xC3, 0xA9 }, .text = "é" },
        .{ .ids = &.{0xC3}, .text = "�" },
        .{ .ids = &.{ 0xC3, 0x41 }, .text = "��" },
        .{ .ids = &.{ 0xF0, 0x9F }, .text = "��" },
        .{ .ids = &.{ 0xFF, 0xC3, 0xA9 }, .text = "���" },
        .{ .ids = &.{ 0xE2, 0x96, 0x81 }, .text = "▁" },
        .{ .ids = &.{ 256, 268, 267 }, .text = "   <eos>" },
        .{ .ids = &.{ 0xC3, 257, 0x41 }, .text = "�aA" },
    };
    for (cases) |case| {
        out.clearRetainingCapacity();
        try fixture.model.decode(gpa, case.ids, &out);
        try std.testing.expectEqualStrings(case.text, out.items);
    }
    out.clearRetainingCapacity();
    try fixture.model.decodeRaw(gpa, &.{0xC3}, &out);
    try fixture.model.decodeRaw(gpa, &.{0xA9}, &out);
    try std.testing.expectEqualStrings("é", out.items);
    try std.testing.expectError(error.UnknownToken, fixture.model.decode(gpa, &.{999999}, &out));
}

test "Gemma metadata mutations reject unresolved merges instead of falling back" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.UnknownMergeToken, TestModel.create(gpa, &.{"a missing"}));
    // Both source symbols exist, but their concatenation does not.
    try std.testing.expectError(error.UnknownMergeToken, TestModel.create(gpa, &.{"c a"}));
    try std.testing.expectError(error.BadMetadata, TestModel.create(gpa, &.{"abc"}));
}

test "optional SentencePiece prefix is inserted once before merge scheduling" {
    const gpa = std.testing.allocator;
    var fixture = try TestModel.create(gpa, &.{"▁ a"});
    defer fixture.deinit(gpa);
    var out: std.ArrayList(Id) = .empty;
    defer out.deinit(gpa);
    const cases = [_]struct { text: []const u8, prepend: bool, ids: []const Id }{
        .{ .text = "", .prepend = true, .ids = &.{} },
        .{ .text = "a", .prepend = false, .ids = &.{257} },
        .{ .text = "a", .prepend = true, .ids = &.{263} },
        .{ .text = " a", .prepend = true, .ids = &.{263} },
        .{ .text = "▁a", .prepend = true, .ids = &.{263} },
        .{ .text = "\na", .prepend = true, .ids = &.{ 256, 269, 257 } },
    };
    for (cases) |case| {
        out.clearRetainingCapacity();
        try fixture.model.encodeWithPrefix(gpa, case.text, case.prepend, &out);
        try std.testing.expectEqualSlices(Id, case.ids, out.items);
    }
}
