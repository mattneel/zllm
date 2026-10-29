//! Byte-level BPE tokenizer built from GGUF `tokenizer.ggml.*` metadata.
//!
//! Encoding follows Hugging Face `tokenizers` for the supported models:
//!
//! 1. Split out added tokens (GGUF token types CONTROL and USER_DEFINED),
//!    leftmost-longest, on the raw text.
//! 2. NFC-normalize the remaining segments.
//! 3. Split each segment with the model's pre-tokenizer regex.
//! 4. Map each piece's bytes to the GPT-2 byte alphabet and apply BPE merges
//!    in rank order, leftmost first on ties.
//!
//! Token strings borrow from the GGUF bytes, which must outlive the tokenizer.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const unicode = @import("unicode.zig");
const pretokenize = @import("pretokenize.zig");

pub const Id = u32;

/// GGUF `tokenizer.ggml.token_type` values.
pub const TokenType = enum(i32) {
    undefined = 0,
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
    _,
};

pub const LoadError = error{
    UnsupportedTokenizer,
    MissingMetadata,
    BadMetadata,
    UnknownMergeToken,
} || Allocator.Error;

pub const Tokenizer = struct {
    tokens: []const []const u8,
    types: []TokenType,
    ids: std.StringHashMapUnmanaged(Id),
    merges: std.AutoHashMapUnmanaged(u64, Merge),
    /// Token id of each byte's single-character GPT-2 alphabet token.
    byte_token: [256]Id,
    /// Added tokens, longest first, matched on raw text before normalization.
    added: []Id,
    pre: pretokenize.Kind,
    eos: ?Id,
    bos: ?Id,
    add_bos: bool,

    const Merge = struct { rank: u32, id: Id };

    pub fn deinit(t: *Tokenizer, gpa: Allocator) void {
        gpa.free(t.tokens);
        gpa.free(t.types);
        t.ids.deinit(gpa);
        t.merges.deinit(gpa);
        gpa.free(t.added);
        t.* = undefined;
    }

    pub fn load(gpa: Allocator, file: gguf.File) LoadError!Tokenizer {
        const model = getString(file, "tokenizer.ggml.model") orelse return error.MissingMetadata;
        if (!std.mem.eql(u8, model, "gpt2")) return error.UnsupportedTokenizer;
        const pre_name = getString(file, "tokenizer.ggml.pre") orelse return error.MissingMetadata;
        const pre = pretokenize.Kind.fromGgufName(pre_name) orelse return error.UnsupportedTokenizer;

        const token_arr = getArray(file, "tokenizer.ggml.tokens", .string) orelse return error.MissingMetadata;
        const type_arr = getArray(file, "tokenizer.ggml.token_type", .i32) orelse return error.MissingMetadata;
        const merge_arr = getArray(file, "tokenizer.ggml.merges", .string) orelse return error.MissingMetadata;
        if (type_arr.len != token_arr.len) return error.BadMetadata;
        const n: usize = @intCast(token_arr.len);
        if (n > std.math.maxInt(Id)) return error.BadMetadata;

        const tokens = try gpa.alloc([]const u8, n);
        errdefer gpa.free(tokens);
        const types = try gpa.alloc(TokenType, n);
        errdefer gpa.free(types);
        var ids: std.StringHashMapUnmanaged(Id) = .empty;
        errdefer ids.deinit(gpa);
        try ids.ensureTotalCapacity(gpa, @intCast(n));

        var added: std.ArrayList(Id) = .empty;
        errdefer added.deinit(gpa);

        var it = token_arr.iterator();
        for (tokens, types, 0..) |*tok, *ty, i| {
            tok.* = it.nextString().?;
            ty.* = @fromBackingInt(@intCast(@as(i32, @bitCast(type_arr.get(u32, i)))));
            // First occurrence wins, as in Hugging Face vocab loading.
            const gop = ids.getOrPutAssumeCapacity(tok.*);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(i);
            if ((ty.* == .control or ty.* == .user_defined) and tok.len > 0) try added.append(gpa, @intCast(i));
        }

        // Longest first so a linear scan at each position is leftmost-longest.
        std.mem.sort(Id, added.items, tokens, struct {
            fn lessThan(toks: []const []const u8, a: Id, b: Id) bool {
                return toks[a].len > toks[b].len;
            }
        }.lessThan);

        var byte_token: [256]Id = undefined;
        for (0..256) |b| {
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(byteToCodepoint(@intCast(b)), &buf) catch unreachable;
            byte_token[b] = ids.get(buf[0..len]) orelse return error.BadMetadata;
        }

        var merges: std.AutoHashMapUnmanaged(u64, Merge) = .empty;
        errdefer merges.deinit(gpa);
        try merges.ensureTotalCapacity(gpa, @intCast(merge_arr.len));
        var mit = merge_arr.iterator();
        var rank: u32 = 0;
        var concat: std.ArrayList(u8) = .empty;
        defer concat.deinit(gpa);
        while (mit.nextString()) |m| : (rank += 1) {
            const sp = std.mem.indexOfScalar(u8, m, ' ') orelse return error.BadMetadata;
            const left = ids.get(m[0..sp]) orelse return error.UnknownMergeToken;
            const right = ids.get(m[sp + 1 ..]) orelse return error.UnknownMergeToken;
            concat.clearRetainingCapacity();
            try concat.appendSlice(gpa, m[0..sp]);
            try concat.appendSlice(gpa, m[sp + 1 ..]);
            const merged = ids.get(concat.items) orelse return error.UnknownMergeToken;
            const gop = merges.getOrPutAssumeCapacity(pairKey(left, right));
            if (!gop.found_existing) gop.value_ptr.* = .{ .rank = rank, .id = merged };
        }

        return .{
            .tokens = tokens,
            .types = types,
            .ids = ids,
            .merges = merges,
            .byte_token = byte_token,
            .added = try added.toOwnedSlice(gpa),
            .pre = pre,
            .eos = getId(file, "tokenizer.ggml.eos_token_id"),
            .bos = getId(file, "tokenizer.ggml.bos_token_id"),
            .add_bos = if (file.get("tokenizer.ggml.add_bos_token")) |v| (v == .bool and v.bool) else false,
        };
    }

    pub const EncodeError = unicode.NormalizeError;

    /// Appends the token ids of `text` to `out`. Adds BOS when the model asks for it.
    pub fn encode(t: *const Tokenizer, gpa: Allocator, text: []const u8, out: *std.ArrayList(Id)) EncodeError!void {
        if (t.add_bos) if (t.bos) |b| try out.append(gpa, b);

        var pos: usize = 0;
        var seg_start: usize = 0;
        while (pos < text.len) {
            if (t.matchAdded(text[pos..])) |id| {
                try t.encodeSegment(gpa, text[seg_start..pos], out);
                try out.append(gpa, id);
                pos += t.tokens[id].len;
                seg_start = pos;
            } else {
                pos += 1;
            }
        }
        try t.encodeSegment(gpa, text[seg_start..], out);
    }

    fn matchAdded(t: *const Tokenizer, rest: []const u8) ?Id {
        for (t.added) |id| {
            if (std.mem.startsWith(u8, rest, t.tokens[id])) return id;
        }
        return null;
    }

    fn encodeSegment(t: *const Tokenizer, gpa: Allocator, raw: []const u8, out: *std.ArrayList(Id)) EncodeError!void {
        if (raw.len == 0) return;
        const text = try unicode.nfc(gpa, raw);
        defer gpa.free(text);

        var symbols: std.ArrayList(Id) = .empty;
        defer symbols.deinit(gpa);
        var pieces = pretokenize.iterator(t.pre, text);
        while (pieces.next()) |piece| {
            symbols.clearRetainingCapacity();
            try symbols.ensureTotalCapacity(gpa, piece.len);
            for (piece) |b| symbols.appendAssumeCapacity(t.byte_token[b]);
            t.mergeAll(&symbols);
            try out.appendSlice(gpa, symbols.items);
        }
    }

    /// Repeatedly merges the lowest-ranked adjacent pair, leftmost on ties.
    fn mergeAll(t: *const Tokenizer, symbols: *std.ArrayList(Id)) void {
        while (symbols.items.len > 1) {
            var best: ?usize = null;
            var best_merge: Merge = undefined;
            for (symbols.items[0 .. symbols.items.len - 1], symbols.items[1..], 0..) |a, b, i| {
                const m = t.merges.get(pairKey(a, b)) orelse continue;
                if (best == null or m.rank < best_merge.rank) {
                    best = i;
                    best_merge = m;
                }
            }
            const i = best orelse return;
            symbols.items[i] = best_merge.id;
            _ = symbols.orderedRemove(i + 1);
        }
    }

    pub const DecodeError = error{UnknownToken} || Allocator.Error;

    /// Appends the bytes of `ids` to `out`. Added tokens decode to their text.
    pub fn decode(t: *const Tokenizer, gpa: Allocator, ids: []const Id, out: *std.ArrayList(u8)) DecodeError!void {
        for (ids) |id| {
            if (id >= t.tokens.len) return error.UnknownToken;
            const s = t.tokens[id];
            switch (t.types[id]) {
                .control, .user_defined => try out.appendSlice(gpa, s),
                else => {
                    var it = std.unicode.Utf8View.initUnchecked(s).iterator();
                    while (it.nextCodepoint()) |cp| {
                        try out.append(gpa, codepointToByte(cp) orelse return error.UnknownToken);
                    }
                },
            }
        }
    }
};

fn pairKey(a: Id, b: Id) u64 {
    return (@as(u64, a) << 32) | b;
}

fn getString(f: gguf.File, key: []const u8) ?[]const u8 {
    const v = f.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

fn getArray(f: gguf.File, key: []const u8, elem: gguf.ValueType) ?gguf.Array {
    const v = f.get(key) orelse return null;
    if (v != .array or v.array.elem_type != elem) return null;
    return v.array;
}

fn getId(f: gguf.File, key: []const u8) ?Id {
    const v = f.get(key) orelse return null;
    return switch (v) {
        .u32 => |x| x,
        .i32 => |x| if (x >= 0) @intCast(x) else null,
        else => null,
    };
}

/// GPT-2 byte-level alphabet: printable Latin-1 bytes map to themselves, the
/// rest to U+0100 upward in byte order.
fn byteToCodepoint(b: u8) u21 {
    if (isDirectByte(b)) return b;
    var n: u21 = 0;
    for (0..b) |x| {
        if (!isDirectByte(@intCast(x))) n += 1;
    }
    return 256 + n;
}

fn isDirectByte(b: u8) bool {
    return (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or b >= 0xAE;
}

const codepoint_to_byte: [324]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var table: [324]u8 = @splat(0);
    for (0..256) |b| table[byteToCodepoint(b)] = b;
    break :blk table;
};

fn codepointToByte(cp: u21) ?u8 {
    if (cp >= codepoint_to_byte.len) return null;
    const b = codepoint_to_byte[cp];
    return if (byteToCodepoint(b) == cp) b else null;
}

test "byte alphabet is a bijection onto 256 code points" {
    var seen: std.AutoHashMapUnmanaged(u21, void) = .empty;
    defer seen.deinit(std.testing.allocator);
    for (0..256) |b| {
        const cp = byteToCodepoint(@intCast(b));
        try std.testing.expectEqual(@as(?u8, @intCast(b)), codepointToByte(cp));
        try seen.put(std.testing.allocator, cp, {});
    }
    try std.testing.expectEqual(@as(u32, 256), seen.count());
    try std.testing.expectEqual(@as(u21, 'A'), byteToCodepoint('A'));
    try std.testing.expectEqual(@as(u21, 0x120), byteToCodepoint(' '));
    try std.testing.expectEqual(@as(u21, 0x10A), byteToCodepoint('\n'));
}
