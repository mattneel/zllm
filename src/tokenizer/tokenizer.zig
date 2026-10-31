//! Hugging Face-compatible BPE tokenizers built from GGUF metadata.
//!
//! Encoding follows Hugging Face `tokenizers` for the supported models:
//!
//! Added tokens match leftmost-longest on raw text before normalization.
//! GPT-2-alphabet models use their NFC/regex profiles and rank-pair merges,
//! including Llama 3's whole-piece `ignore_merges` shortcut. SentencePiece
//! reconstructs exact pair ranks from scored vocabulary splits; Gemma 4 uses
//! explicit merges. Both metaspace families preserve Unicode scalar symbols
//! and byte fallback rather than mapping text to the GPT-2 alphabet.
//!
//! `encode` follows raw tokenizer.json postprocessing; `encodePrompt` also
//! applies the GGUF inference BOS policy without duplicating a template BOS.
//!
//! Token strings borrow from the GGUF bytes, which must outlive the tokenizer.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const unicode = @import("unicode.zig");
const pretokenize = @import("pretokenize.zig");
const bpe = @import("bpe.zig");
const sentencepiece = @import("sentencepiece.zig");
const gemma = @import("gemma.zig");

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
    merges: bpe.MergeTable = .{ .entries = &.{} },
    /// Byte-alphabet vocabulary decoded once at load time, in a packed arena.
    decoded: []Decoded = &.{},
    decoded_bytes: []u8 = &.{},
    /// Raw-byte vocabulary lookup for ignore_merges, without alphabet expansion.
    raw_ids: std.StringHashMapUnmanaged(Id) = .empty,
    /// Token id of each byte's single-character GPT-2 alphabet token.
    byte_token: [256]Id = undefined,
    /// Added tokens in first-byte buckets, longest first within each bucket.
    added: []Id,
    added_offsets: [257]usize,
    pre: ?pretokenize.Kind = null,
    eos: ?Id,
    bos: ?Id,
    add_bos: bool,
    engine: Engine = .gpt2,

    const Engine = union(enum) {
        gpt2,
        sentencepiece: sentencepiece.Model,
        gemma: gemma.Model,

        fn deinit(engine: *Engine, gpa: Allocator) void {
            switch (engine.*) {
                .gpt2 => {},
                .sentencepiece => |*model| model.deinit(gpa),
                .gemma => |*model| model.deinit(gpa),
            }
        }
    };

    const Decoded = struct {
        offset: usize,
        len: usize,
        const invalid = std.math.maxInt(usize);
    };

    pub fn deinit(t: *Tokenizer, gpa: Allocator) void {
        t.engine.deinit(gpa);
        gpa.free(t.tokens);
        gpa.free(t.types);
        t.ids.deinit(gpa);
        t.merges.deinit(gpa);
        gpa.free(t.added);
        gpa.free(t.decoded);
        gpa.free(t.decoded_bytes);
        t.raw_ids.deinit(gpa);
        t.* = undefined;
    }

    pub fn load(gpa: Allocator, file: gguf.File) LoadError!Tokenizer {
        const model = getString(file, "tokenizer.ggml.model") orelse return error.MissingMetadata;
        const is_spm = std.mem.eql(u8, model, "llama");
        const is_gemma = std.mem.eql(u8, model, "gemma4");
        if (!is_spm and !is_gemma and !std.mem.eql(u8, model, "gpt2")) return error.UnsupportedTokenizer;
        const pre: ?pretokenize.Kind = if (is_spm or is_gemma) null else pretokenize.Kind.fromGgufName(
            getString(file, "tokenizer.ggml.pre") orelse return error.MissingMetadata,
        ) orelse return error.UnsupportedTokenizer;

        const token_arr = getArray(file, "tokenizer.ggml.tokens", .string) orelse return error.MissingMetadata;
        const type_arr = getArray(file, "tokenizer.ggml.token_type", .i32) orelse return error.MissingMetadata;
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
            // Gemma's HF-added eos is NORMAL in GGUF, unlike its other added tokens.
            const raw_added = ty.* == .control or ty.* == .user_defined or (is_gemma and std.mem.eql(u8, tok.*, "<eos>"));
            if (raw_added and tok.len > 0) try added.append(gpa, @intCast(i));
        }

        // First-byte buckets avoid testing every added token at every byte.
        std.mem.sort(Id, added.items, tokens, struct {
            fn lessThan(toks: []const []const u8, a: Id, b: Id) bool {
                if (toks[a][0] != toks[b][0]) return toks[a][0] < toks[b][0];
                return toks[a].len > toks[b].len;
            }
        }.lessThan);
        var added_offsets: [257]usize = undefined;
        var added_index: usize = 0;
        for (0..256) |byte| {
            added_offsets[byte] = added_index;
            while (added_index < added.items.len and tokens[added.items[added_index]][0] == byte) : (added_index += 1) {}
        }
        added_offsets[256] = added.items.len;

        if (is_spm or is_gemma) {
            var engine: Engine = if (is_spm)
                .{ .sentencepiece = try sentencepiece.init(gpa, file, tokens, &ids) }
            else
                .{ .gemma = try gemma.init(gpa, file, tokens, &ids) };
            errdefer engine.deinit(gpa);
            return .{
                .tokens = tokens,
                .types = types,
                .ids = ids,
                .added = try added.toOwnedSlice(gpa),
                .added_offsets = added_offsets,
                .eos = getId(file, "tokenizer.ggml.eos_token_id"),
                .bos = getId(file, "tokenizer.ggml.bos_token_id"),
                .add_bos = if (file.get("tokenizer.ggml.add_bos_token")) |v| (v == .bool and v.bool) else false,
                .engine = engine,
            };
        }
        const merge_arr = getArray(file, "tokenizer.ggml.merges", .string) orelse return error.MissingMetadata;

        var byte_token: [256]Id = undefined;
        for (0..256) |b| {
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(byteToCodepoint(@intCast(b)), &buf) catch unreachable;
            byte_token[b] = ids.get(buf[0..len]) orelse return error.BadMetadata;
        }

        var merges = try bpe.MergeTable.init(gpa, @intCast(merge_arr.len));
        errdefer merges.deinit(gpa);
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
            merges.putFirst(left, right, .{ .rank = rank, .id = merged });
        }

        const decoded = try gpa.alloc(Decoded, n);
        errdefer gpa.free(decoded);
        const alphabet = try gpa.alloc(Decoded, n);
        defer gpa.free(alphabet);
        var arena_bytes: std.ArrayList(u8) = .empty;
        errdefer arena_bytes.deinit(gpa);
        for (tokens, types, decoded, alphabet) |token, ty, *decoded_token, *alphabet_token| {
            const start = arena_bytes.items.len;
            var valid = true;
            const view = std.unicode.Utf8View.init(token) catch return error.BadMetadata;
            var codepoints = view.iterator();
            while (codepoints.nextCodepoint()) |cp| {
                const byte = codepointToByte(cp) orelse {
                    valid = false;
                    break;
                };
                try arena_bytes.append(gpa, byte);
            }
            if (!valid) arena_bytes.shrinkRetainingCapacity(start);
            alphabet_token.* = .{ .offset = start, .len = if (valid) arena_bytes.items.len - start else Decoded.invalid };
            if (ty == .control or ty == .user_defined) {
                if (valid and std.mem.eql(u8, arena_bytes.items[start..], token)) {
                    decoded_token.* = alphabet_token.*;
                } else {
                    decoded_token.* = .{ .offset = arena_bytes.items.len, .len = token.len };
                    try arena_bytes.appendSlice(gpa, token);
                }
            } else decoded_token.* = alphabet_token.*;
        }
        const decoded_bytes = try arena_bytes.toOwnedSlice(gpa);
        errdefer gpa.free(decoded_bytes);
        var raw_ids: std.StringHashMapUnmanaged(Id) = .empty;
        errdefer raw_ids.deinit(gpa);
        if (pre.?.ignoresMerges()) {
            try raw_ids.ensureTotalCapacity(gpa, @intCast(n));
            for (alphabet, 0..) |entry, index| {
                if (entry.len == Decoded.invalid) continue;
                const gop = raw_ids.getOrPutAssumeCapacity(decoded_bytes[entry.offset..][0..entry.len]);
                if (!gop.found_existing) gop.value_ptr.* = @intCast(index);
            }
        }

        return .{
            .tokens = tokens,
            .types = types,
            .ids = ids,
            .merges = merges,
            .decoded = decoded,
            .decoded_bytes = decoded_bytes,
            .raw_ids = raw_ids,
            .byte_token = byte_token,
            .added = try added.toOwnedSlice(gpa),
            .added_offsets = added_offsets,
            .pre = pre,
            .eos = getId(file, "tokenizer.ggml.eos_token_id"),
            .bos = getId(file, "tokenizer.ggml.bos_token_id"),
            .add_bos = if (file.get("tokenizer.ggml.add_bos_token")) |v| (v == .bool and v.bool) else false,
        };
    }

    pub const EncodeError = unicode.NormalizeError;

    /// The tokens that end generation, as llama.cpp picks them: those named
    /// like an end of turn, message or text (llama-vocab.cpp's list, without
    /// its model-specific entries), and the end-of-sequence token.
    pub fn endOfGeneration(t: *const Tokenizer, gpa: Allocator, out: *std.ArrayList(Id)) Allocator.Error!void {
        const names = [_][]const u8{
            "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<end_of_turn>",   "<|endoftext|>",
            "<|eom_id|>", "<EOT>",   "_<EOT>",     "[EOT]",   "<|end_of_text|>", "<end_of_utterance>",
            "<｜end▁of▁sentence｜>",
        };
        for (names) |n| if (t.ids.get(n)) |id| try out.append(gpa, id);
        if (t.eos) |id| if (std.mem.indexOfScalar(Id, out.items, id) == null) try out.append(gpa, id);
    }

    /// Raw HF-tokenizer semantics, including its model postprocessor. Gemma's
    /// raw tokenizer.json has no BOS despite its GGUF inference BOS policy.
    pub fn encode(t: *const Tokenizer, gpa: Allocator, text: []const u8, out: *std.ArrayList(Id)) EncodeError!void {
        return t.encodeImpl(gpa, text, out, t.add_bos and t.engine != .gemma);
    }

    /// Inference prompt semantics: honor GGUF BOS, but never add a second BOS
    /// when the rendered chat template already starts with its literal token.
    pub fn encodePrompt(t: *const Tokenizer, gpa: Allocator, text: []const u8, out: *std.ArrayList(Id)) EncodeError!void {
        const has_bos = if (t.bos) |id| std.mem.startsWith(u8, text, t.tokens[id]) else false;
        return t.encodeImpl(gpa, text, out, t.add_bos and !has_bos);
    }

    fn encodeImpl(t: *const Tokenizer, gpa: Allocator, text: []const u8, out: *std.ArrayList(Id), prepend_bos: bool) EncodeError!void {
        if (prepend_bos) if (t.bos) |id| try out.append(gpa, id);
        var scratch: bpe.Scratch = .{};
        defer scratch.deinit(gpa);

        var pos: usize = 0;
        var seg_start: usize = 0;
        while (pos < text.len) {
            if (t.matchAdded(text[pos..])) |id| {
                try t.encodeSegment(gpa, text[seg_start..pos], seg_start == 0, out, &scratch);
                try out.append(gpa, id);
                pos += t.tokens[id].len;
                seg_start = pos;
            } else {
                pos += 1;
            }
        }
        try t.encodeSegment(gpa, text[seg_start..], seg_start == 0, out, &scratch);
    }

    fn matchAdded(t: *const Tokenizer, rest: []const u8) ?Id {
        for (t.added[t.added_offsets[rest[0]]..t.added_offsets[@as(usize, rest[0]) + 1]]) |id| {
            if (std.mem.startsWith(u8, rest, t.tokens[id])) return id;
        }
        return null;
    }

    fn encodeSegment(t: *const Tokenizer, gpa: Allocator, raw: []const u8, first: bool, out: *std.ArrayList(Id), scratch: *bpe.Scratch) EncodeError!void {
        if (raw.len == 0) return;
        switch (t.engine) {
            .sentencepiece => |*model| return model.encode(gpa, raw, first, out),
            .gemma => |*model| return model.encode(gpa, raw, out),
            .gpt2 => {},
        }
        const pre = t.pre.?;
        const normalized = if (pre.nfc()) try unicode.nfc(gpa, raw) else null;
        defer if (normalized) |n| gpa.free(n);
        const text = normalized orelse raw;

        var pieces = pretokenize.iterator(pre, text);
        while (pieces.next()) |piece| {
            if (piece.len == 1) {
                try out.append(gpa, t.byte_token[piece[0]]);
                continue;
            }
            if (pre.ignoresMerges()) {
                if (t.raw_ids.get(piece)) |id| {
                    try out.append(gpa, id);
                    continue;
                }
            }
            try scratch.encode(gpa, &t.merges, &t.byte_token, piece, out);
        }
    }

    pub const DecodeError = error{UnknownToken} || Allocator.Error;

    /// Appends the bytes of `ids` to `out`. Added tokens decode to their text.
    pub fn decode(t: *const Tokenizer, gpa: Allocator, ids: []const Id, out: *std.ArrayList(u8)) DecodeError!void {
        switch (t.engine) {
            .sentencepiece => |*model| return model.decode(gpa, ids, out),
            .gemma => |*model| return model.decode(gpa, ids, out),
            .gpt2 => {},
        }
        return t.decodeRaw(gpa, ids, out);
    }

    /// Internal streaming primitive: fallback bytes are validated as a whole
    /// run by Stream, and SPM's initial-space Strip is applied once there.
    pub fn decodeRaw(t: *const Tokenizer, gpa: Allocator, ids: []const Id, out: *std.ArrayList(u8)) DecodeError!void {
        switch (t.engine) {
            .sentencepiece => |*model| return model.decodeRaw(gpa, ids, out),
            .gemma => |*model| return model.decodeRaw(gpa, ids, out),
            .gpt2 => {},
        }
        for (ids) |id| {
            if (id >= t.tokens.len) return error.UnknownToken;
            const entry = t.decoded[id];
            if (entry.len == Decoded.invalid) return error.UnknownToken;
            try out.appendSlice(gpa, t.decoded_bytes[entry.offset..][0..entry.len]);
        }
    }

    pub fn fallbackByte(t: *const Tokenizer, id: Id) error{UnknownToken}!?u8 {
        if (id >= t.tokens.len) return error.UnknownToken;
        return if (t.engine == .gpt2) null else gemma.fallbackByte(t.tokens[id]);
    }

    pub fn stripsInitialSpace(t: *const Tokenizer) bool {
        return t.engine == .sentencepiece;
    }
};

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

test "predecoded bytes preserve split UTF-8 and streaming stop boundaries" {
    const gpa = std.testing.allocator;
    var bytes = "\xe2\x82\xac<stop>".*;
    var entries = [_]Tokenizer.Decoded{
        .{ .offset = 0, .len = 1 },
        .{ .offset = 1, .len = 2 },
        .{ .offset = 3, .len = 6 },
        .{ .offset = 0, .len = Tokenizer.Decoded.invalid },
    };
    const tok: Tokenizer = .{
        .tokens = &.{ "first-byte", "remaining-bytes", "<stop>", "invalid" },
        .types = &.{},
        .ids = .empty,
        .merges = .{ .entries = &.{} },
        .decoded = &entries,
        .decoded_bytes = &bytes,
        .raw_ids = .empty,
        .byte_token = undefined,
        .added = &.{},
        .added_offsets = @splat(0),
        .pre = .qwen2,
        .eos = null,
        .bos = null,
        .add_bos = false,
    };
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(gpa);
    try tok.decode(gpa, &.{ 0, 1, 2 }, &output);
    try std.testing.expectEqualStrings("€<stop>", output.items);
    try std.testing.expectError(error.UnknownToken, tok.decode(gpa, &.{3}, &output));
    try std.testing.expectError(error.UnknownToken, tok.decode(gpa, &.{4}, &output));
    output.clearRetainingCapacity();
    var stream: @import("stream.zig").Stream = .{ .stops = &.{"<stop>"} };
    defer stream.deinit(gpa);
    try std.testing.expect(!try stream.push(gpa, &tok, 0, &output));
    try std.testing.expectEqual(@as(usize, 0), output.items.len);
    try std.testing.expect(!try stream.push(gpa, &tok, 1, &output));
    try std.testing.expectEqualStrings("€", output.items);
    try std.testing.expect(try stream.push(gpa, &tok, 2, &output));
    try std.testing.expectEqualStrings("€", output.items);
}

const test_fallback_names = blk: {
    var names: [256][6]u8 = undefined;
    const hex = "0123456789ABCDEF";
    for (&names, 0..) |*name, byte| name.* = .{ '<', '0', 'x', hex[byte >> 4], hex[byte & 15], '>' };
    break :blk names;
};
const test_metaspace_tokens = blk: {
    const ordinary = [_][]const u8{ "▁", "a", "b", "▁a", "<bos>", "<eos>", "<turn|>", "▁b", "a▁" };
    var tokens: [256 + ordinary.len][]const u8 = undefined;
    for (&test_fallback_names, 0..) |*name, index| tokens[index] = name;
    @memcpy(tokens[256..], &ordinary);
    break :blk tokens;
};

fn testMetaspaceTokenizer(gpa: Allocator, spm: bool) !Tokenizer {
    const tokens = try gpa.dupe([]const u8, &test_metaspace_tokens);
    errdefer gpa.free(tokens);
    const types = try gpa.alloc(TokenType, tokens.len);
    errdefer gpa.free(types);
    @memset(types, .normal);
    @memset(types[0..256], .byte);
    for ([_]usize{ 260, 261, 262 }) |id| types[id] = .control;
    var ids: std.StringHashMapUnmanaged(Id) = .empty;
    errdefer ids.deinit(gpa);
    for (tokens, 0..) |token, index| try ids.put(gpa, token, @intCast(index));
    var merges = try bpe.MergeTable.init(gpa, 3);
    errdefer merges.deinit(gpa);
    merges.putFirst(256, 257, .{ .rank = 0, .id = 259 });
    merges.putFirst(256, 258, .{ .rank = 1, .id = 263 });
    merges.putFirst(257, 256, .{ .rank = 2, .id = 264 });
    var byte_tokens: [256]Id = undefined;
    for (&byte_tokens, 0..) |*id, byte| id.* = @intCast(byte);
    var ascii_tokens: [128]Id = undefined;
    for (&ascii_tokens, 0..) |*id, byte| {
        const name = [1]u8{@intCast(byte)};
        id.* = if (byte == ' ') 256 else ids.get(&name) orelse byte_tokens[byte];
    }
    const base: gemma.Model = .{ .tokens = tokens, .ids = ids, .merges = merges, .byte_token = byte_tokens, .ascii_token = ascii_tokens };
    const added = try gpa.dupe(Id, &.{ 262, 260, 261 });
    var offsets: [257]usize = undefined;
    for (&offsets, 0..) |*offset, byte| offset.* = if (byte <= '<') 0 else added.len;
    return .{
        .tokens = tokens,
        .types = types,
        .ids = ids,
        .added = added,
        .added_offsets = offsets,
        .eos = 261,
        .bos = 260,
        .add_bos = true,
        .engine = if (spm) .{ .sentencepiece = .{ .base = base, .add_prefix = true } } else .{ .gemma = base },
    };
}

test "raw Gemma omits BOS but raw inference and rendered chat each get exactly one" {
    const gpa = std.testing.allocator;
    var tok = try testMetaspaceTokenizer(gpa, false);
    defer tok.deinit(gpa);
    var ids: std.ArrayList(Id) = .empty;
    defer ids.deinit(gpa);
    try tok.encode(gpa, "a", &ids);
    try std.testing.expectEqualSlices(Id, &.{257}, ids.items);
    ids.clearRetainingCapacity();
    try tok.encodePrompt(gpa, "a", &ids);
    try std.testing.expectEqualSlices(Id, &.{ 260, 257 }, ids.items);
    ids.clearRetainingCapacity();
    // The Gemma chat template emits its BOS literal before its first turn.
    try tok.encodePrompt(gpa, "<bos>a", &ids);
    try std.testing.expectEqualSlices(Id, &.{ 260, 257 }, ids.items);
    ids.clearRetainingCapacity();
    try tok.encode(gpa, "<eos>a", &ids);
    try std.testing.expectEqualSlices(Id, &.{ 261, 257 }, ids.items);
}

test "SentencePiece first prefix is original offset and decoder strips only stream start" {
    const gpa = std.testing.allocator;
    var tok = try testMetaspaceTokenizer(gpa, true);
    defer tok.deinit(gpa);
    var ids: std.ArrayList(Id) = .empty;
    defer ids.deinit(gpa);
    try tok.encode(gpa, "a<eos>b", &ids);
    try std.testing.expectEqualSlices(Id, &.{ 260, 259, 261, 258 }, ids.items);
    ids.clearRetainingCapacity();
    try tok.encode(gpa, "<bos>a", &ids);
    try std.testing.expectEqualSlices(Id, &.{ 260, 260, 257 }, ids.items);
    ids.clearRetainingCapacity();
    try tok.encodePrompt(gpa, "<bos>a", &ids);
    try std.testing.expectEqualSlices(Id, &.{ 260, 257 }, ids.items);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try tok.decode(gpa, &.{ 259, 263 }, &out);
    try std.testing.expectEqualStrings("a b", out.items);
    out.clearRetainingCapacity();
    try tok.decode(gpa, &.{ 260, 259 }, &out);
    try std.testing.expectEqualStrings("<bos> a", out.items);
    out.clearRetainingCapacity();
    var stream: @import("stream.zig").Stream = .{ .stops = &.{} };
    defer stream.deinit(gpa);
    _ = try stream.push(gpa, &tok, 259, &out);
    _ = try stream.push(gpa, &tok, 263, &out);
    try stream.flush(gpa, &out);
    try std.testing.expectEqualStrings("a b", out.items);
}

test "metaspace streaming validates entire fallback runs before releasing or flushing" {
    const gpa = std.testing.allocator;
    var tok = try testMetaspaceTokenizer(gpa, false);
    defer tok.deinit(gpa);
    const cases = [_]struct { ids: []const Id, text: []const u8 }{
        .{ .ids = &.{ 0xC3, 0xA9, 258 }, .text = "éb" },
        .{ .ids = &.{ 0xFF, 0xC3, 0xA9, 258 }, .text = "���b" },
        .{ .ids = &.{ 0xE2, 0x96, 0x81, 258 }, .text = "▁b" },
        .{ .ids = &.{ 0xC3, 0xA9 }, .text = "é" },
        .{ .ids = &.{0xC3}, .text = "�" },
    };
    for (cases) |case| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        var stream: @import("stream.zig").Stream = .{ .stops = &.{} };
        defer stream.deinit(gpa);
        for (case.ids) |id| _ = try stream.push(gpa, &tok, id, &out);
        try stream.flush(gpa, &out);
        try std.testing.expectEqualStrings(case.text, out.items);
    }
}
