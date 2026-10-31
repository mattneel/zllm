//! Pre-tokenizers: split normalized text into pieces before BPE.
//!
//! Each supported GGUF `tokenizer.ggml.pre` value maps to a hand-written
//! matcher equivalent to that model's regex, so no regex engine is needed.

const std = @import("std");
const unicode = @import("unicode.zig");

pub const Kind = enum {
    /// Qwen 3.5 / 3.6:
    /// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
    /// | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    qwen35,
    /// Qwen 2 / 3: as qwen35, with marks outside letter runs:
    /// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
    /// | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    qwen2,
    /// Llama 3: as qwen2, with up to three digits a piece (\p{N}{1,3}).
    llama3,
    /// GPT-2 (Granite 4):
    /// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    gpt2,

    pub fn fromGgufName(name: []const u8) ?Kind {
        const names = [_]struct { []const u8, Kind }{
            .{ "qwen35", .qwen35 },
            .{ "qwen2", .qwen2 },
            .{ "llama-bpe", .llama3 },
            .{ "granite-docling", .gpt2 },
        };
        for (names) |n| if (std.mem.eql(u8, name, n[0])) return n[1];
        return null;
    }

    /// Whether a piece that is itself a token skips the merges (the BPE
    /// model's `ignore_merges`: Llama 3's sets it).
    pub fn ignoresMerges(kind: Kind) bool {
        return kind == .llama3;
    }

    /// Whether the tokenizer NFC-normalizes text first (Qwen's declare it;
    /// Llama 3's and Granite's have no normalizer).
    pub fn nfc(kind: Kind) bool {
        return switch (kind) {
            .qwen35, .qwen2 => true,
            .llama3, .gpt2 => false,
        };
    }
};

pub fn iterator(kind: Kind, text: []const u8) Iterator {
    return .{ .kind = kind, .text = text, .pos = 0 };
}

pub const Iterator = struct {
    kind: Kind,
    /// Valid UTF-8.
    text: []const u8,
    pos: usize,

    pub fn next(it: *Iterator) ?[]const u8 {
        if (it.pos >= it.text.len) return null;
        const start = it.pos;
        const end = switch (it.kind) {
            .qwen35 => matchQwen(it.text, start, .{ .marks = true, .digits = 1 }),
            .qwen2 => matchQwen(it.text, start, .{ .marks = false, .digits = 1 }),
            .llama3 => matchQwen(it.text, start, .{ .marks = false, .digits = 3 }),
            .gpt2 => matchGpt2(it.text, start),
        };
        std.debug.assert(end > start);
        it.pos = end;
        return it.text[start..end];
    }
};

/// Code point at byte offset `i` and its encoded length.
fn at(text: []const u8, i: usize) struct { cp: u21, len: usize } {
    if (text[i] < 0x80) return .{ .cp = text[i], .len = 1 };
    const len = std.unicode.utf8ByteSequenceLength(text[i]) catch unreachable;
    const cp = std.unicode.utf8Decode(text[i..][0..len]) catch unreachable;
    return .{ .cp = cp, .len = len };
}

fn isNewline(cp: u21) bool {
    return cp == '\r' or cp == '\n';
}

/// How the Qwen-style regexes differ.
const Qwen = struct {
    /// Marks (\p{M}) join letter runs and are not punctuation.
    marks: bool,
    /// Digits a piece: \p{N}{1,digits}.
    digits: usize,

    fn isLetter(q: Qwen, cp: u21) bool {
        return unicode.isLetter(cp) or (q.marks and unicode.isMark(cp));
    }
};

/// End of the first alternative matching at `start`; every input matches one.
fn matchQwen(text: []const u8, start: usize, q: Qwen) usize {
    if (contraction(text, start, .insensitive)) |end| return end;
    if (letters(text, start, q)) |end| return end;
    if (unicode.isNumber(at(text, start).cp)) return numbers(text, start, q.digits);
    if (punctuation(text, start, q)) |end| return end;
    return whitespace(text, start);
}

/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
fn matchGpt2(text: []const u8, start: usize) usize {
    if (contraction(text, start, .sensitive)) |end| return end;
    // The optional space belongs to the run after it, whichever class.
    const body = if (text[start] == ' ' and start + 1 < text.len) start + 1 else start;
    const c = at(text, body);
    const class = Class.of(c.cp);
    if (class != .space) {
        var i = body + c.len;
        while (i < text.len) {
            const n = at(text, i);
            if (Class.of(n.cp) != class) break;
            i += n.len;
        }
        return i;
    }
    // \s+(?!\S) leaves the last of several whitespace code points to precede
    // the next piece; \s+ takes a lone one, or the run at the end.
    var i = start;
    var count: usize = 0;
    var last_start = start;
    while (i < text.len) {
        const n = at(text, i);
        if (!unicode.isSpace(n.cp)) break;
        last_start = i;
        i += n.len;
        count += 1;
    }
    std.debug.assert(count > 0);
    return if (count > 1 and i < text.len) last_start else i;
}

/// GPT-2's code point classes.
const Class = enum {
    letter,
    number,
    space,
    other,

    fn of(cp: u21) Class {
        if (unicode.isSpace(cp)) return .space;
        if (unicode.isLetter(cp)) return .letter;
        if (unicode.isNumber(cp)) return .number;
        return .other;
    }
};

/// 's|'t|'re|'ve|'m|'ll|'d, ASCII case-insensitive or not.
fn contraction(text: []const u8, start: usize, case: enum { sensitive, insensitive }) ?usize {
    if (text[start] != '\'') return null;
    const rest = text[start + 1 ..];
    const suffixes = [_][]const u8{ "s", "t", "re", "ve", "m", "ll", "d" };
    for (suffixes) |s| {
        if (rest.len < s.len) continue;
        const same = switch (case) {
            .sensitive => std.mem.eql(u8, rest[0..s.len], s),
            .insensitive => std.ascii.eqlIgnoreCase(rest[0..s.len], s),
        };
        if (same) return start + 1 + s.len;
    }
    return null;
}

/// [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ (without marks: \p{L}+)
fn letters(text: []const u8, start: usize, q: Qwen) ?usize {
    var i = start;
    const first = at(text, i);
    if (!q.isLetter(first.cp)) {
        // Optional prefix: anything but CR, LF, letters and numbers.
        if (isNewline(first.cp) or unicode.isNumber(first.cp)) return null;
        i += first.len;
        if (i >= text.len or !q.isLetter(at(text, i).cp)) return null;
    }
    while (i < text.len) {
        const c = at(text, i);
        if (!q.isLetter(c.cp)) break;
        i += c.len;
    }
    return i;
}

/// \p{N}{1,max}
fn numbers(text: []const u8, start: usize, max: usize) usize {
    var i = start;
    var count: usize = 0;
    while (i < text.len and count < max) : (count += 1) {
        const c = at(text, i);
        if (!unicode.isNumber(c.cp)) break;
        i += c.len;
    }
    return i;
}

/// ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]* (without marks: [^\s\p{L}\p{N}])
fn punctuation(text: []const u8, start: usize, q: Qwen) ?usize {
    var i = start;
    if (text[i] == ' ') i += 1;
    const body_start = i;
    while (i < text.len) {
        const c = at(text, i);
        if (unicode.isSpace(c.cp) or q.isLetter(c.cp) or unicode.isNumber(c.cp)) break;
        i += c.len;
    }
    if (i == body_start) return null;
    while (i < text.len and isNewline(text[i])) i += 1;
    return i;
}

/// \s*[\r\n]+ | \s+(?!\S) | \s+
fn whitespace(text: []const u8, start: usize) usize {
    // Maximal whitespace run, remembering where the last newline ends and
    // where the last code point begins.
    var i = start;
    var after_last_newline: ?usize = null;
    var last_start = start;
    while (i < text.len) {
        const c = at(text, i);
        if (!unicode.isSpace(c.cp)) break;
        last_start = i;
        i += c.len;
        if (isNewline(c.cp)) after_last_newline = i;
    }
    // Anything reaching here that is not whitespace is matched by an earlier
    // alternative, so the run is non-empty.
    std.debug.assert(i > start);

    // \s*[\r\n]+ backtracks the greedy \s* to the last newline in the run.
    if (after_last_newline) |end| return end;
    // \s+(?!\S): at end of text the whole run; otherwise leave the last
    // whitespace code point to precede the next token.
    if (i == text.len) return i;
    if (last_start > start) return last_start;
    // \s+ with a single whitespace code point.
    return i;
}

fn expectPieces(text: []const u8, want: []const []const u8) !void {
    try expectKindPieces(.qwen35, text, want);
}

fn expectKindPieces(kind: Kind, text: []const u8, want: []const []const u8) !void {
    var it = iterator(kind, text);
    for (want) |w| try std.testing.expectEqualStrings(w, it.next() orelse return error.TooFewPieces);
    try std.testing.expectEqual(null, it.next());
}

test "qwen35 pieces" {
    try expectPieces("Hello world", &.{ "Hello", " world" });
    try expectPieces("I'm here, it's 2024!", &.{ "I", "'m", " here", ",", " it", "'s", " ", "2", "0", "2", "4", "!" });
    try expectPieces("a  b", &.{ "a", " ", " b" });
    try expectPieces("a   ", &.{ "a", "   " });
    try expectPieces("x\n\ny", &.{ "x", "\n\n", "y" });
    try expectPieces("x \n  y", &.{ "x", " \n", " ", " y" });
    try expectPieces("foo()\n", &.{ "foo", "()\n" });
    try expectPieces(" ()", &.{" ()"});
    try expectPieces("\u{4E2D}\u{6587}", &.{"\u{4E2D}\u{6587}"});
    try expectPieces("e\u{0301}x", &.{"e\u{0301}x"});
    try expectPieces("'S", &.{"'S"});
    try expectPieces("'x", &.{"'x"});
}

test "qwen2 pieces: marks are punctuation" {
    try expectKindPieces(.qwen2, "e\u{0301}x", &.{ "e", "\u{0301}x" });
    try expectKindPieces(.qwen2, "I'm 2024", &.{ "I", "'m", " ", "2", "0", "2", "4" });
}

test "llama3 pieces: up to three digits" {
    try expectKindPieces(.llama3, "12345 x", &.{ "123", "45", " x" });
    try expectKindPieces(.llama3, "foo()\n", &.{ "foo", "()\n" });
}

test "gpt2 pieces" {
    try expectKindPieces(.gpt2, "Hello world", &.{ "Hello", " world" });
    try expectKindPieces(.gpt2, "I'm 2024!", &.{ "I", "'m", " 2024", "!" });
    try expectKindPieces(.gpt2, "'S", &.{ "'", "S" });
    try expectKindPieces(.gpt2, "a  b", &.{ "a", " ", " b" });
    try expectKindPieces(.gpt2, "x\ny", &.{ "x", "\n", "y" });
    try expectKindPieces(.gpt2, "x\n\n y", &.{ "x", "\n\n", " y" });
    try expectKindPieces(.gpt2, "a   ", &.{ "a", "   " });
    try expectKindPieces(.gpt2, "foo()\n", &.{ "foo", "()", "\n" });
}
