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

    pub fn fromGgufName(name: []const u8) ?Kind {
        if (std.mem.eql(u8, name, "qwen35")) return .qwen35;
        return null;
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
            .qwen35 => matchQwen35(it.text, start),
        };
        std.debug.assert(end > start);
        it.pos = end;
        return it.text[start..end];
    }
};

/// Code point at byte offset `i` and its encoded length.
fn at(text: []const u8, i: usize) struct { cp: u21, len: usize } {
    const len = std.unicode.utf8ByteSequenceLength(text[i]) catch unreachable;
    const cp = std.unicode.utf8Decode(text[i..][0..len]) catch unreachable;
    return .{ .cp = cp, .len = len };
}

fn isNewline(cp: u21) bool {
    return cp == '\r' or cp == '\n';
}

fn isLetterOrMark(cp: u21) bool {
    return unicode.isLetter(cp) or unicode.isMark(cp);
}

/// End of the first alternative matching at `start`; every input matches one.
fn matchQwen35(text: []const u8, start: usize) usize {
    if (contraction(text, start)) |end| return end;
    if (letters(text, start)) |end| return end;

    const c = at(text, start);
    if (unicode.isNumber(c.cp)) return start + c.len;
    if (punctuation(text, start)) |end| return end;
    return whitespace(text, start);
}

/// (?i:'s|'t|'re|'ve|'m|'ll|'d)
fn contraction(text: []const u8, start: usize) ?usize {
    if (text[start] != '\'') return null;
    const rest = text[start + 1 ..];
    const suffixes = [_][]const u8{ "s", "t", "re", "ve", "m", "ll", "d" };
    for (suffixes) |s| {
        if (rest.len >= s.len and std.ascii.eqlIgnoreCase(rest[0..s.len], s)) return start + 1 + s.len;
    }
    return null;
}

/// [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
fn letters(text: []const u8, start: usize) ?usize {
    var i = start;
    const first = at(text, i);
    if (!isLetterOrMark(first.cp)) {
        // Optional prefix: anything but CR, LF, letters and numbers.
        if (isNewline(first.cp) or unicode.isNumber(first.cp)) return null;
        i += first.len;
        if (i >= text.len or !isLetterOrMark(at(text, i).cp)) return null;
    }
    while (i < text.len) {
        const c = at(text, i);
        if (!isLetterOrMark(c.cp)) break;
        i += c.len;
    }
    return i;
}

/// ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
fn punctuation(text: []const u8, start: usize) ?usize {
    var i = start;
    if (text[i] == ' ') i += 1;
    const body_start = i;
    while (i < text.len) {
        const c = at(text, i);
        if (unicode.isSpace(c.cp) or isLetterOrMark(c.cp) or unicode.isNumber(c.cp)) break;
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
    var it = iterator(.qwen35, text);
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
