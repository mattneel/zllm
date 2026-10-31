//! Unicode properties and NFC normalization for the tokenizer.
//!
//! Tables come from `unicode_data.zig` (generated, Unicode 16.0). Only what the
//! supported pre-tokenizers need is exposed: general categories L, M, N, the
//! White_Space property, and canonical composition (NFC).

const std = @import("std");
const Allocator = std.mem.Allocator;
const data = @import("unicode_data.zig");

pub const version = data.version;

fn inRanges(ranges: []const data.Range, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const r = ranges[mid];
        if (cp < r.first) {
            hi = mid;
        } else if (cp > r.last) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

/// General category L*.
pub fn isLetter(cp: u21) bool {
    if (cp < 0x80) return std.ascii.isAlphabetic(@intCast(cp));
    return inRanges(&data.letter, cp);
}

/// General category M*.
pub fn isMark(cp: u21) bool {
    if (cp < 0x300) return false;
    return inRanges(&data.mark, cp);
}

/// General category N*.
pub fn isNumber(cp: u21) bool {
    if (cp < 0x80) return cp >= '0' and cp <= '9';
    return inRanges(&data.number, cp);
}

/// White_Space property (what `\s` matches in Unicode regex engines).
pub fn isSpace(cp: u21) bool {
    if (cp < 0x80) return cp == ' ' or (cp >= 0x09 and cp <= 0x0D);
    return inRanges(&data.white_space, cp);
}

pub fn combiningClass(cp: u21) u8 {
    if (cp < 0x300) return 0;
    const t = &data.combining_class;
    var lo: usize = 0;
    var hi: usize = t.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < t[mid].first) {
            hi = mid;
        } else if (cp > t[mid].last) {
            lo = mid + 1;
        } else {
            return t[mid].ccc;
        }
    }
    return 0;
}

fn decompositionOf(cp: u21) ?[]const u21 {
    const t = &data.decomposition;
    var lo: usize = 0;
    var hi: usize = t.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (cp < t[mid].cp) {
            hi = mid;
        } else if (cp > t[mid].cp) {
            lo = mid + 1;
        } else {
            return data.decomposition_data[t[mid].start..][0..t[mid].len];
        }
    }
    return null;
}

fn composePair(a: u21, b: u21) ?u21 {
    // Hangul LV and LVT.
    if (a >= hangul.l_base and a < hangul.l_base + hangul.l_count and
        b >= hangul.v_base and b < hangul.v_base + hangul.v_count)
    {
        const l = a - hangul.l_base;
        const v = b - hangul.v_base;
        return hangul.s_base + (l * hangul.v_count + v) * hangul.t_count;
    }
    if (a >= hangul.s_base and a < hangul.s_base + hangul.s_count and
        (a - hangul.s_base) % hangul.t_count == 0 and
        b > hangul.t_base and b < hangul.t_base + hangul.t_count)
    {
        return a + (b - hangul.t_base);
    }

    const t = &data.composition;
    var lo: usize = 0;
    var hi: usize = t.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const e = t[mid];
        const order = if (a != e.first) std.math.order(a, e.first) else std.math.order(b, e.second);
        switch (order) {
            .lt => hi = mid,
            .gt => lo = mid + 1,
            .eq => return e.composite,
        }
    }
    return null;
}

const hangul = struct {
    const s_base: u21 = 0xAC00;
    const l_base: u21 = 0x1100;
    const v_base: u21 = 0x1161;
    const t_base: u21 = 0x11A7;
    const l_count: u21 = 19;
    const v_count: u21 = 21;
    const t_count: u21 = 28;
    const s_count: u21 = l_count * v_count * t_count;
};

fn decomposeInto(gpa: Allocator, out: *std.ArrayList(u21), cp: u21) Allocator.Error!void {
    if (cp >= hangul.s_base and cp < hangul.s_base + hangul.s_count) {
        const s = cp - hangul.s_base;
        try out.append(gpa, hangul.l_base + s / (hangul.v_count * hangul.t_count));
        try out.append(gpa, hangul.v_base + (s % (hangul.v_count * hangul.t_count)) / hangul.t_count);
        const t = s % hangul.t_count;
        if (t != 0) try out.append(gpa, hangul.t_base + t);
        return;
    }
    if (decompositionOf(cp)) |parts| {
        for (parts) |p| try decomposeInto(gpa, out, p);
    } else {
        try out.append(gpa, cp);
    }
}

pub const NormalizeError = error{InvalidUtf8} || Allocator.Error;

/// Returns owned NFC bytes only when normalization may change `s`; null means
/// the validated input can be borrowed unchanged, with no allocation or copy.
pub fn nfc(gpa: Allocator, s: []const u8) NormalizeError!?[]u8 {
    if (!std.unicode.utf8ValidateSlice(s)) return error.InvalidUtf8;

    // ASCII and text without decomposable or combining code points is
    // already NFC; most input takes this path.
    if (isTriviallyNfc(s)) return null;

    var cps: std.ArrayList(u21) = .empty;
    defer cps.deinit(gpa);
    var it = std.unicode.Utf8View.initUnchecked(s).iterator();
    while (it.nextCodepoint()) |cp| try decomposeInto(gpa, &cps, cp);

    canonicalOrder(cps.items);
    const n = compose(cps.items);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, s.len);
    for (cps.items[0..n]) |cp| {
        var buf: [4]u8 = undefined;
        const len = std.unicode.utf8Encode(cp, &buf) catch unreachable;
        try out.appendSlice(gpa, buf[0..len]);
    }
    return try out.toOwnedSlice(gpa);
}

fn isTriviallyNfc(s: []const u8) bool {
    var it = std.unicode.Utf8View.initUnchecked(s).iterator();
    while (it.nextCodepoint()) |cp| {
        if (cp >= 0xC0 and inRanges(&data.nfc_unsafe, cp)) return false;
    }
    return true;
}

/// Stable sort of each run of non-starters by combining class.
fn canonicalOrder(cps: []u21) void {
    var i: usize = 1;
    while (i < cps.len) : (i += 1) {
        const cc = combiningClass(cps[i]);
        if (cc == 0) continue;
        var j = i;
        while (j > 0) : (j -= 1) {
            const prev = combiningClass(cps[j - 1]);
            if (prev == 0 or prev <= cc) break;
            std.mem.swap(u21, &cps[j], &cps[j - 1]);
        }
    }
}

/// Canonical composition in place; returns the new length.
fn compose(cps: []u21) usize {
    if (cps.len == 0) return 0;
    var out: usize = 1;
    var starter: ?usize = if (combiningClass(cps[0]) == 0) 0 else null;
    var last_cc: u8 = combiningClass(cps[0]);
    for (cps[1..]) |cp| {
        const cc = combiningClass(cp);
        if (starter) |si| {
            const adjacent = out - 1 == si;
            const unblocked = adjacent or (last_cc != 0 and last_cc < cc);
            if (unblocked) {
                if (composePair(cps[si], cp)) |c| {
                    cps[si] = c;
                    continue;
                }
            }
        }
        if (cc == 0) starter = out;
        last_cc = cc;
        cps[out] = cp;
        out += 1;
    }
    return out;
}

test "categories" {
    try std.testing.expect(isLetter('a') and isLetter(0x00E9) and isLetter(0x4E2D));
    try std.testing.expect(!isLetter('1') and !isLetter(' ') and !isLetter(0x0301));
    try std.testing.expect(isMark(0x0301) and isMark(0x093F));
    try std.testing.expect(isNumber('7') and isNumber(0x0663) and isNumber(0x2163));
    try std.testing.expect(isSpace(' ') and isSpace('\n') and isSpace(0x3000) and isSpace(0x00A0));
    try std.testing.expect(!isSpace(0x200B) and !isSpace(0x1C));
}

fn expectNfc(input: []const u8, want: []const u8) !void {
    const got = try nfc(std.testing.allocator, input);
    defer if (got) |owned| std.testing.allocator.free(owned);
    try std.testing.expectEqualStrings(want, got orelse input);
}

test "nfc composes, reorders and leaves NFC text alone" {
    try expectNfc("plain ascii", "plain ascii");
    try expectNfc("e\u{0301}", "\u{00E9}");
    // Singleton decomposition: Angstrom sign -> A with ring.
    try expectNfc("\u{212B}", "\u{00C5}");
    // Combining marks out of canonical order get sorted, then composed.
    try expectNfc("a\u{0323}\u{0302}", "\u{1EAD}");
    try expectNfc("a\u{0302}\u{0323}", "\u{1EAD}");
    // Hangul jamo compose to a syllable; syllables stay.
    try expectNfc("\u{1100}\u{1161}\u{11A8}", "\u{AC01}");
    try expectNfc("\u{AC01}", "\u{AC01}");
    // Composition exclusion stays decomposed.
    try expectNfc("\u{0958}", "\u{0915}\u{093C}");
    // Starter + starter composite (Oriya vowel sign).
    try expectNfc("\u{0B47}\u{0B3E}", "\u{0B4B}");
}

test "nfc rejects invalid utf-8" {
    try std.testing.expectError(error.InvalidUtf8, nfc(std.testing.allocator, "\xff"));
}

test "already normalized text borrows input without allocation" {
    try std.testing.expectEqual(@as(?[]u8, null), try nfc(std.testing.failing_allocator, "plain ascii"));
    try std.testing.expectEqual(@as(?[]u8, null), try nfc(std.testing.failing_allocator, "日本語"));
}
