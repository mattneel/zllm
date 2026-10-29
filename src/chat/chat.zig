//! Chat prompts in the fixed Qwen template: froggeric/Qwen-Fixed-Chat-Templates
//! @855bffc49448e299789730ff92c9b8d834d6cc14, chat_template.jinja
//! ("qwen3.8-froggeric-v22.5"), transcribed for text conversations without
//! tools or images. tools/check_chat.py checks it against the Jinja template
//! rendered as Hugging Face transformers renders it.
//!
//! Earlier assistant turns keep their reasoning (preserve_thinking defaults
//! to true), so each new prompt extends the previous one token for token and
//! the engine reuses the whole conversation as a prefix.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Role = enum { system, developer, user, assistant };

pub const Message = struct {
    role: Role,
    content: []const u8,
    /// An assistant turn's reasoning, if kept apart from `content`
    /// (reasoning_content). Otherwise a `<think>` block inside `content` is
    /// split off.
    reasoning: ?[]const u8 = null,
};

pub const Options = struct {
    enable_thinking: bool = true,
    /// none, off, minimal, low, medium, high, xhigh, max, ultracode, extreme.
    reasoning_effort: ?[]const u8 = null,
    preserve_thinking: bool = true,
    add_generation_prompt: bool = true,
};

pub const Error = Allocator.Error || error{NoMessages};

const Effort = enum { low, medium, xhigh };

/// Tags in system and user messages that switch thinking, in the order the
/// template strips them.
const tags = [_][]const u8{
    "<|think_off|>",     "<|think_on|>",  "<|think_xhigh|>",  "<|think_high|>", "<|think_ultracode|>",
    "<|think_extreme|>", "<|think_max|>", "<|think_medium|>", "<|think_low|>",  "<|think_minimal|>",
};

pub fn render(gpa: Allocator, messages: []const Message, opts: Options, out: *std.ArrayList(u8)) Error!void {
    if (messages.len == 0) return error.NoMessages;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var thinking = opts.enable_thinking;
    var effort: Effort = .medium;
    if (opts.reasoning_effort) |raw| {
        const r = try std.ascii.allocLowerString(arena, raw);
        if (oneOf(r, &.{ "none", "off" })) {
            thinking = false;
        } else if (oneOf(r, &.{ "minimal", "low" })) {
            effort = .low;
        } else if (oneOf(r, &.{ "high", "xhigh", "max", "ultracode", "extreme" })) {
            effort = .xhigh;
        }
    }
    for (messages) |m| {
        if (m.role == .assistant) continue;
        const c = m.content;
        if (has(c, "<|think_off|>")) {
            thinking = false;
        } else if (has(c, "<|think_on|>")) {
            thinking = true;
        } else if (has(c, "<|think_xhigh|>") or has(c, "<|think_high|>") or has(c, "<|think_ultracode|>") or has(c, "<|think_extreme|>") or has(c, "<|think_max|>")) {
            thinking = true;
            effort = .xhigh;
        } else if (has(c, "<|think_low|>") or has(c, "<|think_minimal|>")) {
            thinking = true;
            effort = .low;
        } else if (has(c, "<|think_medium|>")) {
            thinking = true;
            effort = .medium;
        }
    }
    const instructions: []const u8 = if (!thinking) "" else switch (effort) {
        .xhigh => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
        .low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
        .medium => "",
    };

    // Leading system and developer messages merge into one system turn.
    var head: usize = 0;
    while (head < messages.len and isSystem(messages[head].role)) head += 1;
    var system: std.ArrayList(u8) = .empty;
    for (messages[0..head]) |m| {
        const part = try stripTags(arena, trim(m.content));
        if (part.len == 0) continue;
        if (system.items.len > 0) try system.appendSlice(arena, "\n\n");
        try system.appendSlice(arena, part);
    }
    if (system.items.len > 0) {
        try out.appendSlice(gpa, "<|im_start|>system\n");
        if (instructions.len > 0) try out.print(gpa, "{s}\n\n", .{instructions});
        try out.print(gpa, "{s}<|im_end|>\n", .{system.items});
    } else if (instructions.len > 0) {
        try out.print(gpa, "<|im_start|>system\n{s}<|im_end|>\n", .{instructions});
    }

    const rest = messages[head..];
    // Assistant reasoning after the last real user query is always kept.
    // Without tools every user message is a real query.
    const last_index: usize = if (rest.len > 0) rest.len - 1 else 0;
    var last_query: usize = if (last_index > 50) last_index else 0;
    var i = rest.len;
    while (i > 0) {
        i -= 1;
        if (rest[i].role == .user) {
            const c = trim(rest[i].content);
            if (!(std.mem.startsWith(u8, c, "<tool_response>") and std.mem.endsWith(u8, c, "</tool_response>"))) {
                last_query = i;
                break;
            }
        }
    }

    for (rest, 0..) |m, index| {
        var content = trim(m.content);
        switch (m.role) {
            .system, .developer => {
                content = try stripTags(arena, content);
                try out.print(gpa, "<|im_start|>system\n{s}<|im_end|>\n", .{content});
            },
            .user => {
                content = try stripTags(arena, content);
                try out.print(gpa, "<|im_start|>user\n{s}<|im_end|>\n", .{content});
            },
            .assistant => {
                var reasoning: []const u8 = "";
                if (m.reasoning != null and m.reasoning.?.len > 0) {
                    const lead: ?[]const u8 = if (std.mem.startsWith(u8, content, "<think>") and has(content, "</think>"))
                        "</think>"
                    else if (std.mem.startsWith(u8, content, "<thinking>") and has(content, "</thinking>"))
                        "</thinking>"
                    else if (std.mem.startsWith(u8, content, "</think>"))
                        "</think>"
                    else if (std.mem.startsWith(u8, content, "</thinking>"))
                        "</thinking>"
                    else
                        null;
                    if (lead) |end| content = lstripNewlines(afterLast(content, end));
                    reasoning = m.reasoning.?;
                } else {
                    const end: ?[]const u8 = if (std.mem.startsWith(u8, content, "</think>"))
                        "</think>"
                    else if (std.mem.startsWith(u8, content, "</thinking>"))
                        "</thinking>"
                    else if (has(content, "\n</think>"))
                        "\n</think>"
                    else if (has(content, "\n</thinking>"))
                        "\n</thinking>"
                    else if (has(content, "\n</ think>"))
                        "\n</ think>"
                    else if (has(content, "\n</think >"))
                        "\n</think >"
                    else if (std.mem.startsWith(u8, content, "<think>") and has(content, "</think>"))
                        "</think>"
                    else if (std.mem.startsWith(u8, content, "<thinking>") and has(content, "</thinking>"))
                        "</thinking>"
                    else
                        null;
                    if (end) |e| {
                        const start: []const u8 = if (has(e, "thinking")) "<thinking>" else "<think>";
                        reasoning = std.mem.trimEnd(u8, beforeFirst(content, e), "\n");
                        if (has(reasoning, start)) reasoning = lstripNewlines(afterLast(reasoning, start));
                        content = lstripNewlines(afterLast(content, e));
                    }
                }
                reasoning = trim(reasoning);
                if (opts.preserve_thinking or index > last_query) {
                    try out.print(gpa, "<|im_start|>assistant\n<think>\n{s}\n</think>\n\n{s}", .{ reasoning, content });
                } else {
                    try out.print(gpa, "<|im_start|>assistant\n{s}", .{content});
                }
                try out.appendSlice(gpa, "<|im_end|>\n");
            },
        }
    }

    if (opts.add_generation_prompt) {
        try out.appendSlice(gpa, "<|im_start|>assistant\n");
        try out.appendSlice(gpa, if (thinking) "<think>\n" else "<think>\n\n</think>\n\n");
    }
}

fn isSystem(r: Role) bool {
    return r == .system or r == .developer;
}

fn has(s: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, s, needle) != null;
}

fn oneOf(s: []const u8, options: []const []const u8) bool {
    for (options) |o| if (std.mem.eql(u8, s, o)) return true;
    return false;
}

/// Python's s.split(sep)[-1].
fn afterLast(s: []const u8, sep: []const u8) []const u8 {
    const i = std.mem.lastIndexOf(u8, s, sep) orelse return s;
    return s[i + sep.len ..];
}

/// Python's s.split(sep)[0].
fn beforeFirst(s: []const u8, sep: []const u8) []const u8 {
    const i = std.mem.indexOf(u8, s, sep) orelse return s;
    return s[0..i];
}

fn lstripNewlines(s: []const u8) []const u8 {
    return std.mem.trimStart(u8, s, "\n");
}

/// Removes every think tag, trimming after each as the template does.
fn stripTags(arena: Allocator, s: []const u8) Allocator.Error![]const u8 {
    var cur = s;
    for (tags) |tag| {
        if (!has(cur, tag)) continue;
        const joined = try std.mem.replaceOwned(u8, arena, cur, tag, "");
        cur = trim(joined);
    }
    return cur;
}

/// Jinja's `trim` filter: Python's str.strip(), which strips Unicode
/// whitespace (str.isspace), not just ASCII.
pub fn trim(s: []const u8) []const u8 {
    var start: usize = 0;
    while (start < s.len) {
        const n = spaceLen(s[start..]);
        if (n == 0) break;
        start += n;
    }
    var end = s.len;
    while (end > start) {
        // Step back to the start of the last code point.
        var j = end - 1;
        while (j > start and s[j] & 0xc0 == 0x80 and end - j < 4) j -= 1;
        if (spaceLen(s[j..end]) != end - j) break;
        end = j;
    }
    return s[start..end];
}

/// Byte length of the whitespace code point `s` starts with, or 0.
fn spaceLen(s: []const u8) usize {
    const n = std.unicode.utf8ByteSequenceLength(s[0]) catch return 0;
    if (n > s.len) return 0;
    const cp = std.unicode.utf8Decode(s[0..n]) catch return 0;
    return if (isSpace(cp)) n else 0;
}

fn isSpace(cp: u21) bool {
    return switch (cp) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

test "a two-turn conversation keeps the first reply's reasoning" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try render(gpa, &.{
        .{ .role = .system, .content = "Be brief." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "<think>\nGreet back.\n</think>\n\nHello!" },
        .{ .role = .user, .content = " How are you? " },
    }, .{}, &out);
    try std.testing.expectEqualStrings(
        "<|im_start|>system\nBe brief.<|im_end|>\n" ++
            "<|im_start|>user\nHi<|im_end|>\n" ++
            "<|im_start|>assistant\n<think>\nGreet back.\n</think>\n\nHello!<|im_end|>\n" ++
            "<|im_start|>user\nHow are you?<|im_end|>\n" ++
            "<|im_start|>assistant\n<think>\n",
        out.items,
    );
}

test "think tags switch thinking off and are stripped" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try render(gpa, &.{.{ .role = .user, .content = "<|think_off|> Hi" }}, .{}, &out);
    try std.testing.expectEqualStrings("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", out.items);
}

test "trim strips Unicode whitespace like Python" {
    try std.testing.expectEqualStrings("a b", trim("\u{3000} \ta b\u{a0}\n"));
    try std.testing.expectEqualStrings("", trim(" \u{2028} "));
    try std.testing.expectEqualStrings("€", trim("€"));
}
