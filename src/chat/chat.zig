//! Chat prompts in the fixed Qwen template: froggeric/Qwen-Fixed-Chat-Templates
//! @855bffc49448e299789730ff92c9b8d834d6cc14, chat_template.jinja
//! ("qwen3.8-froggeric-v22.5"), transcribed for text conversations with its
//! default options: tools are offered and called in its XML format, and
//! neither tool calls nor tool responses are truncated. Images are not
//! supported. tools/check_chat.py checks it against the Jinja template
//! rendered as Hugging Face transformers renders it.
//!
//! Tools and tool-call arguments are JSON values written as transformers'
//! `tojson` writes them (Python's json.dumps, keys in order, non-ASCII
//! kept). Numbers are written as given (parse with `.parse_numbers =
//! false`), which matches Python except for spellings Python normalizes,
//! such as 1E2 or 1.50.
//!
//! Earlier assistant turns keep their reasoning (preserve_thinking defaults
//! to true), so each new prompt extends the previous one token for token and
//! the engine reuses the whole conversation as a prefix.

const std = @import("std");
const Allocator = std.mem.Allocator;
pub const jinja = @import("jinja.zig");
const gguf = @import("../gguf/gguf.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");

/// Whether `render` writes the prompt format of models of GGUF
/// architecture `arch`: Qwen 3.5–3.8 and the Gyro-Coder use this template;
/// other architectures need their own.
pub fn supports(arch: []const u8) bool {
    return std.mem.eql(u8, arch, "qwen35") or std.mem.eql(u8, arch, "qwen4exp");
}

/// A model's own chat template (GGUF `tokenizer.chat_template`), rendered by
/// zllm.chat.jinja as transformers' apply_chat_template renders it. For the
/// models `supports` does not cover.
pub const Template = struct {
    tmpl: jinja.Template,
    bos: ?[]const u8,
    eos: ?[]const u8,

    /// Parses `source`, which the template borrows, from `arena`. `bos` and
    /// `eos` are the special tokens' text (transformers' bos_token, eos_token).
    pub fn parse(arena: Allocator, source: []const u8, bos: ?[]const u8, eos: ?[]const u8) jinja.Error!Template {
        return .{ .tmpl = try jinja.Template.parse(arena, source), .bos = bos, .eos = eos };
    }

    pub const Input = struct {
        /// The messages as the request gives them (a JSON array).
        messages: std.json.Value,
        /// The functions offered (a JSON array), or null.
        tools: ?std.json.Value = null,
        add_generation_prompt: bool = true,
        /// More keyword arguments for the template (enable_thinking,
        /// reasoning_effort, date_string, ...), as vLLM's chat_template_kwargs.
        kwargs: ?std.json.ObjectMap = null,
        /// The time strftime_now formats; null leaves it undefined.
        now: ?jinja.Date = null,
    };

    /// The template `file` carries (copied into `arena`: the header may not
    /// outlive the load), with `tok`'s special tokens; null if it has none.
    pub fn fromGguf(arena: Allocator, file: gguf.File, tok: *const tokenizer.Tokenizer) jinja.Error!?Template {
        const src = file.get("tokenizer.chat_template") orelse return null;
        if (src != .string) return null;
        return try parse(
            arena,
            try arena.dupe(u8, src.string),
            if (tok.bos) |id| tok.tokens[id] else null,
            if (tok.eos) |id| tok.tokens[id] else null,
        );
    }

    /// Renders `in` into `out`, from `arena`. On error.Raised, `raised` has
    /// the template's raise_exception message.
    pub fn render(t: Template, arena: Allocator, in: Input, out: *std.ArrayList(u8), raised: *?[]const u8) jinja.Error!void {
        var globals: std.ArrayList(jinja.Template.Global) = .empty;
        try globals.append(arena, .{ .name = "messages", .value = try jinja.fromJson(arena, in.messages) });
        // transformers always passes tools and documents, None when absent.
        try globals.append(arena, .{ .name = "tools", .value = if (in.tools) |v| try jinja.fromJson(arena, v) else .none });
        try globals.append(arena, .{ .name = "documents", .value = .none });
        try globals.append(arena, .{ .name = "add_generation_prompt", .value = .{ .boolean = in.add_generation_prompt } });
        if (t.bos) |s| try globals.append(arena, .{ .name = "bos_token", .value = .{ .string = s } });
        if (t.eos) |s| try globals.append(arena, .{ .name = "eos_token", .value = .{ .string = s } });
        if (in.kwargs) |kw| {
            var it = kw.iterator();
            while (it.next()) |e| try globals.append(arena, .{ .name = e.key_ptr.*, .value = try jinja.fromJson(arena, e.value_ptr.*) });
        }
        try t.tmpl.render(arena, globals.items, in.now, out, raised);
    }
};

pub const Role = enum { system, developer, user, assistant, tool };

/// A call an assistant turn made, as in the OpenAI and transformers formats.
pub const ToolCall = struct {
    function: struct {
        name: []const u8,
        /// An object: each member becomes a parameter.
        arguments: ?std.json.Value = null,
    },
};

pub const Message = struct {
    role: Role,
    content: []const u8 = "",
    /// An assistant turn's reasoning, if kept apart from `content`
    /// (reasoning_content). Otherwise a `<think>` block inside `content` is
    /// split off.
    reasoning: ?[]const u8 = null,
    /// An assistant turn's calls.
    tool_calls: []const ToolCall = &.{},
};

pub const Options = struct {
    enable_thinking: bool = true,
    /// none, off, minimal, low, medium, high, xhigh, max, ultracode, extreme.
    reasoning_effort: ?[]const u8 = null,
    preserve_thinking: bool = true,
    add_generation_prompt: bool = true,
    /// The functions offered, each as the JSON the template lists, e.g.
    /// {"type": "function", "function": {"name": ..., "parameters": ...}}.
    tools: []const std.json.Value = &.{},
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
        if (!(isSystem(m.role) or m.role == .user)) continue;
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
    if (opts.tools.len > 0) {
        try out.appendSlice(gpa, "<|im_start|>system\n");
        if (instructions.len > 0) try out.print(gpa, "{s}\n\n", .{instructions});
        try out.appendSlice(gpa, "# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (opts.tools) |t| {
            try out.append(gpa, '\n');
            try writeJson(gpa, out, t);
        }
        try out.appendSlice(gpa, "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n");
        if (thinking) try out.appendSlice(gpa, "<think>\nBrief explanation of tool call\n</think>\n");
        try out.appendSlice(gpa, "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n");
        if (thinking) try out.appendSlice(gpa, "- You can use the <think></think> block to plan your next tool call OR to synthesize data and formulate your final response to the user.\n- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.\n");
        try out.appendSlice(gpa, "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.\n");
        try out.appendSlice(gpa, if (thinking)
            "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY after thinking, with NO conversational text before it.\n"
        else
            "- If you choose to call a tool, you MUST output the <tool_call> block IMMEDIATELY, with NO conversational text before it.\n");
        try out.appendSlice(gpa, "- The <tool_call> and <function> tags MUST be at the very beginning of a new line, with NO spaces or indentation before them.\n- To call multiple functions, output a separate, completely closed <tool_call></tool_call> block for EACH function. Do NOT nest <tool_call> blocks.\n- If you have all necessary data, provide your final answer directly to the user without any tool call.\n</IMPORTANT>");
        if (system.items.len > 0) try out.print(gpa, "\n\n{s}", .{system.items});
        try out.appendSlice(gpa, "<|im_end|>\n");
    } else if (system.items.len > 0) {
        try out.appendSlice(gpa, "<|im_start|>system\n");
        if (instructions.len > 0) try out.print(gpa, "{s}\n\n", .{instructions});
        try out.print(gpa, "{s}<|im_end|>\n", .{system.items});
    } else if (instructions.len > 0) {
        try out.print(gpa, "<|im_start|>system\n{s}<|im_end|>\n", .{instructions});
    }

    const rest = messages[head..];
    // Assistant reasoning after the last real user query is always kept. A
    // user message that is only a tool response is not a real query.
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

    var failures: u32 = 0;
    for (rest, 0..) |m, index| {
        var content = trim(m.content);
        switch (m.role) {
            .system, .developer => {
                content = try stripTags(arena, content);
                try out.print(gpa, "<|im_start|>system\n{s}<|im_end|>\n", .{content});
            },
            .user => {
                failures = 0;
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
                for (m.tool_calls, 0..) |call, ci| {
                    const lead: []const u8 = if (ci > 0) "\n" else if (trim(content).len > 0) "\n\n" else "";
                    try out.print(gpa, "{s}<tool_call>\n<function={s}>\n", .{ lead, call.function.name });
                    if (call.function.arguments) |args| switch (args) {
                        .object => |o| for (o.keys(), o.values()) |k, v| {
                            try out.print(gpa, "<parameter={s}>\n", .{k});
                            if (v == .string) try out.appendSlice(gpa, v.string) else try writeJson(gpa, out, v);
                            try out.appendSlice(gpa, "\n</parameter>\n");
                        },
                        .string => |s| try out.appendSlice(gpa, s),
                        else => try writeJson(gpa, out, args),
                    };
                    try out.appendSlice(gpa, "</function>\n</tool_call>");
                }
                try out.appendSlice(gpa, "<|im_end|>\n");
            },
            .tool => {
                failures = if (try looksFailed(arena, content)) failures + 1 else 0;
                if (index == 0 or rest[index - 1].role != .tool) try out.appendSlice(gpa, "<|im_start|>user");
                try out.print(gpa, "\n<tool_response>\n{s}", .{content});
                if (failures >= 2) {
                    try out.print(gpa, "\n\n⚠️ SYSTEM WARNING: {d} consecutive tool errors detected. Your previous approach is incorrect. You MUST use a fundamentally different approach or corrected arguments.", .{failures});
                } else if (failures == 1) {
                    try out.appendSlice(gpa, "\n\n⚠️ SYSTEM WARNING: The previous tool call returned an error. Diagnose the failure and retry with completely corrected arguments.");
                }
                try out.appendSlice(gpa, "\n</tool_response>");
                if (index + 1 == rest.len or rest[index + 1].role != .tool) try out.appendSlice(gpa, "<|im_end|>\n");
            },
        }
    }

    if (opts.add_generation_prompt) {
        try out.appendSlice(gpa, "<|im_start|>assistant\n");
        try out.appendSlice(gpa, if (thinking) "<think>\n" else "<think>\n\n</think>\n\n");
    }
}

/// The template's guess that a tool response reports an error: a strong
/// sign, or a weak one in a short response without command output, unless
/// it reads as code (which mentions errors without being one). It looks at
/// the content lowercased, and mostly at its first 120 characters.
/// (Lowercasing here is ASCII; Python's also maps a few non-ASCII letters,
/// such as the Kelvin sign, to ASCII ones.)
fn looksFailed(arena: Allocator, content: []const u8) Allocator.Error!bool {
    const all = try std.ascii.allocLowerString(arena, content);
    var end: usize = 0;
    var chars: usize = 0;
    while (end < all.len and chars < 120) : (chars += 1) end += std.unicode.utf8ByteSequenceLength(all[end]) catch 1;
    const head = all[0..@min(end, all.len)];
    const code = has(all, "throw new ") or has(all, "throw error") or has(all, "console.error") or
        has(all, "logger.error") or has(all, "logging.error") or
        has(head, "import ") or has(head, "def ") or has(head, "function ");
    const exit_zero = has(head, "exit code: 0") or has(head, "process exited with code 0");
    const error_ok = has(head, "\"error\": null") or has(head, "\"error\":null") or has(head, "\"error\": false") or
        has(head, "\"error\":false") or has(head, "\"error\": \"\"") or has(head, "\"error\":\"\"");
    const strong = (has(head, "\"error\":") and !error_ok) or has(head, "\"status\": \"error\"") or has(head, "\"status\":\"error\"") or
        has(head, "traceback (most recent call last):") or has(head, "command not found") or has(head, "invalid syntax") or
        has(head, "fatal:") or ((has(head, "exit code: ") or has(head, "process exited with code")) and !exit_zero) or
        std.mem.startsWith(u8, head, "exception:") or std.mem.startsWith(u8, head, "failed to ");
    const weak = has(head, "error:") or has(head, "err!");
    const suppressed = has(head, "$ ") or has(head, "took ") or (std.unicode.utf8CountCodepoints(content) catch content.len) >= 600;
    return !code and (strong or (weak and !suppressed));
}

/// `v` as transformers' `tojson` writes it: Python's json.dumps with
/// ensure_ascii=False and the default ", " and ": " separators.
pub fn writeJson(gpa: Allocator, out: *std.ArrayList(u8), v: std.json.Value) Allocator.Error!void {
    switch (v) {
        .null => try out.appendSlice(gpa, "null"),
        .bool => |b| try out.appendSlice(gpa, if (b) "true" else "false"),
        .integer => |i| try out.print(gpa, "{d}", .{i}),
        .float => |f| try out.print(gpa, "{d}", .{f}),
        .number_string => |s| try out.appendSlice(gpa, s),
        .string => |s| try writeJsonString(gpa, out, s),
        .array => |a| {
            try out.append(gpa, '[');
            for (a.items, 0..) |item, i| {
                if (i > 0) try out.appendSlice(gpa, ", ");
                try writeJson(gpa, out, item);
            }
            try out.append(gpa, ']');
        },
        .object => |o| {
            try out.append(gpa, '{');
            for (o.keys(), o.values(), 0..) |k, item, i| {
                if (i > 0) try out.appendSlice(gpa, ", ");
                try writeJsonString(gpa, out, k);
                try out.appendSlice(gpa, ": ");
                try writeJson(gpa, out, item);
            }
            try out.append(gpa, '}');
        },
    }
}

fn writeJsonString(gpa: Allocator, out: *std.ArrayList(u8), s: []const u8) Allocator.Error!void {
    try out.append(gpa, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(gpa, "\\\""),
        '\\' => try out.appendSlice(gpa, "\\\\"),
        '\n' => try out.appendSlice(gpa, "\\n"),
        '\r' => try out.appendSlice(gpa, "\\r"),
        '\t' => try out.appendSlice(gpa, "\\t"),
        0x08 => try out.appendSlice(gpa, "\\b"),
        0x0c => try out.appendSlice(gpa, "\\f"),
        0...0x07, 0x0b, 0x0e...0x1f => try out.print(gpa, "\\u{x:0>4}", .{c}),
        else => try out.append(gpa, c),
    };
    try out.append(gpa, '"');
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
