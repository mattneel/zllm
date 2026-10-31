//! A Jinja2 subset for GGUF chat templates (`tokenizer.chat_template`),
//! rendering as Hugging Face transformers' `apply_chat_template` does: an
//! immutable sandboxed environment with trim_blocks and lstrip_blocks, the
//! loop-controls extension, `raise_exception`, `strftime_now` and
//! transformers' `tojson`. Values behave as Python's: `str()`, truthiness,
//! equality, floor division, and strings indexed by code point.
//!
//! What it does not know (a filter, test, method or statement outside the
//! subset) is an error, never skipped: a template renders exactly or not at
//! all. tools/check_chat.py checks it against jinja2.
//!
//! JSON numbers from requests stay as written (`Value.number`), printed and
//! serialized as given, like zllm.chat's fixed template.

const std = @import("std");
const Allocator = std.mem.Allocator;
const unicode = @import("../tokenizer/unicode.zig");

pub const Error = Allocator.Error || error{
    /// The template does not parse.
    TemplateSyntax,
    /// A Python or Jinja runtime error (a type error, an unknown filter, an
    /// attribute of an undefined value).
    TemplateError,
    /// The template called raise_exception (Rendered.raised has its message).
    Raised,
};

pub const Value = union(enum) {
    undefined,
    none,
    boolean: bool,
    integer: i64,
    float: f64,
    /// A JSON number as written.
    number: []const u8,
    string: []const u8,
    list: []const Value,
    dict: *Dict,
    /// jinja2's namespace(): attributes a template may set.
    namespace: *Dict,
    callable: *const Callable,

    pub fn str(s: []const u8) Value {
        return .{ .string = s };
    }
};

/// An ordered string-keyed map, as Python's dict.
pub const Dict = struct {
    keys: std.ArrayList([]const u8) = .empty,
    values: std.ArrayList(Value) = .empty,

    pub fn get(d: *const Dict, key: []const u8) ?Value {
        for (d.keys.items, 0..) |k, i| if (std.mem.eql(u8, k, key)) return d.values.items[i];
        return null;
    }

    pub fn put(d: *Dict, arena: Allocator, key: []const u8, v: Value) Allocator.Error!void {
        for (d.keys.items, 0..) |k, i| if (std.mem.eql(u8, k, key)) {
            d.values.items[i] = v;
            return;
        };
        try d.keys.append(arena, key);
        try d.values.append(arena, v);
    }
};

pub const Callable = union(enum) {
    macro: *const Macro,
    builtin: Builtin,
    /// A string or dict method bound to its object.
    method: struct { self: Value, name: []const u8 },
};

const Builtin = enum { namespace, range, dict, raise_exception, strftime_now };

/// A calendar time for strftime_now.
pub const Date = struct {
    year: u32,
    month: u8, // 1-12
    day: u8, // 1-31
    hour: u8 = 0,
    minute: u8 = 0,
    second: u8 = 0,
    weekday: u8 = 0, // 0 = Monday

    /// The UTC date of Unix time `seconds`.
    pub fn fromUnix(seconds: i64) Date {
        const s: u64 = @intCast(@max(seconds, 0));
        const es: std.time.epoch.EpochSeconds = .{ .secs = s };
        const day = es.getEpochDay();
        const yd = day.calculateYearDay();
        const md = yd.calculateMonthDay();
        const ds = es.getDaySeconds();
        return .{
            .year = yd.year,
            .month = md.month.numeric(),
            .day = md.day_index + 1,
            .hour = ds.getHoursIntoDay(),
            .minute = ds.getMinutesIntoHour(),
            .second = ds.getSecondsIntoMinute(),
            // 1970-01-01 was a Thursday.
            .weekday = @intCast((day.day + 3) % 7),
        };
    }
};

// ---------------------------------------------------------------- syntax

const Expr = union(enum) {
    literal: Value,
    name: []const u8,
    list: []const *const Expr,
    dict: []const Pair,
    attr: struct { obj: *const Expr, name: []const u8 },
    item: struct { obj: *const Expr, index: *const Expr },
    slice: struct { obj: *const Expr, start: ?*const Expr, stop: ?*const Expr, step: ?*const Expr },
    call: Call,
    filter: struct { value: *const Expr, name: []const u8, args: Args },
    @"test": struct { value: *const Expr, name: []const u8, args: Args, negate: bool },
    neg: *const Expr,
    not_: *const Expr,
    binary: struct { op: BinOp, l: *const Expr, r: *const Expr },
    cond: struct { cond: *const Expr, then: *const Expr, otherwise: ?*const Expr },
};

const Pair = struct { key: *const Expr, value: *const Expr };
const Kwarg = struct { name: []const u8, value: *const Expr };
const Args = struct { pos: []const *const Expr = &.{}, kw: []const Kwarg = &.{} };
const Call = struct { callee: *const Expr, args: Args };

const BinOp = enum { add, sub, mul, div, floordiv, mod, pow, concat, eq, ne, lt, le, gt, ge, in, not_in, and_, or_ };

const Node = union(enum) {
    text: []const u8,
    output: *const Expr,
    if_: []const Branch,
    for_: *const For,
    set: *const Set,
    macro: *const Macro,
    break_,
    continue_,
    /// {% generation %}: its body, rendered as is.
    block: []const Node,
};

const Branch = struct { cond: ?*const Expr, body: []const Node };
const For = struct {
    targets: []const []const u8,
    iter: *const Expr,
    filter: ?*const Expr,
    body: []const Node,
    otherwise: []const Node,
};
const Target = struct { name: []const u8, attr: ?[]const u8 = null };
const Set = struct { targets: []const Target, value: ?*const Expr, body: []const Node };
pub const Macro = struct {
    name: []const u8,
    params: []const Param,
    body: []const Node,
};
const Param = struct { name: []const u8, default: ?*const Expr };

const TokKind = enum { name, string, integer, float, op };
const Tok = struct { kind: TokKind, text: []const u8 };

const TagKind = enum { output, block, comment };
const Seg = union(enum) {
    text: []const u8,
    tag: struct { kind: TagKind, toks: []const Tok, minus_left: bool, plus_left: bool, minus_right: bool },
};

pub const Template = struct {
    nodes: []const Node,

    /// Parses `source`; the template borrows it and allocates from `arena`.
    pub fn parse(arena: Allocator, source: []const u8) Error!Template {
        const segs = try segment(arena, source);
        var p: Parser = .{ .arena = arena, .segs = segs };
        const nodes = try p.body(&.{});
        if (p.i != segs.len) return error.TemplateSyntax;
        return .{ .nodes = nodes };
    }

    pub const Global = struct { name: []const u8, value: Value };

    /// Renders with `globals` (the template's variables) into `out`, from
    /// `arena`. `now` defines strftime_now. On error.Raised, `raised` holds
    /// raise_exception's message.
    pub fn render(t: Template, arena: Allocator, globals: []const Global, now: ?Date, out: *std.ArrayList(u8), raised: *?[]const u8) Error!void {
        var g = try arena.create(Dict);
        g.* = .{};
        for (globals) |v| try g.put(arena, v.name, v.value);
        var r: Renderer = .{ .arena = arena, .globals = g, .now = now, .raised = raised };
        _ = try r.nodes(t.nodes, out);
    }
};

/// Splits `src` into text and tags, applying jinja2's whitespace rules:
/// `-` strips all whitespace on its side; trim_blocks drops the newline
/// after a block or comment tag; lstrip_blocks drops spaces and tabs before
/// one when only they precede it on its line; newlines become "\n" and one
/// trailing newline is dropped (keep_trailing_newline off).
fn segment(arena: Allocator, raw: []const u8) Error![]Seg {
    const src = try normalizeNewlines(arena, raw);
    var segs: std.ArrayList(Seg) = .empty;
    var i: usize = 0;
    var text_start: usize = 0;
    while (i + 1 < src.len) {
        if (src[i] != '{' or (src[i + 1] != '{' and src[i + 1] != '%' and src[i + 1] != '#')) {
            i += 1;
            continue;
        }
        try segs.append(arena, .{ .text = src[text_start..i] });
        const kind: TagKind = switch (src[i + 1]) {
            '{' => .output,
            '%' => .block,
            else => .comment,
        };
        var j = i + 2;
        const minus_left = j < src.len and src[j] == '-';
        const plus_left = j < src.len and src[j] == '+';
        if (minus_left or plus_left) j += 1;
        if (kind == .comment) {
            const end = std.mem.indexOfPos(u8, src, j, "#}") orelse return error.TemplateSyntax;
            const minus_right = end > j and src[end - 1] == '-';
            try segs.append(arena, .{ .tag = .{ .kind = .comment, .toks = &.{}, .minus_left = minus_left, .plus_left = plus_left, .minus_right = minus_right } });
            i = end + 2;
        } else {
            var lex: Lexer = .{ .arena = arena, .src = src, .i = j, .close = if (kind == .output) "}}" else "%}" };
            const toks = try lex.tag();
            if (kind == .block and toks.len > 0 and std.mem.eql(u8, toks[0].text, "raw")) return error.TemplateSyntax;
            try segs.append(arena, .{ .tag = .{ .kind = kind, .toks = toks, .minus_left = minus_left, .plus_left = plus_left, .minus_right = lex.minus_right } });
            i = lex.i;
        }
        text_start = i;
    }
    try segs.append(arena, .{ .text = src[text_start..] });

    // Whitespace control, tag by tag; texts sit at even indices.
    var line_start = true;
    var k: usize = 1;
    while (k < segs.items.len) : (k += 2) {
        const tag = segs.items[k].tag;
        const before = &segs.items[k - 1].text;
        if (tag.minus_left) {
            before.* = std.mem.trimEnd(u8, before.*, " \t\n\r\x0b\x0c");
        } else if (tag.kind != .output and !tag.plus_left) {
            const nl = if (std.mem.lastIndexOfScalar(u8, before.*, '\n')) |n| n + 1 else 0;
            if ((nl > 0 or line_start) and std.mem.trimStart(u8, before.*[nl..], " \t").len == 0) before.* = before.*[0..nl];
        }
        const after = &segs.items[k + 1].text;
        line_start = false;
        if (tag.minus_right) {
            const kept = std.mem.trimStart(u8, after.*, " \t\n\r\x0b\x0c");
            line_start = kept.len < after.len and after.*[after.len - kept.len - 1] == '\n';
            after.* = kept;
        } else if (tag.kind != .output and after.len > 0 and after.*[0] == '\n') {
            after.* = after.*[1..];
            line_start = true;
        }
    }
    const last = &segs.items[segs.items.len - 1].text;
    if (last.len > 0 and last.*[last.len - 1] == '\n') last.* = last.*[0 .. last.len - 1];
    return segs.items;
}

fn normalizeNewlines(arena: Allocator, s: []const u8) Allocator.Error![]const u8 {
    if (std.mem.indexOfScalar(u8, s, '\r') == null) return s;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\r') {
            try out.append(arena, '\n');
            if (i + 1 < s.len and s[i + 1] == '\n') i += 1;
        } else try out.append(arena, s[i]);
    }
    return out.items;
}

const Lexer = struct {
    arena: Allocator,
    src: []const u8,
    i: usize,
    close: []const u8,
    minus_right: bool = false,

    /// Tokens up to the tag's close (outside brackets), which it consumes.
    fn tag(l: *Lexer) Error![]const Tok {
        var toks: std.ArrayList(Tok) = .empty;
        var depth: usize = 0;
        const ops = [_][]const u8{ "**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">", "(", ")", "[", "]", "{", "}", ",", ".", ":", "|", "=" };
        while (true) {
            while (l.i < l.src.len and std.ascii.isWhitespace(l.src[l.i])) l.i += 1;
            if (l.i >= l.src.len) return error.TemplateSyntax;
            const rest = l.src[l.i..];
            if (depth == 0) {
                if (std.mem.startsWith(u8, rest, l.close)) {
                    l.i += 2;
                    return toks.items;
                }
                if ((rest[0] == '-' or rest[0] == '+') and std.mem.startsWith(u8, rest[1..], l.close)) {
                    l.minus_right = rest[0] == '-';
                    l.i += 3;
                    return toks.items;
                }
            }
            const c = rest[0];
            if (std.ascii.isAlphabetic(c) or c == '_') {
                var n: usize = 1;
                while (n < rest.len and (std.ascii.isAlphanumeric(rest[n]) or rest[n] == '_')) n += 1;
                try toks.append(l.arena, .{ .kind = .name, .text = rest[0..n] });
                l.i += n;
            } else if (std.ascii.isDigit(c)) {
                var n: usize = 0;
                var float = false;
                while (n < rest.len and (std.ascii.isDigit(rest[n]) or rest[n] == '_')) n += 1;
                if (n + 1 < rest.len and rest[n] == '.' and std.ascii.isDigit(rest[n + 1])) {
                    float = true;
                    n += 1;
                    while (n < rest.len and std.ascii.isDigit(rest[n])) n += 1;
                }
                if (n < rest.len and (rest[n] == 'e' or rest[n] == 'E')) {
                    var m = n + 1;
                    if (m < rest.len and (rest[m] == '+' or rest[m] == '-')) m += 1;
                    if (m < rest.len and std.ascii.isDigit(rest[m])) {
                        float = true;
                        n = m;
                        while (n < rest.len and std.ascii.isDigit(rest[n])) n += 1;
                    }
                }
                try toks.append(l.arena, .{ .kind = if (float) .float else .integer, .text = rest[0..n] });
                l.i += n;
            } else if (c == '\'' or c == '"') {
                try toks.append(l.arena, .{ .kind = .string, .text = try l.string(c) });
            } else {
                const op = for (ops) |o| {
                    if (std.mem.startsWith(u8, rest, o)) break o;
                } else return error.TemplateSyntax;
                if (std.mem.eql(u8, op, "(") or std.mem.eql(u8, op, "[") or std.mem.eql(u8, op, "{")) depth += 1;
                if (std.mem.eql(u8, op, ")") or std.mem.eql(u8, op, "]") or std.mem.eql(u8, op, "}")) {
                    if (depth == 0) return error.TemplateSyntax;
                    depth -= 1;
                }
                try toks.append(l.arena, .{ .kind = .op, .text = op });
                l.i += op.len;
            }
        }
    }

    /// A quoted string with Python's escapes, decoded.
    fn string(l: *Lexer, quote: u8) Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        var i = l.i + 1;
        while (i < l.src.len and l.src[i] != quote) : (i += 1) {
            const c = l.src[i];
            if (c != '\\' or i + 1 >= l.src.len) {
                try out.append(l.arena, c);
                continue;
            }
            i += 1;
            const e = l.src[i];
            switch (e) {
                'n' => try out.append(l.arena, '\n'),
                't' => try out.append(l.arena, '\t'),
                'r' => try out.append(l.arena, '\r'),
                '0' => try out.append(l.arena, 0),
                '\\', '\'', '"' => try out.append(l.arena, e),
                'x', 'u', 'U' => {
                    const digits: usize = switch (e) {
                        'x' => 2,
                        'u' => 4,
                        else => 8,
                    };
                    if (i + digits >= l.src.len) return error.TemplateSyntax;
                    const cp = std.fmt.parseInt(u21, l.src[i + 1 ..][0..digits], 16) catch return error.TemplateSyntax;
                    var buf: [4]u8 = undefined;
                    const n = std.unicode.utf8Encode(cp, &buf) catch return error.TemplateSyntax;
                    try out.appendSlice(l.arena, buf[0..n]);
                    i += digits;
                },
                else => try out.appendSlice(l.arena, &.{ '\\', e }),
            }
        }
        if (i >= l.src.len) return error.TemplateSyntax;
        l.i = i + 1;
        return out.items;
    }
};

const Parser = struct {
    arena: Allocator,
    segs: []const Seg,
    i: usize = 0,
    // The tag being parsed.
    toks: []const Tok = &.{},
    t: usize = 0,

    /// Nodes up to a block tag whose first word is in `ends` (left unconsumed).
    fn body(p: *Parser, ends: []const []const u8) Error![]const Node {
        var nodes: std.ArrayList(Node) = .empty;
        while (p.i < p.segs.len) {
            switch (p.segs[p.i]) {
                .text => |s| {
                    if (s.len > 0) try nodes.append(p.arena, .{ .text = s });
                    p.i += 1;
                },
                .tag => |tag| switch (tag.kind) {
                    .comment => p.i += 1,
                    .output => {
                        p.toks = tag.toks;
                        p.t = 0;
                        const e = try p.expr();
                        try p.endOfTag();
                        try nodes.append(p.arena, .{ .output = e });
                        p.i += 1;
                    },
                    .block => {
                        if (tag.toks.len == 0) return error.TemplateSyntax;
                        const word = tag.toks[0].text;
                        for (ends) |e| if (std.mem.eql(u8, word, e)) return nodes.items;
                        p.toks = tag.toks;
                        p.t = 1;
                        p.i += 1;
                        try nodes.append(p.arena, try p.statement(word));
                    },
                },
            }
        }
        if (ends.len > 0) return error.TemplateSyntax;
        return nodes.items;
    }

    /// The block tag at segs[i] (an end tag), whose first word it returns.
    fn endTag(p: *Parser) Error![]const u8 {
        if (p.i >= p.segs.len) return error.TemplateSyntax;
        const tag = p.segs[p.i].tag;
        p.toks = tag.toks;
        p.t = 1;
        p.i += 1;
        return tag.toks[0].text;
    }

    fn statement(p: *Parser, word: []const u8) Error!Node {
        const a = p.arena;
        if (std.mem.eql(u8, word, "if")) {
            var branches: std.ArrayList(Branch) = .empty;
            var cond: ?*const Expr = try p.expr();
            try p.endOfTag();
            while (true) {
                const nodes = try p.body(&.{ "elif", "else", "endif" });
                try branches.append(a, .{ .cond = cond, .body = nodes });
                const end = try p.endTag();
                if (std.mem.eql(u8, end, "endif")) break;
                if (std.mem.eql(u8, end, "elif")) {
                    cond = try p.expr();
                } else {
                    cond = null;
                }
                try p.endOfTag();
            }
            try p.endOfTag();
            return .{ .if_ = branches.items };
        }
        if (std.mem.eql(u8, word, "for")) {
            var targets: std.ArrayList([]const u8) = .empty;
            try targets.append(a, try p.name());
            while (p.isOp(",")) {
                p.t += 1;
                try targets.append(a, try p.name());
            }
            if (!p.isName("in")) return error.TemplateSyntax;
            p.t += 1;
            const iter = try p.orExpr();
            var filter: ?*const Expr = null;
            if (p.isName("if")) {
                p.t += 1;
                filter = try p.expr();
            }
            try p.endOfTag();
            const f = try a.create(For);
            f.* = .{ .targets = targets.items, .iter = iter, .filter = filter, .body = try p.body(&.{ "else", "endfor" }), .otherwise = &.{} };
            if (std.mem.eql(u8, try p.endTag(), "else")) {
                try p.endOfTag();
                f.otherwise = try p.body(&.{"endfor"});
                _ = try p.endTag();
            }
            try p.endOfTag();
            return .{ .for_ = f };
        }
        if (std.mem.eql(u8, word, "set")) {
            var targets: std.ArrayList(Target) = .empty;
            while (true) {
                var t: Target = .{ .name = try p.name() };
                if (p.isOp(".")) {
                    p.t += 1;
                    t.attr = try p.name();
                }
                try targets.append(a, t);
                if (!p.isOp(",")) break;
                p.t += 1;
            }
            const s = try a.create(Set);
            if (p.isOp("=")) {
                p.t += 1;
                s.* = .{ .targets = targets.items, .value = try p.tupleExpr(), .body = &.{} };
                try p.endOfTag();
            } else {
                try p.endOfTag();
                s.* = .{ .targets = targets.items, .value = null, .body = try p.body(&.{"endset"}) };
                _ = try p.endTag();
                try p.endOfTag();
            }
            return .{ .set = s };
        }
        if (std.mem.eql(u8, word, "macro")) {
            const m = try a.create(Macro);
            const macro_name = try p.name();
            var params: std.ArrayList(Param) = .empty;
            try p.expectOp("(");
            while (!p.isOp(")")) {
                var param: Param = .{ .name = try p.name(), .default = null };
                if (p.isOp("=")) {
                    p.t += 1;
                    param.default = try p.expr();
                }
                try params.append(a, param);
                if (!p.isOp(",")) break;
                p.t += 1;
            }
            try p.expectOp(")");
            try p.endOfTag();
            m.* = .{ .name = macro_name, .params = params.items, .body = try p.body(&.{"endmacro"}) };
            _ = try p.endTag();
            if (p.t < p.toks.len and p.toks[p.t].kind == .name) p.t += 1; // {% endmacro name %}
            try p.endOfTag();
            return .{ .macro = m };
        }
        if (std.mem.eql(u8, word, "break") or std.mem.eql(u8, word, "continue")) {
            try p.endOfTag();
            return if (word[0] == 'b') .break_ else .continue_;
        }
        if (std.mem.eql(u8, word, "generation")) {
            try p.endOfTag();
            const nodes = try p.body(&.{"endgeneration"});
            _ = try p.endTag();
            try p.endOfTag();
            return .{ .block = nodes };
        }
        return error.TemplateSyntax;
    }

    fn endOfTag(p: *Parser) Error!void {
        if (p.t != p.toks.len) return error.TemplateSyntax;
    }

    fn peek(p: *Parser) ?Tok {
        return if (p.t < p.toks.len) p.toks[p.t] else null;
    }

    fn isOp(p: *Parser, op: []const u8) bool {
        const t = p.peek() orelse return false;
        return t.kind == .op and std.mem.eql(u8, t.text, op);
    }

    fn isName(p: *Parser, n: []const u8) bool {
        const t = p.peek() orelse return false;
        return t.kind == .name and std.mem.eql(u8, t.text, n);
    }

    fn expectOp(p: *Parser, op: []const u8) Error!void {
        if (!p.isOp(op)) return error.TemplateSyntax;
        p.t += 1;
    }

    fn name(p: *Parser) Error![]const u8 {
        const t = p.peek() orelse return error.TemplateSyntax;
        if (t.kind != .name) return error.TemplateSyntax;
        p.t += 1;
        return t.text;
    }

    fn new(p: *Parser, e: Expr) Error!*const Expr {
        const ptr = try p.arena.create(Expr);
        ptr.* = e;
        return ptr;
    }

    /// expr (',' expr)*: several make a tuple (a list).
    fn tupleExpr(p: *Parser) Error!*const Expr {
        const first = try p.expr();
        if (!p.isOp(",")) return first;
        var items: std.ArrayList(*const Expr) = .empty;
        try items.append(p.arena, first);
        while (p.isOp(",")) {
            p.t += 1;
            try items.append(p.arena, try p.expr());
        }
        return p.new(.{ .list = items.items });
    }

    fn expr(p: *Parser) Error!*const Expr {
        var e = try p.orExpr();
        while (p.isName("if")) {
            p.t += 1;
            const cond = try p.orExpr();
            var otherwise: ?*const Expr = null;
            if (p.isName("else")) {
                p.t += 1;
                otherwise = try p.expr();
            }
            e = try p.new(.{ .cond = .{ .cond = cond, .then = e, .otherwise = otherwise } });
        }
        return e;
    }

    fn orExpr(p: *Parser) Error!*const Expr {
        var e = try p.andExpr();
        while (p.isName("or")) {
            p.t += 1;
            e = try p.new(.{ .binary = .{ .op = .or_, .l = e, .r = try p.andExpr() } });
        }
        return e;
    }

    fn andExpr(p: *Parser) Error!*const Expr {
        var e = try p.notExpr();
        while (p.isName("and")) {
            p.t += 1;
            e = try p.new(.{ .binary = .{ .op = .and_, .l = e, .r = try p.notExpr() } });
        }
        return e;
    }

    fn notExpr(p: *Parser) Error!*const Expr {
        if (p.isName("not")) {
            p.t += 1;
            return p.new(.{ .not_ = try p.notExpr() });
        }
        return p.compare();
    }

    fn compare(p: *Parser) Error!*const Expr {
        var e = try p.math1();
        while (true) {
            const op: BinOp = if (p.isOp("==")) .eq else if (p.isOp("!=")) .ne else if (p.isOp("<")) .lt else if (p.isOp("<=")) .le else if (p.isOp(">")) .gt else if (p.isOp(">=")) .ge else if (p.isName("in")) .in else if (p.isName("not") and p.t + 1 < p.toks.len and p.toks[p.t + 1].kind == .name and std.mem.eql(u8, p.toks[p.t + 1].text, "in")) .not_in else break;
            p.t += if (op == .not_in) 2 else 1;
            e = try p.new(.{ .binary = .{ .op = op, .l = e, .r = try p.math1() } });
        }
        return e;
    }

    fn math1(p: *Parser) Error!*const Expr {
        var e = try p.concat();
        while (true) {
            const op: BinOp = if (p.isOp("+")) .add else if (p.isOp("-")) .sub else break;
            p.t += 1;
            e = try p.new(.{ .binary = .{ .op = op, .l = e, .r = try p.concat() } });
        }
        return e;
    }

    fn concat(p: *Parser) Error!*const Expr {
        var e = try p.math2();
        while (p.isOp("~")) {
            p.t += 1;
            e = try p.new(.{ .binary = .{ .op = .concat, .l = e, .r = try p.math2() } });
        }
        return e;
    }

    fn math2(p: *Parser) Error!*const Expr {
        var e = try p.pow();
        while (true) {
            const op: BinOp = if (p.isOp("*")) .mul else if (p.isOp("/")) .div else if (p.isOp("//")) .floordiv else if (p.isOp("%")) .mod else break;
            p.t += 1;
            e = try p.new(.{ .binary = .{ .op = op, .l = e, .r = try p.pow() } });
        }
        return e;
    }

    fn pow(p: *Parser) Error!*const Expr {
        var e = try p.unary(true);
        while (p.isOp("**")) {
            p.t += 1;
            e = try p.new(.{ .binary = .{ .op = .pow, .l = e, .r = try p.unary(true) } });
        }
        return e;
    }

    fn unary(p: *Parser, with_filter: bool) Error!*const Expr {
        var e = if (p.isOp("-")) blk: {
            p.t += 1;
            break :blk try p.new(.{ .neg = try p.unary(false) });
        } else if (p.isOp("+")) blk: {
            p.t += 1;
            break :blk try p.unary(false);
        } else try p.postfix(try p.primary());
        if (with_filter) e = try p.filterExpr(e);
        return e;
    }

    fn primary(p: *Parser) Error!*const Expr {
        const t = p.peek() orelse return error.TemplateSyntax;
        p.t += 1;
        switch (t.kind) {
            .name => {
                const lits = [_]struct { []const u8, Value }{
                    .{ "true", .{ .boolean = true } },   .{ "True", .{ .boolean = true } },
                    .{ "false", .{ .boolean = false } }, .{ "False", .{ .boolean = false } },
                    .{ "none", .none },                  .{ "None", .none },
                };
                for (lits) |l| if (std.mem.eql(u8, t.text, l[0])) return p.new(.{ .literal = l[1] });
                return p.new(.{ .name = t.text });
            },
            .string => {
                // Adjacent strings concatenate.
                var s = t.text;
                while (p.peek()) |n| {
                    if (n.kind != .string) break;
                    s = try std.mem.concat(p.arena, u8, &.{ s, n.text });
                    p.t += 1;
                }
                return p.new(.{ .literal = .{ .string = s } });
            },
            .integer => {
                var buf: std.ArrayList(u8) = .empty;
                for (t.text) |c| if (c != '_') try buf.append(p.arena, c);
                return p.new(.{ .literal = .{ .integer = std.fmt.parseInt(i64, buf.items, 10) catch return error.TemplateSyntax } });
            },
            .float => return p.new(.{ .literal = .{ .float = std.fmt.parseFloat(f64, t.text) catch return error.TemplateSyntax } }),
            .op => {
                if (std.mem.eql(u8, t.text, "(")) {
                    if (p.isOp(")")) {
                        p.t += 1;
                        return p.new(.{ .list = &.{} });
                    }
                    const e = try p.tupleExpr();
                    try p.expectOp(")");
                    return e;
                }
                if (std.mem.eql(u8, t.text, "[")) {
                    var items: std.ArrayList(*const Expr) = .empty;
                    while (!p.isOp("]")) {
                        try items.append(p.arena, try p.expr());
                        if (!p.isOp(",")) break;
                        p.t += 1;
                    }
                    try p.expectOp("]");
                    return p.new(.{ .list = items.items });
                }
                if (std.mem.eql(u8, t.text, "{")) {
                    var pairs: std.ArrayList(Pair) = .empty;
                    while (!p.isOp("}")) {
                        const k = try p.expr();
                        try p.expectOp(":");
                        try pairs.append(p.arena, .{ .key = k, .value = try p.expr() });
                        if (!p.isOp(",")) break;
                        p.t += 1;
                    }
                    try p.expectOp("}");
                    return p.new(.{ .dict = pairs.items });
                }
                return error.TemplateSyntax;
            },
        }
    }

    fn postfix(p: *Parser, start: *const Expr) Error!*const Expr {
        var e = start;
        while (true) {
            if (p.isOp(".")) {
                p.t += 1;
                const t = p.peek() orelse return error.TemplateSyntax;
                p.t += 1;
                e = switch (t.kind) {
                    .name => try p.new(.{ .attr = .{ .obj = e, .name = t.text } }),
                    .integer => try p.new(.{ .item = .{ .obj = e, .index = try p.new(.{ .literal = .{ .integer = std.fmt.parseInt(i64, t.text, 10) catch return error.TemplateSyntax } }) } }),
                    else => return error.TemplateSyntax,
                };
            } else if (p.isOp("[")) {
                p.t += 1;
                e = try p.subscript(e);
            } else if (p.isOp("(")) {
                p.t += 1;
                e = try p.new(.{ .call = .{ .callee = e, .args = try p.args() } });
            } else return e;
        }
    }

    fn subscript(p: *Parser, obj: *const Expr) Error!*const Expr {
        var parts: [3]?*const Expr = .{ null, null, null };
        var n: usize = 0;
        var colons: usize = 0;
        while (true) {
            if (p.isOp("]")) break;
            if (p.isOp(":")) {
                p.t += 1;
                colons += 1;
                n = colons;
                if (colons > 2) return error.TemplateSyntax;
                continue;
            }
            if (parts[n] != null) return error.TemplateSyntax;
            parts[n] = try p.expr();
        }
        try p.expectOp("]");
        if (colons == 0) return p.new(.{ .item = .{ .obj = obj, .index = parts[0] orelse return error.TemplateSyntax } });
        return p.new(.{ .slice = .{ .obj = obj, .start = parts[0], .stop = parts[1], .step = parts[2] } });
    }

    /// Call arguments after '(', through ')'.
    fn args(p: *Parser) Error!Args {
        var pos: std.ArrayList(*const Expr) = .empty;
        var kw: std.ArrayList(Kwarg) = .empty;
        while (!p.isOp(")")) {
            const t = p.peek() orelse return error.TemplateSyntax;
            if (t.kind == .name and p.t + 1 < p.toks.len and p.toks[p.t + 1].kind == .op and std.mem.eql(u8, p.toks[p.t + 1].text, "=")) {
                p.t += 2;
                try kw.append(p.arena, .{ .name = t.text, .value = try p.expr() });
            } else {
                if (kw.items.len > 0) return error.TemplateSyntax;
                try pos.append(p.arena, try p.expr());
            }
            if (!p.isOp(",")) break;
            p.t += 1;
        }
        try p.expectOp(")");
        return .{ .pos = pos.items, .kw = kw.items };
    }

    fn filterExpr(p: *Parser, start: *const Expr) Error!*const Expr {
        var e = start;
        while (true) {
            if (p.isOp("|")) {
                p.t += 1;
                const n = try p.name();
                var a: Args = .{};
                if (p.isOp("(")) {
                    p.t += 1;
                    a = try p.args();
                }
                e = try p.new(.{ .filter = .{ .value = e, .name = n, .args = a } });
            } else if (p.isName("is")) {
                p.t += 1;
                var negate = false;
                if (p.isName("not")) {
                    p.t += 1;
                    negate = true;
                }
                const n = try p.name();
                var a: Args = .{};
                if (p.isOp("(")) {
                    p.t += 1;
                    a = try p.args();
                } else if (p.peek()) |t| {
                    // One argument without parentheses: `is divisibleby 3`.
                    const keyword = t.kind == .name and for ([_][]const u8{ "else", "or", "and", "if", "in", "not", "is" }) |k| {
                        if (std.mem.eql(u8, t.text, k)) break true;
                    } else false;
                    const starts = t.kind != .op or std.mem.eql(u8, t.text, "(") or std.mem.eql(u8, t.text, "[") or std.mem.eql(u8, t.text, "{");
                    if (starts and !keyword) {
                        const arg = try p.postfix(try p.primary());
                        const list = try p.arena.alloc(*const Expr, 1);
                        list[0] = arg;
                        a = .{ .pos = list };
                    }
                }
                e = try p.new(.{ .@"test" = .{ .value = e, .name = n, .args = a, .negate = negate } });
            } else if (p.isOp("(")) {
                p.t += 1;
                e = try p.new(.{ .call = .{ .callee = e, .args = try p.args() } });
            } else return e;
        }
    }
};

// ---------------------------------------------------------------- evaluation

const Flow = enum { normal, break_, continue_ };

const Renderer = struct {
    arena: Allocator,
    globals: *Dict,
    /// Local scopes (loops, macros), innermost last.
    frames: std.ArrayList(*Dict) = .empty,
    now: ?Date,
    raised: *?[]const u8,

    fn nodes(r: *Renderer, list: []const Node, out: *std.ArrayList(u8)) Error!Flow {
        for (list) |n| {
            const flow = try r.node(n, out);
            if (flow != .normal) return flow;
        }
        return .normal;
    }

    fn node(r: *Renderer, n: Node, out: *std.ArrayList(u8)) Error!Flow {
        switch (n) {
            .text => |s| try out.appendSlice(r.arena, s),
            .output => |e| try r.writeStr(out, try r.eval(e)),
            .if_ => |branches| for (branches) |b| {
                if (b.cond == null or truthy(try r.eval(b.cond.?))) return r.nodes(b.body, out);
            },
            .for_ => |f| try r.forLoop(f, out),
            .set => |s| {
                const v = if (s.value) |e| try r.eval(e) else blk: {
                    var buf: std.ArrayList(u8) = .empty;
                    _ = try r.nodes(s.body, &buf);
                    break :blk Value{ .string = buf.items };
                };
                if (s.targets.len == 1) {
                    try r.assign(s.targets[0], v);
                } else {
                    const items = try r.iterate(v);
                    if (items.len != s.targets.len) return error.TemplateError;
                    for (s.targets, items) |t, item| try r.assign(t, item);
                }
            },
            .macro => |m| {
                const c = try r.arena.create(Callable);
                c.* = .{ .macro = m };
                try r.scope().put(r.arena, m.name, .{ .callable = c });
            },
            .break_ => return .break_,
            .continue_ => return .continue_,
            .block => |body| return r.nodes(body, out),
        }
        return .normal;
    }

    fn scope(r: *Renderer) *Dict {
        return if (r.frames.items.len > 0) r.frames.items[r.frames.items.len - 1] else r.globals;
    }

    fn assign(r: *Renderer, t: Target, v: Value) Error!void {
        if (t.attr) |attr| {
            const obj = try r.lookup(t.name);
            if (obj != .namespace) return error.TemplateError;
            return obj.namespace.put(r.arena, attr, v);
        }
        try r.scope().put(r.arena, t.name, v);
    }

    fn lookup(r: *Renderer, n: []const u8) Error!Value {
        var i = r.frames.items.len;
        while (i > 0) {
            i -= 1;
            if (r.frames.items[i].get(n)) |v| return v;
        }
        if (r.globals.get(n)) |v| return v;
        const b: Builtin = std.meta.stringToEnum(Builtin, n) orelse return .undefined;
        if (b == .strftime_now and r.now == null) return .undefined;
        const c = try r.arena.create(Callable);
        c.* = .{ .builtin = b };
        return .{ .callable = c };
    }

    fn forLoop(r: *Renderer, f: *const For, out: *std.ArrayList(u8)) Error!void {
        const all = try r.iterate(try r.eval(f.iter));
        const frame = try r.arena.create(Dict);
        frame.* = .{};
        try r.frames.append(r.arena, frame);
        defer r.frames.items.len -= 1;
        // The loop filter runs before counting, as in jinja2.
        var items: std.ArrayList(Value) = .empty;
        for (all) |item| {
            if (f.filter) |cond| {
                try r.bind(frame, f.targets, item);
                if (!truthy(try r.eval(cond))) continue;
            }
            try items.append(r.arena, item);
        }
        if (items.items.len == 0) {
            _ = try r.nodes(f.otherwise, out);
            return;
        }
        const n = items.items.len;
        for (items.items, 0..) |item, i| {
            try r.bind(frame, f.targets, item);
            const loop = try r.arena.create(Dict);
            loop.* = .{};
            const fields = [_]struct { []const u8, Value }{
                .{ "index", .{ .integer = @intCast(i + 1) } },
                .{ "index0", .{ .integer = @intCast(i) } },
                .{ "revindex", .{ .integer = @intCast(n - i) } },
                .{ "revindex0", .{ .integer = @intCast(n - i - 1) } },
                .{ "first", .{ .boolean = i == 0 } },
                .{ "last", .{ .boolean = i + 1 == n } },
                .{ "length", .{ .integer = @intCast(n) } },
                .{ "previtem", if (i > 0) items.items[i - 1] else .undefined },
                .{ "nextitem", if (i + 1 < n) items.items[i + 1] else .undefined },
            };
            for (fields) |fl| try loop.put(r.arena, fl[0], fl[1]);
            try frame.put(r.arena, "loop", .{ .dict = loop });
            if (try r.nodes(f.body, out) == .break_) break;
        }
    }

    fn bind(r: *Renderer, frame: *Dict, targets: []const []const u8, item: Value) Error!void {
        if (targets.len == 1) return frame.put(r.arena, targets[0], item);
        const parts = try r.iterate(item);
        if (parts.len != targets.len) return error.TemplateError;
        for (targets, parts) |t, v| try frame.put(r.arena, t, v);
    }

    /// The items `for` visits: a list's, a dict's keys, a string's code
    /// points; none for undefined.
    fn iterate(r: *Renderer, v: Value) Error![]const Value {
        return switch (v) {
            .list => |l| l,
            .undefined => &.{},
            .dict => |d| blk: {
                const items = try r.arena.alloc(Value, d.keys.items.len);
                for (items, d.keys.items) |*o, k| o.* = .{ .string = k };
                break :blk items;
            },
            .string => |s| codePoints(r.arena, s),
            else => error.TemplateError,
        };
    }

    fn eval(r: *Renderer, e: *const Expr) Error!Value {
        const a = r.arena;
        switch (e.*) {
            .literal => |v| return v,
            .name => |n| return r.lookup(n),
            .list => |items| {
                const out = try a.alloc(Value, items.len);
                for (out, items) |*o, item| o.* = try r.eval(item);
                return .{ .list = out };
            },
            .dict => |pairs| {
                const d = try a.create(Dict);
                d.* = .{};
                for (pairs) |pr| {
                    const k = try r.eval(pr.key);
                    if (k != .string) return error.TemplateError;
                    try d.put(a, k.string, try r.eval(pr.value));
                }
                return .{ .dict = d };
            },
            .attr => |x| return r.getAttr(try r.eval(x.obj), x.name),
            .item => |x| return r.getItem(try r.eval(x.obj), try r.eval(x.index)),
            .slice => |x| {
                const obj = try r.eval(x.obj);
                const start = if (x.start) |s| try r.eval(s) else Value.none;
                const stop = if (x.stop) |s| try r.eval(s) else Value.none;
                const step = if (x.step) |s| try r.eval(s) else Value.none;
                return r.slice(obj, start, stop, step);
            },
            .call => |c| {
                const callee = try r.eval(c.callee);
                if (callee != .callable) return error.TemplateError;
                return r.call(callee.callable, try r.evalArgs(c.args));
            },
            .filter => |f| return r.filter(f.name, try r.eval(f.value), try r.evalArgs(f.args)),
            .@"test" => |t| {
                const ok = try r.runTest(t.name, try r.eval(t.value), try r.evalArgs(t.args));
                return .{ .boolean = ok != t.negate };
            },
            .neg => |x| {
                const v = try r.eval(x);
                return switch (try numeric(v)) {
                    .int => |i| .{ .integer = -i },
                    .float => |f| .{ .float = -f },
                };
            },
            .not_ => |x| return .{ .boolean = !truthy(try r.eval(x)) },
            .cond => |c| {
                if (truthy(try r.eval(c.cond))) return r.eval(c.then);
                return if (c.otherwise) |o| r.eval(o) else .undefined;
            },
            .binary => |b| {
                const l = try r.eval(b.l);
                switch (b.op) {
                    .and_ => return if (truthy(l)) r.eval(b.r) else l,
                    .or_ => return if (truthy(l)) l else r.eval(b.r),
                    else => {},
                }
                return r.binary(b.op, l, try r.eval(b.r));
            },
        }
    }

    const EvalArgs = struct { pos: []const Value, kw: []const struct { name: []const u8, value: Value } };

    fn evalArgs(r: *Renderer, args: Args) Error!EvalArgs {
        const pos = try r.arena.alloc(Value, args.pos.len);
        for (pos, args.pos) |*o, e| o.* = try r.eval(e);
        const kw = try r.arena.alloc(@typeInfo(@FieldType(EvalArgs, "kw")).pointer.child, args.kw.len);
        for (kw, args.kw) |*o, k| o.* = .{ .name = k.name, .value = try r.eval(k.value) };
        return .{ .pos = pos, .kw = kw };
    }

    fn binary(r: *Renderer, op: BinOp, l: Value, rv: Value) Error!Value {
        const a = r.arena;
        switch (op) {
            .eq => return .{ .boolean = equal(l, rv) },
            .ne => return .{ .boolean = !equal(l, rv) },
            .lt, .le, .gt, .ge => {
                const ord = try order(l, rv);
                return .{ .boolean = switch (op) {
                    .lt => ord == .lt,
                    .le => ord != .gt,
                    .gt => ord == .gt,
                    else => ord != .lt,
                } };
            },
            .in => return .{ .boolean = try contains(rv, l) },
            .not_in => return .{ .boolean = !try contains(rv, l) },
            .concat => {
                var buf: std.ArrayList(u8) = .empty;
                try r.writeStr(&buf, l);
                try r.writeStr(&buf, rv);
                return .{ .string = buf.items };
            },
            .add => {
                if (l == .string and rv == .string) return .{ .string = try std.mem.concat(a, u8, &.{ l.string, rv.string }) };
                if (l == .list and rv == .list) return .{ .list = try std.mem.concat(a, Value, &.{ l.list, rv.list }) };
            },
            .mul => {
                if (l == .string or rv == .string) {
                    const s = if (l == .string) l.string else rv.string;
                    const n = (try numeric(if (l == .string) rv else l)).asInt() orelse return error.TemplateError;
                    var buf: std.ArrayList(u8) = .empty;
                    var k: i64 = 0;
                    while (k < n) : (k += 1) try buf.appendSlice(a, s);
                    return .{ .string = buf.items };
                }
            },
            else => {},
        }
        const x = try numeric(l);
        const y = try numeric(rv);
        if (x == .int and y == .int and op != .div and op != .pow) {
            const i = x.int;
            const j = y.int;
            return .{ .integer = switch (op) {
                .add => std.math.add(i64, i, j) catch return error.TemplateError,
                .sub => std.math.sub(i64, i, j) catch return error.TemplateError,
                .mul => std.math.mul(i64, i, j) catch return error.TemplateError,
                .floordiv => if (j == 0) return error.TemplateError else @divFloor(i, j),
                .mod => if (j == 0) return error.TemplateError else @mod(i, j),
                else => unreachable,
            } };
        }
        if (x == .int and y == .int and op == .pow and y.int >= 0) {
            return .{ .integer = std.math.powi(i64, x.int, y.int) catch return error.TemplateError };
        }
        const f = x.asFloat();
        const g = y.asFloat();
        return .{ .float = switch (op) {
            .add => f + g,
            .sub => f - g,
            .mul => f * g,
            .div => if (g == 0) return error.TemplateError else f / g,
            .floordiv => if (g == 0) return error.TemplateError else @floor(f / g),
            .mod => if (g == 0) return error.TemplateError else f - g * @floor(f / g),
            .pow => std.math.pow(f64, f, g),
            else => unreachable,
        } };
    }

    fn getAttr(r: *Renderer, obj: Value, n: []const u8) Error!Value {
        switch (obj) {
            .undefined => return error.TemplateError,
            .dict => |d| {
                // Python's getattr first: a dict's methods shadow its keys.
                if (isMethod(.dict, n)) return r.method(obj, n);
                return d.get(n) orelse .undefined;
            },
            .namespace => |d| return d.get(n) orelse .undefined,
            .string => return if (isMethod(.string, n)) r.method(obj, n) else .undefined,
            else => return .undefined,
        }
    }

    fn method(r: *Renderer, obj: Value, n: []const u8) Error!Value {
        const c = try r.arena.create(Callable);
        c.* = .{ .method = .{ .self = obj, .name = n } };
        return .{ .callable = c };
    }

    fn getItem(r: *Renderer, obj: Value, index: Value) Error!Value {
        switch (obj) {
            .undefined => return error.TemplateError,
            .dict, .namespace => |d| {
                if (index != .string) return .undefined;
                if (d.get(index.string)) |v| return v;
                return if (obj == .dict) r.getAttr(obj, index.string) else .undefined;
            },
            .list => |l| {
                const i = (try numeric(index)).asInt() orelse return .undefined;
                const at = if (i < 0) @as(i64, @intCast(l.len)) + i else i;
                return if (at >= 0 and at < l.len) l[@intCast(at)] else .undefined;
            },
            .string => |s| {
                if (index == .string) return r.getAttr(obj, index.string);
                const cps = try codePoints(r.arena, s);
                const i = (try numeric(index)).asInt() orelse return .undefined;
                const at = if (i < 0) @as(i64, @intCast(cps.len)) + i else i;
                return if (at >= 0 and at < cps.len) cps[@intCast(at)] else .undefined;
            },
            else => return .undefined,
        }
    }

    fn slice(r: *Renderer, obj: Value, start: Value, stop: Value, step: Value) Error!Value {
        const items: []const Value = switch (obj) {
            .list => |l| l,
            .string => |s| try codePoints(r.arena, s),
            .undefined => return error.TemplateError,
            else => return error.TemplateError,
        };
        const n: i64 = @intCast(items.len);
        const st: i64 = if (step == .none) 1 else (try numeric(step)).asInt() orelse return error.TemplateError;
        if (st == 0) return error.TemplateError;
        const Bound = struct {
            fn get(v: Value, len: i64, default: i64, s: i64) Error!i64 {
                if (v == .none) return default;
                var i = (try numeric(v)).asInt() orelse return error.TemplateError;
                if (i < 0) i += len;
                return if (s > 0) std.math.clamp(i, 0, len) else std.math.clamp(i, -1, len - 1);
            }
        };
        const lo = try Bound.get(start, n, if (st > 0) 0 else n - 1, st);
        const hi = try Bound.get(stop, n, if (st > 0) n else -1, st);
        var out: std.ArrayList(Value) = .empty;
        var i = lo;
        while (if (st > 0) i < hi else i > hi) : (i += st) try out.append(r.arena, items[@intCast(i)]);
        if (obj == .list) return .{ .list = out.items };
        var buf: std.ArrayList(u8) = .empty;
        for (out.items) |c| try buf.appendSlice(r.arena, c.string);
        return .{ .string = buf.items };
    }

    fn call(r: *Renderer, c: *const Callable, args: EvalArgs) Error!Value {
        const a = r.arena;
        switch (c.*) {
            .macro => |m| {
                const frame = try a.create(Dict);
                frame.* = .{};
                if (args.pos.len > m.params.len) return error.TemplateError;
                for (m.params, 0..) |param, i| {
                    const v = if (i < args.pos.len) args.pos[i] else if (kwarg(args, param.name)) |k| k else if (param.default) |d| try r.eval(d) else Value.undefined;
                    try frame.put(a, param.name, v);
                }
                // A macro sees the template's globals, not its caller's locals.
                const saved = r.frames;
                r.frames = .empty;
                try r.frames.append(a, frame);
                defer r.frames = saved;
                var buf: std.ArrayList(u8) = .empty;
                _ = try r.nodes(m.body, &buf);
                return .{ .string = buf.items };
            },
            .builtin => |b| switch (b) {
                .namespace, .dict => {
                    const d = try a.create(Dict);
                    d.* = .{};
                    for (args.pos) |p| {
                        if (p != .dict) return error.TemplateError;
                        for (p.dict.keys.items, p.dict.values.items) |k, v| try d.put(a, k, v);
                    }
                    for (args.kw) |k| try d.put(a, k.name, k.value);
                    return if (b == .namespace) .{ .namespace = d } else .{ .dict = d };
                },
                .range => {
                    var bounds: [3]i64 = .{ 0, 0, 1 };
                    switch (args.pos.len) {
                        1 => bounds[1] = (try numeric(args.pos[0])).asInt() orelse return error.TemplateError,
                        2, 3 => for (args.pos, 0..) |p, i| {
                            bounds[i] = (try numeric(p)).asInt() orelse return error.TemplateError;
                        },
                        else => return error.TemplateError,
                    }
                    if (bounds[2] == 0) return error.TemplateError;
                    var out: std.ArrayList(Value) = .empty;
                    var i = bounds[0];
                    while (if (bounds[2] > 0) i < bounds[1] else i > bounds[1]) : (i += bounds[2]) try out.append(a, .{ .integer = i });
                    return .{ .list = out.items };
                },
                .raise_exception => {
                    var buf: std.ArrayList(u8) = .empty;
                    if (args.pos.len > 0) try r.writeStr(&buf, args.pos[0]);
                    r.raised.* = buf.items;
                    return error.Raised;
                },
                .strftime_now => {
                    if (args.pos.len != 1 or args.pos[0] != .string) return error.TemplateError;
                    return .{ .string = try strftime(a, args.pos[0].string, r.now.?) };
                },
            },
            .method => |m| return r.callMethod(m.self, m.name, args),
        }
    }

    fn callMethod(r: *Renderer, self: Value, n: []const u8, args: EvalArgs) Error!Value {
        const a = r.arena;
        if (self == .dict) {
            const d = self.dict;
            if (std.mem.eql(u8, n, "items")) return .{ .list = try dictItems(a, d) };
            if (std.mem.eql(u8, n, "keys")) return .{ .list = try r.iterate(self) };
            if (std.mem.eql(u8, n, "values")) return .{ .list = d.values.items };
            if (std.mem.eql(u8, n, "get")) {
                if (args.pos.len == 0 or args.pos[0] != .string) return error.TemplateError;
                return d.get(args.pos[0].string) orelse if (args.pos.len > 1) args.pos[1] else .none;
            }
            return error.TemplateError;
        }
        const s = self.string;
        const arg0: ?Value = if (args.pos.len > 0) args.pos[0] else null;
        if (std.mem.eql(u8, n, "strip") or std.mem.eql(u8, n, "lstrip") or std.mem.eql(u8, n, "rstrip")) {
            const chars: ?[]const u8 = if (arg0) |c| switch (c) {
                .string => |cs| cs,
                .none => null,
                else => return error.TemplateError,
            } else null;
            return .{ .string = pyStrip(s, chars, n[0] != 'r', n[0] != 'l') };
        }
        if (std.mem.eql(u8, n, "split") or std.mem.eql(u8, n, "rsplit")) {
            const sep: ?[]const u8 = if (arg0) |c| switch (c) {
                .string => |cs| cs,
                .none => null,
                else => return error.TemplateError,
            } else null;
            const maxsplit: i64 = if (args.pos.len > 1) (try numeric(args.pos[1])).asInt() orelse -1 else -1;
            if (n[0] == 'r' and maxsplit >= 0) return error.TemplateError;
            return .{ .list = try pySplit(a, s, sep, maxsplit) };
        }
        if (std.mem.eql(u8, n, "startswith") or std.mem.eql(u8, n, "endswith")) {
            const prefixes: []const Value = switch (arg0 orelse return error.TemplateError) {
                .string => &.{arg0.?},
                .list => |l| l,
                else => return error.TemplateError,
            };
            for (prefixes) |p| {
                if (p != .string) return error.TemplateError;
                if (if (n[0] == 's') std.mem.startsWith(u8, s, p.string) else std.mem.endsWith(u8, s, p.string)) return .{ .boolean = true };
            }
            return .{ .boolean = false };
        }
        if (std.mem.eql(u8, n, "upper")) return .{ .string = try asciiCase(a, s, true) };
        if (std.mem.eql(u8, n, "lower")) return .{ .string = try asciiCase(a, s, false) };
        if (std.mem.eql(u8, n, "replace")) {
            if (args.pos.len < 2 or args.pos[0] != .string or args.pos[1] != .string) return error.TemplateError;
            return .{ .string = try std.mem.replaceOwned(u8, a, s, args.pos[0].string, args.pos[1].string) };
        }
        return error.TemplateError;
    }

    fn filter(r: *Renderer, n: []const u8, v: Value, args: EvalArgs) Error!Value {
        const a = r.arena;
        const F = enum { tojson, trim, length, count, first, last, default, d, items, dictsort, safe, string, list, join, upper, lower, replace, int, float, abs, reverse, map, selectattr, rejectattr, unique, capitalize };
        const f = std.meta.stringToEnum(F, n) orelse return error.TemplateError;
        switch (f) {
            .tojson => {
                var o: JsonOptions = .{};
                for (args.kw) |k| {
                    if (std.mem.eql(u8, k.name, "ensure_ascii")) {
                        o.ensure_ascii = truthy(k.value);
                    } else if (std.mem.eql(u8, k.name, "indent")) {
                        o.indent = if (k.value == .none) null else @intCast((try numeric(k.value)).asInt() orelse return error.TemplateError);
                    } else if (std.mem.eql(u8, k.name, "sort_keys")) {
                        o.sort_keys = truthy(k.value);
                    } else if (std.mem.eql(u8, k.name, "separators")) {
                        if (k.value != .list or k.value.list.len != 2 or k.value.list[0] != .string or k.value.list[1] != .string) return error.TemplateError;
                        o.separators = .{ k.value.list[0].string, k.value.list[1].string };
                    } else return error.TemplateError;
                }
                if (args.pos.len > 0) o.indent = @intCast((try numeric(args.pos[0])).asInt() orelse return error.TemplateError);
                var buf: std.ArrayList(u8) = .empty;
                try writeJson(a, &buf, v, o, 0);
                return .{ .string = buf.items };
            },
            .trim => {
                const s = try r.toStr(v);
                const chars: ?[]const u8 = if (args.pos.len > 0 and args.pos[0] == .string) args.pos[0].string else null;
                return .{ .string = pyStrip(s, chars, true, true) };
            },
            .length, .count => return .{ .integer = @intCast(switch (v) {
                .string => |s| std.unicode.utf8CountCodepoints(s) catch s.len,
                .list => |l| l.len,
                .dict => |d| d.keys.items.len,
                .undefined => 0,
                else => return error.TemplateError,
            }) },
            .first, .last => {
                const items = try r.iterate(v);
                if (items.len == 0) return .undefined;
                return if (f == .first) items[0] else items[items.len - 1];
            },
            .default, .d => {
                const fallback: Value = if (args.pos.len > 0) args.pos[0] else .{ .string = "" };
                const boolean = (args.pos.len > 1 and truthy(args.pos[1])) or (kwarg(args, "boolean") != null and truthy(kwarg(args, "boolean").?));
                if (v == .undefined or (boolean and !truthy(v))) return fallback;
                return v;
            },
            .items => return switch (v) {
                .dict => |d| .{ .list = try dictItems(a, d) },
                .undefined => .{ .list = &.{} },
                else => error.TemplateError,
            },
            .dictsort => return .{ .list = try dictSort(a, v, args) },
            .safe => return v,
            .string => return .{ .string = try r.toStr(v) },
            .list => return .{ .list = try r.iterate(v) },
            .join => {
                const sep: []const u8 = if (args.pos.len > 0) try r.toStr(args.pos[0]) else "";
                var buf: std.ArrayList(u8) = .empty;
                for (try r.iterate(v), 0..) |item, i| {
                    if (i > 0) try buf.appendSlice(a, sep);
                    try r.writeStr(&buf, item);
                }
                return .{ .string = buf.items };
            },
            .upper, .lower => return .{ .string = try asciiCase(a, try r.toStr(v), f == .upper) },
            .capitalize => {
                const s = try asciiCase(a, try r.toStr(v), false);
                const m = try a.dupe(u8, s);
                if (m.len > 0) m[0] = std.ascii.toUpper(m[0]);
                return .{ .string = m };
            },
            .replace => {
                if (args.pos.len < 2) return error.TemplateError;
                return .{ .string = try std.mem.replaceOwned(u8, a, try r.toStr(v), try r.toStr(args.pos[0]), try r.toStr(args.pos[1])) };
            },
            .int => return .{ .integer = switch (numeric(v) catch return .{ .integer = 0 }) {
                .int => |i| i,
                .float => |x| @intFromFloat(@trunc(x)),
            } },
            .float => return .{ .float = (numeric(v) catch return .{ .float = 0 }).asFloat() },
            .abs => return switch (try numeric(v)) {
                .int => |i| .{ .integer = @intCast(@abs(i)) },
                .float => |x| .{ .float = @abs(x) },
            },
            .reverse => {
                if (v == .string) return r.slice(v, .none, .none, .{ .integer = -1 });
                const items = try r.iterate(v);
                const out = try a.alloc(Value, items.len);
                for (out, 0..) |*o, i| o.* = items[items.len - 1 - i];
                return .{ .list = out };
            },
            .map => {
                const attr = kwarg(args, "attribute") orelse {
                    if (args.pos.len == 0 or args.pos[0] != .string) return error.TemplateError;
                    var out: std.ArrayList(Value) = .empty;
                    for (try r.iterate(v)) |item| try out.append(a, try r.filter(args.pos[0].string, item, .{ .pos = args.pos[1..], .kw = &.{} }));
                    return .{ .list = out.items };
                };
                if (attr != .string) return error.TemplateError;
                var out: std.ArrayList(Value) = .empty;
                for (try r.iterate(v)) |item| try out.append(a, try r.getItem(item, attr));
                return .{ .list = out.items };
            },
            .selectattr, .rejectattr => {
                if (args.pos.len == 0 or args.pos[0] != .string) return error.TemplateError;
                var out: std.ArrayList(Value) = .empty;
                for (try r.iterate(v)) |item| {
                    const field = try r.getItem(item, args.pos[0]);
                    const ok = if (args.pos.len > 1) blk: {
                        if (args.pos[1] != .string) return error.TemplateError;
                        break :blk try r.runTest(args.pos[1].string, field, .{ .pos = args.pos[2..], .kw = &.{} });
                    } else truthy(field);
                    if (ok == (f == .selectattr)) try out.append(a, item);
                }
                return .{ .list = out.items };
            },
            .unique => {
                var out: std.ArrayList(Value) = .empty;
                for (try r.iterate(v)) |item| {
                    for (out.items) |seen| {
                        if (equal(seen, item)) break;
                    } else try out.append(a, item);
                }
                return .{ .list = out.items };
            },
        }
    }

    fn runTest(r: *Renderer, n: []const u8, v: Value, args: EvalArgs) Error!bool {
        _ = r;
        const T = enum { defined, undefined, none, string, number, integer, float, boolean, true, false, mapping, iterable, sequence, callable, odd, even, eq, equalto, sameas, in, lower, upper, divisibleby };
        const t = std.meta.stringToEnum(T, n) orelse return error.TemplateError;
        return switch (t) {
            .defined => v != .undefined,
            .undefined => v == .undefined,
            .none => v == .none,
            .string => v == .string,
            .number => v == .integer or v == .float or v == .number or v == .boolean,
            .integer => v == .integer or (v == .number and (try numeric(v)) == .int),
            .float => v == .float or (v == .number and (try numeric(v)) == .float),
            .boolean => v == .boolean,
            .true => v == .boolean and v.boolean,
            .false => v == .boolean and !v.boolean,
            .mapping => v == .dict,
            .iterable => v == .list or v == .dict or v == .string or v == .undefined,
            .sequence => v == .list or v == .dict or v == .string or v == .undefined,
            .callable => v == .callable,
            .odd, .even => blk: {
                const i = (try numeric(v)).asInt() orelse return error.TemplateError;
                break :blk (@mod(i, 2) == 1) == (t == .odd);
            },
            .eq, .equalto, .sameas => args.pos.len == 1 and equal(v, args.pos[0]),
            .in => args.pos.len == 1 and try contains(args.pos[0], v),
            .lower, .upper => v == .string and blk: {
                for (v.string) |c| if (if (t == .lower) std.ascii.isUpper(c) else std.ascii.isLower(c)) break :blk false;
                break :blk true;
            },
            .divisibleby => blk: {
                if (args.pos.len != 1) return error.TemplateError;
                const i = (try numeric(v)).asInt() orelse return error.TemplateError;
                const j = (try numeric(args.pos[0])).asInt() orelse return error.TemplateError;
                break :blk j != 0 and @mod(i, j) == 0;
            },
        };
    }

    fn toStr(r: *Renderer, v: Value) Error![]const u8 {
        if (v == .string) return v.string;
        var buf: std.ArrayList(u8) = .empty;
        try r.writeStr(&buf, v);
        return buf.items;
    }

    /// Python's str(v); undefined prints nothing.
    fn writeStr(r: *Renderer, out: *std.ArrayList(u8), v: Value) Error!void {
        switch (v) {
            .string => |s| try out.appendSlice(r.arena, s),
            .undefined => {},
            else => try writeRepr(r.arena, out, v),
        }
    }
};

fn kwarg(args: Renderer.EvalArgs, n: []const u8) ?Value {
    for (args.kw) |k| if (std.mem.eql(u8, k.name, n)) return k.value;
    return null;
}

const Kind = enum { dict, string };

fn isMethod(kind: Kind, n: []const u8) bool {
    const names: []const []const u8 = switch (kind) {
        .dict => &.{ "items", "keys", "values", "get" },
        .string => &.{ "strip", "lstrip", "rstrip", "split", "rsplit", "startswith", "endswith", "upper", "lower", "replace" },
    };
    for (names) |m| if (std.mem.eql(u8, m, n)) return true;
    return false;
}

fn dictItems(a: Allocator, d: *const Dict) Allocator.Error![]const Value {
    const out = try a.alloc(Value, d.keys.items.len);
    for (out, d.keys.items, d.values.items) |*o, k, v| {
        const pair = try a.alloc(Value, 2);
        pair[0] = .{ .string = k };
        pair[1] = v;
        o.* = .{ .list = pair };
    }
    return out;
}

/// Jinja's stable dictsort: pairs, not keys, with case-insensitive keys by
/// default. Case conversion follows this renderer's ASCII lower/upper subset.
fn dictSort(a: Allocator, v: Value, args: Renderer.EvalArgs) Error![]const Value {
    if (v != .dict or args.pos.len > 3) return error.TemplateError;
    var options = [3]Value{ .{ .boolean = false }, .{ .string = "key" }, .{ .boolean = false } };
    var supplied = [3]bool{ false, false, false };
    for (args.pos, 0..) |arg, i| {
        options[i] = arg;
        supplied[i] = true;
    }
    for (args.kw) |arg| {
        const i: usize = if (std.mem.eql(u8, arg.name, "case_sensitive"))
            0
        else if (std.mem.eql(u8, arg.name, "by"))
            1
        else if (std.mem.eql(u8, arg.name, "reverse"))
            2
        else
            return error.TemplateError;
        if (supplied[i]) return error.TemplateError;
        options[i] = arg.value;
        supplied[i] = true;
    }
    if (options[1] != .string) return error.TemplateError;
    const by: usize = if (std.mem.eql(u8, options[1].string, "key"))
        0
    else if (std.mem.eql(u8, options[1].string, "value"))
        1
    else
        return error.TemplateError;
    const case_sensitive = truthy(options[0]);
    const reverse = truthy(options[2]);
    const pairs = @constCast(try dictItems(a, v.dict));
    if (pairs.len < 2) return pairs;
    // Insertion sort keeps equal folded keys in original dictionary order,
    // including in reverse mode; comparisons can propagate Python type errors.
    for (1..pairs.len) |i| {
        const pair = pairs[i];
        var j = i;
        while (j > 0) : (j -= 1) {
            const left = pairs[j - 1].list[by];
            const right = pair.list[by];
            const cmp = if (!case_sensitive and left == .string and right == .string)
                asciiLowerOrder(left.string, right.string)
            else
                try order(left, right);
            if (cmp != (if (reverse) std.math.Order.lt else .gt)) break;
            pairs[j] = pairs[j - 1];
        }
        pairs[j] = pair;
    }
    return pairs;
}

fn asciiLowerOrder(a: []const u8, b: []const u8) std.math.Order {
    for (a[0..@min(a.len, b.len)], b[0..@min(a.len, b.len)]) |x, y| {
        const cmp = std.math.order(std.ascii.toLower(x), std.ascii.toLower(y));
        if (cmp != .eq) return cmp;
    }
    return std.math.order(a.len, b.len);
}

fn codePoints(a: Allocator, s: []const u8) Error![]const Value {
    var out: std.ArrayList(Value) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
        const end = @min(i + n, s.len);
        try out.append(a, .{ .string = s[i..end] });
        i = end;
    }
    return out.items;
}

const Num = union(enum) {
    int: i64,
    float: f64,

    fn asFloat(n: Num) f64 {
        return switch (n) {
            .int => |i| @floatFromInt(i),
            .float => |f| f,
        };
    }

    fn asInt(n: Num) ?i64 {
        return switch (n) {
            .int => |i| i,
            .float => null,
        };
    }
};

fn numeric(v: Value) Error!Num {
    return switch (v) {
        .integer => |i| .{ .int = i },
        .float => |f| .{ .float = f },
        .boolean => |b| .{ .int = @intFromBool(b) },
        .number => |s| if (std.mem.indexOfAny(u8, s, ".eE") == null)
            if (std.fmt.parseInt(i64, s, 10)) |i| .{ .int = i } else |_| .{ .float = std.fmt.parseFloat(f64, s) catch return error.TemplateError }
        else
            .{ .float = std.fmt.parseFloat(f64, s) catch return error.TemplateError },
        else => error.TemplateError,
    };
}

fn isNumber(v: Value) bool {
    return v == .integer or v == .float or v == .number or v == .boolean;
}

fn truthy(v: Value) bool {
    return switch (v) {
        .undefined, .none => false,
        .boolean => |b| b,
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .number => |s| (numeric(.{ .number = s }) catch return true).asFloat() != 0,
        .string => |s| s.len > 0,
        .list => |l| l.len > 0,
        .dict => |d| d.keys.items.len > 0,
        .namespace, .callable => true,
    };
}

fn equal(a: Value, b: Value) bool {
    if (isNumber(a) and isNumber(b)) {
        const x = numeric(a) catch return false;
        const y = numeric(b) catch return false;
        if (x == .int and y == .int) return x.int == y.int;
        return x.asFloat() == y.asFloat();
    }
    return switch (a) {
        .undefined => b == .undefined,
        .none => b == .none,
        .string => |s| b == .string and std.mem.eql(u8, s, b.string),
        .list => |l| b == .list and l.len == b.list.len and for (l, b.list) |x, y| {
            if (!equal(x, y)) break false;
        } else true,
        .dict => |d| b == .dict and d.keys.items.len == b.dict.keys.items.len and for (d.keys.items, d.values.items) |k, v| {
            const other = b.dict.get(k) orelse break false;
            if (!equal(v, other)) break false;
        } else true,
        .namespace => |d| b == .namespace and d == b.namespace,
        .callable => |c| b == .callable and c == b.callable,
        else => false,
    };
}

fn order(a: Value, b: Value) Error!std.math.Order {
    if (isNumber(a) and isNumber(b)) {
        const x = try numeric(a);
        const y = try numeric(b);
        if (x == .int and y == .int) return std.math.order(x.int, y.int);
        return std.math.order(x.asFloat(), y.asFloat());
    }
    if (a == .string and b == .string) return std.mem.order(u8, a.string, b.string);
    if (a == .list and b == .list) {
        for (a.list, b.list) |x, y| {
            if (equal(x, y)) continue;
            return order(x, y);
        }
        return std.math.order(a.list.len, b.list.len);
    }
    return error.TemplateError;
}

fn contains(container: Value, item: Value) Error!bool {
    return switch (container) {
        .string => |s| if (item == .string) std.mem.indexOf(u8, s, item.string) != null else error.TemplateError,
        .list => |l| for (l) |x| {
            if (equal(x, item)) break true;
        } else false,
        .dict, .namespace => |d| item == .string and d.get(item.string) != null,
        .undefined => false,
        else => error.TemplateError,
    };
}

/// Python's whitespace (str.isspace) for the code point starting at s[i],
/// with its length.
fn spaceAt(s: []const u8, i: usize) ?usize {
    const n = std.unicode.utf8ByteSequenceLength(s[i]) catch return null;
    if (i + n > s.len) return null;
    const cp = std.unicode.utf8Decode(s[i..][0..n]) catch return null;
    const space = switch (cp) {
        '\t', '\n', 0x0b, 0x0c, '\r', 0x1c...0x1f, ' ', 0x85 => true,
        else => cp >= 0x80 and unicode.isSpace(cp),
    };
    return if (space) n else null;
}

/// str.strip / lstrip / rstrip, by whitespace or by a set of characters.
fn pyStrip(s: []const u8, chars: ?[]const u8, left: bool, right: bool) []const u8 {
    var lo: usize = 0;
    var hi: usize = s.len;
    if (chars) |set| {
        if (left) while (lo < hi) {
            const n = std.unicode.utf8ByteSequenceLength(s[lo]) catch 1;
            if (lo + n > hi or std.mem.indexOf(u8, set, s[lo..][0..n]) == null) break;
            lo += n;
        };
        if (right) while (hi > lo) {
            var start = hi - 1;
            while (start > lo and (s[start] & 0xc0) == 0x80) start -= 1;
            if (std.mem.indexOf(u8, set, s[start..hi]) == null) break;
            hi = start;
        };
        return s[lo..hi];
    }
    if (left) {
        while (lo < hi) lo += spaceAt(s, lo) orelse break;
    }
    if (right) while (hi > lo) {
        var start = hi - 1;
        while (start > lo and (s[start] & 0xc0) == 0x80) start -= 1;
        if (spaceAt(s, start) == null) break;
        hi = start;
    };
    return s[lo..hi];
}

/// str.split(sep, maxsplit).
fn pySplit(a: Allocator, s: []const u8, sep: ?[]const u8, maxsplit: i64) Error![]const Value {
    var out: std.ArrayList(Value) = .empty;
    if (sep) |d| {
        if (d.len == 0) return error.TemplateError;
        var rest = s;
        var splits: i64 = 0;
        while (maxsplit < 0 or splits < maxsplit) : (splits += 1) {
            const at = std.mem.indexOf(u8, rest, d) orelse break;
            try out.append(a, .{ .string = rest[0..at] });
            rest = rest[at + d.len ..];
        }
        try out.append(a, .{ .string = rest });
        return out.items;
    }
    // Runs of whitespace; no empty strings.
    var i: usize = 0;
    var splits: i64 = 0;
    while (i < s.len) {
        while (i < s.len) i += spaceAt(s, i) orelse break;
        if (i >= s.len) break;
        if (maxsplit >= 0 and splits >= maxsplit) {
            try out.append(a, .{ .string = pyStrip(s[i..], null, false, true) });
            break;
        }
        const start = i;
        while (i < s.len and spaceAt(s, i) == null) i += 1;
        try out.append(a, .{ .string = s[start..i] });
        splits += 1;
    }
    return out.items;
}

fn asciiCase(a: Allocator, s: []const u8, upper: bool) Allocator.Error![]const u8 {
    const out = try a.dupe(u8, s);
    for (out) |*c| c.* = if (upper) std.ascii.toUpper(c.*) else std.ascii.toLower(c.*);
    return out;
}

/// Python's repr(v) (str(v) for anything but strings inside containers).
fn writeRepr(a: Allocator, out: *std.ArrayList(u8), v: Value) Error!void {
    switch (v) {
        .undefined => {},
        .none => try out.appendSlice(a, "None"),
        .boolean => |b| try out.appendSlice(a, if (b) "True" else "False"),
        .integer => |i| try out.print(a, "{d}", .{i}),
        .float => |f| try writeFloat(a, out, f),
        .number => |s| try out.appendSlice(a, s),
        .string => |s| try writeStrRepr(a, out, s),
        .list => |l| {
            try out.append(a, '[');
            for (l, 0..) |item, i| {
                if (i > 0) try out.appendSlice(a, ", ");
                try writeRepr(a, out, item);
            }
            try out.append(a, ']');
        },
        .dict => |d| {
            try out.append(a, '{');
            for (d.keys.items, d.values.items, 0..) |k, item, i| {
                if (i > 0) try out.appendSlice(a, ", ");
                try writeStrRepr(a, out, k);
                try out.appendSlice(a, ": ");
                try writeRepr(a, out, item);
            }
            try out.append(a, '}');
        },
        .namespace => try out.appendSlice(a, "<Namespace>"),
        .callable => try out.appendSlice(a, "<callable>"),
    }
}

fn writeStrRepr(a: Allocator, out: *std.ArrayList(u8), s: []const u8) Allocator.Error!void {
    const quote: u8 = if (std.mem.indexOfScalar(u8, s, '\'') != null and std.mem.indexOfScalar(u8, s, '"') == null) '"' else '\'';
    try out.append(a, quote);
    for (s) |c| switch (c) {
        '\\' => try out.appendSlice(a, "\\\\"),
        '\n' => try out.appendSlice(a, "\\n"),
        '\r' => try out.appendSlice(a, "\\r"),
        '\t' => try out.appendSlice(a, "\\t"),
        0...8, 0x0b, 0x0c, 0x0e...0x1f, 0x7f => try out.print(a, "\\x{x:0>2}", .{c}),
        else => {
            if (c == quote) try out.append(a, '\\');
            try out.append(a, c);
        },
    };
    try out.append(a, quote);
}

/// Python's repr of a float: the shortest round-trip digits, positional
/// for exponents −4 to 15, else d.ddde±XX.
fn writeFloat(a: Allocator, out: *std.ArrayList(u8), f: f64) Allocator.Error!void {
    if (std.math.isNan(f)) return out.appendSlice(a, "nan");
    if (std.math.isInf(f)) return out.appendSlice(a, if (f > 0) "inf" else "-inf");
    var buf: [64]u8 = undefined;
    const sci = std.fmt.bufPrint(&buf, "{e}", .{f}) catch unreachable;
    const e_at = std.mem.indexOfScalar(u8, sci, 'e').?;
    const exp = std.fmt.parseInt(i32, sci[e_at + 1 ..], 10) catch unreachable;
    var mant = sci[0..e_at];
    if (mant[0] == '-') {
        try out.append(a, '-');
        mant = mant[1..];
    }
    var digits: [32]u8 = undefined;
    var nd: usize = 0;
    for (mant) |c| if (c != '.') {
        digits[nd] = c;
        nd += 1;
    };
    while (nd > 1 and digits[nd - 1] == '0') nd -= 1;
    const d = digits[0..nd];
    if (exp >= -4 and exp < 16) {
        if (exp < 0) {
            try out.appendSlice(a, "0.");
            try out.appendNTimes(a, '0', @intCast(-exp - 1));
            try out.appendSlice(a, d);
        } else {
            const point: usize = @intCast(exp + 1);
            if (d.len <= point) {
                try out.appendSlice(a, d);
                try out.appendNTimes(a, '0', point - d.len);
                try out.appendSlice(a, ".0");
            } else {
                try out.appendSlice(a, d[0..point]);
                try out.append(a, '.');
                try out.appendSlice(a, d[point..]);
            }
        }
        return;
    }
    try out.append(a, d[0]);
    if (d.len > 1) {
        try out.append(a, '.');
        try out.appendSlice(a, d[1..]);
    }
    try out.print(a, "e{c}{d:0>2}", .{ @as(u8, if (exp < 0) '-' else '+'), @abs(exp) });
}

pub const JsonOptions = struct {
    ensure_ascii: bool = false,
    indent: ?usize = null,
    sort_keys: bool = false,
    /// (item, key) separators; by default ", " and ": ", or "," with an indent.
    separators: ?[2][]const u8 = null,
};

/// Python's json.dumps(v, ...): transformers' tojson.
pub fn writeJson(a: Allocator, out: *std.ArrayList(u8), v: Value, o: JsonOptions, depth: usize) Error!void {
    const seps = o.separators orelse [2][]const u8{ if (o.indent != null) "," else ", ", ": " };
    switch (v) {
        .none => try out.appendSlice(a, "null"),
        .boolean => |b| try out.appendSlice(a, if (b) "true" else "false"),
        .integer => |i| try out.print(a, "{d}", .{i}),
        .float => |f| {
            if (std.math.isNan(f)) return out.appendSlice(a, "NaN");
            if (std.math.isInf(f)) return out.appendSlice(a, if (f > 0) "Infinity" else "-Infinity");
            try writeFloat(a, out, f);
        },
        .number => |s| try out.appendSlice(a, s),
        .string => |s| try writeJsonString(a, out, s, o.ensure_ascii),
        .list => |l| {
            if (l.len == 0) return out.appendSlice(a, "[]");
            try out.append(a, '[');
            for (l, 0..) |item, i| {
                if (i > 0) try out.appendSlice(a, seps[0]);
                try newline(a, out, o, depth + 1);
                try writeJson(a, out, item, o, depth + 1);
            }
            try newline(a, out, o, depth);
            try out.append(a, ']');
        },
        .dict => |d| {
            if (d.keys.items.len == 0) return out.appendSlice(a, "{}");
            const order_ = try a.alloc(usize, d.keys.items.len);
            for (order_, 0..) |*o_, i| o_.* = i;
            if (o.sort_keys) std.mem.sort(usize, order_, d, struct {
                fn lt(dd: *Dict, x: usize, y: usize) bool {
                    return std.mem.lessThan(u8, dd.keys.items[x], dd.keys.items[y]);
                }
            }.lt);
            try out.append(a, '{');
            for (order_, 0..) |k, i| {
                if (i > 0) try out.appendSlice(a, seps[0]);
                try newline(a, out, o, depth + 1);
                try writeJsonString(a, out, d.keys.items[k], o.ensure_ascii);
                try out.appendSlice(a, seps[1]);
                try writeJson(a, out, d.values.items[k], o, depth + 1);
            }
            try newline(a, out, o, depth);
            try out.append(a, '}');
        },
        // json.dumps rejects jinja2's Undefined and anything else.
        .undefined, .namespace, .callable => return error.TemplateError,
    }
}

fn newline(a: Allocator, out: *std.ArrayList(u8), o: JsonOptions, depth: usize) Allocator.Error!void {
    const n = o.indent orelse return;
    try out.append(a, '\n');
    try out.appendNTimes(a, ' ', n * depth);
}

fn writeJsonString(a: Allocator, out: *std.ArrayList(u8), s: []const u8, ascii: bool) Allocator.Error!void {
    try out.append(a, '"');
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        switch (c) {
            '"' => try out.appendSlice(a, "\\\""),
            '\\' => try out.appendSlice(a, "\\\\"),
            '\n' => try out.appendSlice(a, "\\n"),
            '\r' => try out.appendSlice(a, "\\r"),
            '\t' => try out.appendSlice(a, "\\t"),
            0x08 => try out.appendSlice(a, "\\b"),
            0x0c => try out.appendSlice(a, "\\f"),
            0...7, 0x0b, 0x0e...0x1f => try out.print(a, "\\u{x:0>4}", .{c}),
            else => if (c < 0x80 or !ascii) {
                try out.append(a, c);
            } else {
                const n = std.unicode.utf8ByteSequenceLength(c) catch 1;
                const cp = if (i + n <= s.len) std.unicode.utf8Decode(s[i..][0..n]) catch 0xfffd else 0xfffd;
                if (cp > 0xffff) {
                    const u = cp - 0x10000;
                    try out.print(a, "\\u{x:0>4}\\u{x:0>4}", .{ 0xd800 + (u >> 10), 0xdc00 + (u & 0x3ff) });
                } else try out.print(a, "\\u{x:0>4}", .{cp});
                i += n;
                continue;
            },
        }
        i += 1;
    }
    try out.append(a, '"');
}

fn strftime(a: Allocator, fmt: []const u8, d: Date) Error![]const u8 {
    const months = [_][]const u8{ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" };
    const days = [_][]const u8{ "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday" };
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < fmt.len) : (i += 1) {
        if (fmt[i] != '%' or i + 1 >= fmt.len) {
            try out.append(a, fmt[i]);
            continue;
        }
        i += 1;
        switch (fmt[i]) {
            'd' => try out.print(a, "{d:0>2}", .{d.day}),
            'm' => try out.print(a, "{d:0>2}", .{d.month}),
            'Y' => try out.print(a, "{d}", .{d.year}),
            'y' => try out.print(a, "{d:0>2}", .{d.year % 100}),
            'b' => try out.appendSlice(a, months[d.month - 1][0..3]),
            'B' => try out.appendSlice(a, months[d.month - 1]),
            'a' => try out.appendSlice(a, days[d.weekday][0..3]),
            'A' => try out.appendSlice(a, days[d.weekday]),
            'H' => try out.print(a, "{d:0>2}", .{d.hour}),
            'M' => try out.print(a, "{d:0>2}", .{d.minute}),
            'S' => try out.print(a, "{d:0>2}", .{d.second}),
            '%' => try out.append(a, '%'),
            else => return error.TemplateError,
        }
    }
    return out.items;
}

/// A JSON value (std.json, numbers as written) as a template value.
pub fn fromJson(a: Allocator, v: std.json.Value) Allocator.Error!Value {
    return switch (v) {
        .null => .none,
        .bool => |b| .{ .boolean = b },
        .integer => |i| .{ .integer = i },
        .float => |f| .{ .float = f },
        .number_string => |s| .{ .number = s },
        .string => |s| .{ .string = s },
        .array => |arr| blk: {
            const out = try a.alloc(Value, arr.items.len);
            for (out, arr.items) |*o, item| o.* = try fromJson(a, item);
            break :blk .{ .list = out };
        },
        .object => |obj| blk: {
            const d = try a.create(Dict);
            d.* = .{};
            var it = obj.iterator();
            while (it.next()) |e| try d.put(a, e.key_ptr.*, try fromJson(a, e.value_ptr.*));
            break :blk .{ .dict = d };
        },
    };
}

fn testRender(source: []const u8, globals: []const Template.Global, want: []const u8) !void {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const t = try Template.parse(a, source);
    var out: std.ArrayList(u8) = .empty;
    var raised: ?[]const u8 = null;
    try t.render(a, globals, .{ .year = 2026, .month = 10, .day = 30 }, &out, &raised);
    try std.testing.expectEqualStrings(want, out.items);
}

test "jinja whitespace control" {
    try testRender("a\n{% if true %}\n  b\n{% endif %}\nc\n", &.{}, "a\n  b\nc");
    try testRender("a  {%- if true -%}  b  {%- endif %} c", &.{}, "ab c");
    try testRender("x\n    {% if true %}y{% endif %}", &.{}, "x\ny");
    try testRender("x {{ 1 }}\n{# note #}\nz", &.{}, "x 1\nz");
    try testRender("{%+ if true %}k{% endif %}", &.{}, "k");
}

test "jinja values print as Python's" {
    try testRender("{{ none }} {{ true }} {{ 1.5 }} {{ 1e16 }} {{ 0.0001 }} {{ 1e-5 }} {{ 3 / 2 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ -7 % 3 }}", &.{}, "None True 1.5 1e+16 0.0001 1e-05 1.5 3 -4 2");
    try testRender("{{ ['a', \"b'c\", 1, none] }} {{ {'k': [true]} }}", &.{}, "['a', \"b'c\", 1, None] {'k': [True]}");
    try testRender("{{ 'x' ~ 1 ~ undefined_name }}|{{ 'ab' * 2 }}", &.{}, "x1|abab");
}

test "jinja strings index by code point" {
    try testRender("{{ 'héllo'[1] }} {{ 'héllo'[1:3] }} {{ 'héllo'|length }} {{ 'héllo'[::-1] }} {{ 'abc'[-1] }}", &.{}, "é él 5 olléh c");
}

test "jinja loops, scopes and namespaces" {
    try testRender("{% set x = 1 %}{% for i in [1, 2, 3] %}{% set x = i %}{{ loop.index0 }}{{ loop.last }}{% endfor %}{{ x }}", &.{}, "0False1False2True1");
    try testRender("{% set ns = namespace(n=0) %}{% for i in range(4) if i is even %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}", &.{}, "2");
    try testRender("{% for i in [] %}a{% else %}empty{% endfor %}", &.{}, "empty");
    try testRender("{% for i in [1, 2, 3] %}{% if i == 2 %}{% break %}{% endif %}{{ i }}{% endfor %}", &.{}, "1");
    try testRender("{% for k, v in {'a': 1, 'b': 2}.items() %}{{ k }}={{ v }};{% endfor %}", &.{}, "a=1;b=2;");
}

test "jinja macros, filters and tests" {
    try testRender("{% macro m(a, b='d') %}<{{ a }}{{ b }}>{% endmacro %}{{ m(1) }}{{ m(1, b=2) }}", &.{}, "<1d><12>");
    try testRender("{{ '  x  '|trim }}|{{ [1, 2]|first }}{{ [1, 2]|last }}|{{ none|default('d', true) }}|{{ undefined_name|default('u') }}", &.{}, "x|12|d|u");
    try testRender("{{ x is defined }} {{ x is not defined }} {{ 'a' is string }} {{ {} is mapping }} {{ 'a' is iterable }} {{ none is none }}", &.{}, "False True True True True True");
    try testRender("{{ 'a b  c'.split() }} {{ 'a,b'.split(',') }} {{ 'x</think>y'.split('</think>')|last }} {{ '\\n\\nhi\\n'.lstrip('\\n') }}", &.{}, "['a', 'b', 'c'] ['a', 'b'] y hi\n");
}

test "jinja dictsort preserves pairs and stable case-insensitive ordering" {
    try testRender("{% for k, v in {'b': 2, 'A': 1, 'a': 3, 'B': 4}|dictsort %}{{ k }}={{ v }};{% endfor %}", &.{}, "A=1;a=3;b=2;B=4;");
    try testRender("{% for k, v in {'b': 2, 'A': 1, 'a': 3, 'B': 4}|dictsort(reverse=true) %}{{ k }}={{ v }};{% endfor %}", &.{}, "b=2;B=4;A=1;a=3;");
    try testRender("{% for k, v in {'b': 2, 'A': 1, 'a': 3, 'B': 4}|dictsort(case_sensitive=true) %}{{ k }}={{ v }};{% endfor %}", &.{}, "A=1;B=4;a=3;b=2;");
    try testRender("{% for k, v in {'b': 2, 'A': 1, 'a': 3}|dictsort(false, 'value', true) %}{{ k }}={{ v }};{% endfor %}", &.{}, "a=3;b=2;A=1;");
    try testRender("{% for k, v in {'z': 'apple', 'a': 'Banana'}|dictsort(by='value') %}{{ k }}={{ v }};{% endfor %}", &.{}, "z=apple;a=Banana;");
    try testRender("{{ {}|dictsort|length }}|{% for k, v in {'z': {'x': 2}}|dictsort %}{{ k }}:{{ v.x }}{% endfor %}", &.{}, "0|z:2");
}

test "jinja dictsort argument and input mutations fail closed" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for ([_][]const u8{
        "{{ []|dictsort }}",
        "{{ missing|dictsort }}",
        "{{ {}|dictsort(by='invalid') }}",
        "{{ {}|dictsort(by=1) }}",
        "{{ {}|dictsort(unknown=true) }}",
        "{{ {}|dictsort(false, case_sensitive=true) }}",
        "{{ {}|dictsort(false, 'key', false, true) }}",
        "{{ {'a': 1, 'b': 'x'}|dictsort(by='value') }}",
    }) |source| {
        const t = try Template.parse(a, source);
        var out: std.ArrayList(u8) = .empty;
        var raised: ?[]const u8 = null;
        try std.testing.expectError(error.TemplateError, t.render(a, &.{}, null, &out, &raised));
    }
}

test "jinja tojson is transformers' json.dumps" {
    try testRender("{{ {'a': 'é\"', 'b': [1, none, true]}|tojson }}", &.{}, "{\"a\": \"é\\\"\", \"b\": [1, null, true]}");
    try testRender("{{ {'a': 'é'}|tojson(ensure_ascii=True) }}", &.{}, "{\"a\": \"\\u00e9\"}");
    try testRender("{{ {'a': [1, 2], 'b': {}}|tojson(indent=2) }}", &.{}, "{\n  \"a\": [\n    1,\n    2\n  ],\n  \"b\": {}\n}");
    try testRender("{{ strftime_now('%d %b %Y') }}", &.{}, "30 Oct 2026");
}

test "jinja errors" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var out: std.ArrayList(u8) = .empty;
    var raised: ?[]const u8 = null;
    const t = try Template.parse(a, "{{ raise_exception('no ' ~ 'way') }}");
    try std.testing.expectError(error.Raised, t.render(a, &.{}, null, &out, &raised));
    try std.testing.expectEqualStrings("no way", raised.?);
    const u = try Template.parse(a, "{{ x|nosuchfilter }}");
    try std.testing.expectError(error.TemplateError, u.render(a, &.{}, null, &out, &raised));
    try std.testing.expectError(error.TemplateSyntax, Template.parse(a, "{% if x %}"));
    try std.testing.expectError(error.TemplateError, (try Template.parse(a, "{{ x.y.z }}")).render(a, &.{}, null, &out, &raised));
}
