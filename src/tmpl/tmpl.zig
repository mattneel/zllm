//! The Handlebars subset kernel templates are written in (spec §4): kernel
//! parameters are baked into the WGSL this renders, shapes stay in uniforms.
//!
//!     {{x}}                        a value: integer, boolean, or string
//!     {{helper a b}}               add, mul, ceil_div, vec_type, enable
//!     {{add (mul a 2) 1}}          sub-expressions in parentheses
//!     {{#if x}}…{{else}}…{{/if}}   x true, nonzero, or non-empty
//!     {{#each xs}}…{{/each}}       {{this}} and {{@index}} for each element
//!     {{#range a b}}…{{/range}}    {{this}} = a, a + 1, …, b − 1
//!     {{> name}}                   a partial, rendered in the current scope
//!     {{! comment}}                nothing
//!
//! Literals are integers and "strings". There is no other code: templates
//! cannot loop without a bound or call anything but the helpers.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Value = union(enum) {
    int: i64,
    boolean: bool,
    str: []const u8,
    list: []const Value,

    fn truthy(v: Value) bool {
        return switch (v) {
            .int => |i| i != 0,
            .boolean => |b| b,
            .str => |s| s.len > 0,
            .list => |l| l.len > 0,
        };
    }
};

pub const Binding = struct { name: []const u8, value: Value };
pub const Partial = struct { name: []const u8, source: []const u8 };

pub const Error = Allocator.Error || error{
    /// Malformed tag, expression or block nesting.
    Syntax,
    UnknownName,
    UnknownHelper,
    UnknownPartial,
    /// A value of the wrong kind for its use (a list printed, a string added).
    Type,
    Overflow,
    DivisionByZero,
    /// Partials nested past max_partial_depth (a partial including itself).
    PartialDepth,
};

pub const max_partial_depth = 16;

/// Renders `source` with `vars` and `partials`, appending to `out`.
pub fn render(gpa: Allocator, source: []const u8, vars: []const Binding, partials: []const Partial, out: *std.ArrayList(u8)) Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const r: Renderer = .{ .gpa = gpa, .arena = arena_state.allocator(), .vars = vars, .partials = partials, .out = out };
    try r.template(source, null, 0);
}

const Helper = enum { add, mul, ceil_div, vec_type, enable };

const Expr = union(enum) {
    int: i64,
    str: []const u8,
    name: []const u8,
    call: struct { helper: Helper, args: []const Expr },
};

const Node = union(enum) {
    text: []const u8,
    expr: Expr,
    @"if": struct { cond: Expr, then: []const Node, otherwise: []const Node },
    each: struct { list: Expr, body: []const Node },
    range: struct { from: Expr, to: Expr, body: []const Node },
    partial: []const u8,
};

/// Parses a template into nodes.
const Parser = struct {
    arena: Allocator,
    src: []const u8,
    pos: usize = 0,

    const Stop = enum { end, close, @"else" };

    /// Nodes up to the end, a `{{/…}}` (its name checked against `block`),
    /// or a top-level `{{else}}` inside an if.
    fn nodes(p: *Parser, closer: ?[]const u8, stop: *Stop) Error![]const Node {
        var list: std.ArrayList(Node) = .empty;
        while (p.pos < p.src.len) {
            const open = std.mem.indexOfPos(u8, p.src, p.pos, "{{") orelse {
                try list.append(p.arena, .{ .text = p.src[p.pos..] });
                p.pos = p.src.len;
                break;
            };
            if (open > p.pos) try list.append(p.arena, .{ .text = p.src[p.pos..open] });
            const close = std.mem.indexOfPos(u8, p.src, open + 2, "}}") orelse return error.Syntax;
            const tag = std.mem.trim(u8, p.src[open + 2 .. close], " \t\r\n");
            p.pos = close + 2;
            if (tag.len == 0) return error.Syntax;
            switch (tag[0]) {
                '!' => {},
                '/' => {
                    const name = std.mem.trim(u8, tag[1..], " ");
                    if (closer == null or !std.mem.eql(u8, name, closer.?)) return error.Syntax;
                    stop.* = .close;
                    return list.items;
                },
                '>' => try list.append(p.arena, .{ .partial = std.mem.trim(u8, tag[1..], " ") }),
                '#' => try list.append(p.arena, try p.block(tag[1..])),
                else => if (std.mem.eql(u8, tag, "else")) {
                    if (closer == null or !std.mem.eql(u8, closer.?, "if")) return error.Syntax;
                    stop.* = .@"else";
                    return list.items;
                } else {
                    try list.append(p.arena, .{ .expr = try p.expr(tag) });
                },
            }
        }
        if (closer != null) return error.Syntax;
        stop.* = .end;
        return list.items;
    }

    fn block(p: *Parser, tag: []const u8) Error!Node {
        const sp = std.mem.indexOfScalar(u8, tag, ' ') orelse return error.Syntax;
        const kind = tag[0..sp];
        const rest = std.mem.trim(u8, tag[sp..], " ");
        var stop: Stop = undefined;
        if (std.mem.eql(u8, kind, "if")) {
            const cond = try p.expr(rest);
            const then = try p.nodes("if", &stop);
            const otherwise = if (stop == .@"else") try p.nodes("if", &stop) else &.{};
            if (stop != .close) return error.Syntax;
            return .{ .@"if" = .{ .cond = cond, .then = then, .otherwise = otherwise } };
        }
        if (std.mem.eql(u8, kind, "each")) {
            const list = try p.expr(rest);
            const body = try p.nodes("each", &stop);
            return .{ .each = .{ .list = list, .body = body } };
        }
        if (std.mem.eql(u8, kind, "range")) {
            const bounds = try p.atoms(rest);
            if (bounds.len != 2) return error.Syntax;
            const body = try p.nodes("range", &stop);
            return .{ .range = .{ .from = bounds[0], .to = bounds[1], .body = body } };
        }
        return error.Syntax;
    }

    /// `name`, a literal, or `helper arg …`.
    fn expr(p: *Parser, text: []const u8) Error!Expr {
        const parts = try p.atoms(text);
        if (parts.len == 1) return parts[0];
        if (parts.len == 0 or parts[0] != .name) return error.Syntax;
        const helper = std.meta.stringToEnum(Helper, parts[0].name) orelse return error.UnknownHelper;
        return .{ .call = .{ .helper = helper, .args = parts[1..] } };
    }

    /// Space-separated atoms: names, integers, "strings", (sub-expressions).
    fn atoms(p: *Parser, text: []const u8) Error![]const Expr {
        var list: std.ArrayList(Expr) = .empty;
        var i: usize = 0;
        while (i < text.len) {
            const c = text[i];
            if (c == ' ') {
                i += 1;
                continue;
            }
            if (c == '(') {
                var depth: usize = 1;
                var j = i + 1;
                while (j < text.len and depth > 0) : (j += 1) {
                    if (text[j] == '(') depth += 1;
                    if (text[j] == ')') depth -= 1;
                }
                if (depth != 0) return error.Syntax;
                try list.append(p.arena, try p.expr(std.mem.trim(u8, text[i + 1 .. j - 1], " ")));
                i = j;
            } else if (c == '"') {
                const end = std.mem.indexOfScalarPos(u8, text, i + 1, '"') orelse return error.Syntax;
                try list.append(p.arena, .{ .str = text[i + 1 .. end] });
                i = end + 1;
            } else {
                const end = std.mem.indexOfAnyPos(u8, text, i, " ()\"") orelse text.len;
                const word = text[i..end];
                if (word.len == 0) return error.Syntax;
                if (word[0] == '-' or std.ascii.isDigit(word[0])) {
                    try list.append(p.arena, .{ .int = std.fmt.parseInt(i64, word, 10) catch return error.Syntax });
                } else {
                    try list.append(p.arena, .{ .name = word });
                }
                i = end;
            }
        }
        return list.items;
    }
};

/// `this` and `@index` inside each and range, innermost first.
const Scope = struct {
    this: Value,
    index: i64,
    outer: ?*const Scope,
};

const Renderer = struct {
    gpa: Allocator,
    arena: Allocator,
    vars: []const Binding,
    partials: []const Partial,
    out: *std.ArrayList(u8),

    fn template(r: Renderer, source: []const u8, scope: ?*const Scope, depth: usize) Error!void {
        if (depth > max_partial_depth) return error.PartialDepth;
        var p: Parser = .{ .arena = r.arena, .src = source };
        var stop: Parser.Stop = undefined;
        try r.nodes(try p.nodes(null, &stop), scope, depth);
    }

    fn nodes(r: Renderer, list: []const Node, scope: ?*const Scope, depth: usize) Error!void {
        for (list) |n| switch (n) {
            .text => |t| try r.out.appendSlice(r.gpa, t),
            .expr => |e| try r.print(try r.eval(e, scope)),
            .@"if" => |b| try r.nodes(if ((try r.eval(b.cond, scope)).truthy()) b.then else b.otherwise, scope, depth),
            .each => |b| {
                const list_value = try r.eval(b.list, scope);
                if (list_value != .list) return error.Type;
                for (list_value.list, 0..) |v, i| {
                    const inner: Scope = .{ .this = v, .index = @intCast(i), .outer = scope };
                    try r.nodes(b.body, &inner, depth);
                }
            },
            .range => |b| {
                const from = try r.int(b.from, scope);
                const to = try r.int(b.to, scope);
                var i = from;
                while (i < to) : (i += 1) {
                    const inner: Scope = .{ .this = .{ .int = i }, .index = i - from, .outer = scope };
                    try r.nodes(b.body, &inner, depth);
                }
            },
            .partial => |name| {
                const source = for (r.partials) |pt| {
                    if (std.mem.eql(u8, pt.name, name)) break pt.source;
                } else return error.UnknownPartial;
                try r.template(source, scope, depth + 1);
            },
        };
    }

    fn print(r: Renderer, v: Value) Error!void {
        switch (v) {
            .int => |i| try r.out.print(r.gpa, "{d}", .{i}),
            .boolean => |b| try r.out.appendSlice(r.gpa, if (b) "true" else "false"),
            .str => |s| try r.out.appendSlice(r.gpa, s),
            .list => return error.Type,
        }
    }

    fn int(r: Renderer, e: Expr, scope: ?*const Scope) Error!i64 {
        const v = try r.eval(e, scope);
        return if (v == .int) v.int else error.Type;
    }

    fn eval(r: Renderer, e: Expr, scope: ?*const Scope) Error!Value {
        switch (e) {
            .int => |i| return .{ .int = i },
            .str => |s| return .{ .str = s },
            .name => |n| {
                if (std.mem.eql(u8, n, "this") or std.mem.eql(u8, n, "@index")) {
                    const s = scope orelse return error.UnknownName;
                    return if (n[0] == '@') .{ .int = s.index } else s.this;
                }
                for (r.vars) |b| if (std.mem.eql(u8, b.name, n)) return b.value;
                return error.UnknownName;
            },
            .call => |c| return r.call(c.helper, c.args, scope),
        }
    }

    fn call(r: Renderer, helper: Helper, args: []const Expr, scope: ?*const Scope) Error!Value {
        switch (helper) {
            .add, .mul => {
                if (args.len < 2) return error.Syntax;
                var acc = try r.int(args[0], scope);
                for (args[1..]) |a| {
                    const v = try r.int(a, scope);
                    acc = if (helper == .add) try std.math.add(i64, acc, v) else try std.math.mul(i64, acc, v);
                }
                return .{ .int = acc };
            },
            .ceil_div => {
                if (args.len != 2) return error.Syntax;
                const a = try r.int(args[0], scope);
                const b = try r.int(args[1], scope);
                if (b == 0) return error.DivisionByZero;
                return .{ .int = try std.math.divCeil(i64, a, b) };
            },
            .vec_type => {
                // vec_type n [scalar]: the scalar for 1, else vecN<scalar>.
                if (args.len < 1 or args.len > 2) return error.Syntax;
                const n = try r.int(args[0], scope);
                const scalar: []const u8 = if (args.len == 2) blk: {
                    const s = try r.eval(args[1], scope);
                    break :blk if (s == .str) s.str else return error.Type;
                } else "f32";
                if (n < 1 or n > 4) return error.Type;
                if (n == 1) return .{ .str = scalar };
                return .{ .str = try std.fmt.allocPrint(r.arena, "vec{d}<{s}>", .{ n, scalar }) };
            },
            .enable => {
                // enable "feature": the WGSL directive.
                if (args.len != 1) return error.Syntax;
                const f = try r.eval(args[0], scope);
                if (f != .str) return error.Type;
                return .{ .str = try std.fmt.allocPrint(r.arena, "enable {s};", .{f.str}) };
            },
        }
    }
};

fn expectRender(want: []const u8, source: []const u8, vars: []const Binding, partials: []const Partial) !void {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try render(gpa, source, vars, partials, &out);
    try std.testing.expectEqualStrings(want, out.items);
}

test "values, helpers and sub-expressions" {
    const vars = [_]Binding{
        .{ .name = "rows", .value = .{ .int = 8 } },
        .{ .name = "f16", .value = .{ .boolean = true } },
        .{ .name = "name", .value = .{ .str = "matvec" } },
    };
    try expectRender("const ROWS: u32 = 8u; // matvec", "const ROWS: u32 = {{rows}}u; // {{name}}", &vars, &.{});
    try expectRender("33 64 3", "{{add rows 25}} {{mul rows 2 4}} {{ceil_div 17 rows}}", &vars, &.{});
    try expectRender("17", "{{add (mul rows 2) 1}}", &vars, &.{});
    try expectRender("vec4<f32> f32 vec2<f16>", "{{vec_type 4}} {{vec_type 1}} {{vec_type 2 \"f16\"}}", &vars, &.{});
    try expectRender("enable f16;\n", "{{#if f16}}{{enable \"f16\"}}\n{{/if}}", &vars, &.{});
    try expectRender("-3 true", "{{add -5 2}} {{f16}}", &vars, &.{});
}

test "blocks: if/else, each, range, nesting" {
    const items = [_]Value{ .{ .int = 3 }, .{ .int = 5 } };
    const vars = [_]Binding{
        .{ .name = "on", .value = .{ .int = 0 } },
        .{ .name = "xs", .value = .{ .list = &items } },
        .{ .name = "n", .value = .{ .int = 3 } },
    };
    try expectRender("b", "{{#if on}}a{{else}}b{{/if}}", &vars, &.{});
    try expectRender("x", "{{#if xs}}x{{/if}}", &vars, &.{});
    try expectRender("0:3 1:5 ", "{{#each xs}}{{@index}}:{{this}} {{/each}}", &vars, &.{});
    try expectRender("[1 2] ", "{{#range 1 n}}{{#if (add this -1)}}{{this}}]{{else}}[{{this}} {{/if}}{{/range}} ", &vars, &.{});
    // Inner blocks of the same kind close in order; each sees its own `this`.
    try expectRender("3:012 5:01234 ", "{{#each xs}}{{this}}:{{#range 0 this}}{{this}}{{/range}} {{/each}}", &vars, &.{});
    try expectRender("", "{{#range n 1}}never{{/range}}{{! a comment }}", &vars, &.{});
}

test "partials see the current scope" {
    const items = [_]Value{ .{ .int = 7 }, .{ .int = 9 } };
    const vars = [_]Binding{.{ .name = "xs", .value = .{ .list = &items } }};
    const partials = [_]Partial{.{ .name = "item", .source = "<{{this}}>" }};
    try expectRender("<7><9>", "{{#each xs}}{{> item}}{{/each}}", &vars, &partials);
}

test "errors" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    const items = [_]Value{.{ .int = 1 }};
    const vars = [_]Binding{ .{ .name = "xs", .value = .{ .list = &items } }, .{ .name = "s", .value = .{ .str = "a" } } };
    const self = [_]Partial{.{ .name = "loop", .source = "{{> loop}}" }};
    try std.testing.expectError(error.Syntax, render(gpa, "{{#if xs}}open", &vars, &.{}, &out));
    try std.testing.expectError(error.Syntax, render(gpa, "{{#if xs}}a{{/each}}", &vars, &.{}, &out));
    try std.testing.expectError(error.Syntax, render(gpa, "{{/if}}", &vars, &.{}, &out));
    try std.testing.expectError(error.Syntax, render(gpa, "{{#each xs}}{{else}}{{/each}}", &vars, &.{}, &out));
    try std.testing.expectError(error.Syntax, render(gpa, "{{x", &vars, &.{}, &out));
    try std.testing.expectError(error.UnknownName, render(gpa, "{{nope}}", &vars, &.{}, &out));
    try std.testing.expectError(error.UnknownName, render(gpa, "{{this}}", &vars, &.{}, &out));
    try std.testing.expectError(error.UnknownHelper, render(gpa, "{{sub 1 2}}", &vars, &.{}, &out));
    try std.testing.expectError(error.UnknownPartial, render(gpa, "{{> nope}}", &vars, &.{}, &out));
    try std.testing.expectError(error.Type, render(gpa, "{{xs}}", &vars, &.{}, &out));
    try std.testing.expectError(error.Type, render(gpa, "{{add s 1}}", &vars, &.{}, &out));
    try std.testing.expectError(error.Type, render(gpa, "{{vec_type 5}}", &vars, &.{}, &out));
    try std.testing.expectError(error.DivisionByZero, render(gpa, "{{ceil_div 1 0}}", &vars, &.{}, &out));
    try std.testing.expectError(error.Overflow, render(gpa, "{{mul 9223372036854775807 2}}", &vars, &.{}, &out));
    try std.testing.expectError(error.PartialDepth, render(gpa, "{{> loop}}", &vars, &self, &out));
}
