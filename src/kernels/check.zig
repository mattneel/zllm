//! GPU kernel checks: each kernel on seeded random inputs against a float64
//! CPU computation of the same formula. Shapes are chosen to hit tile and
//! chunk remainders. Run with `zllm gpu-check` on any Dawn host.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cmd = @import("../gpu/cmd.zig");
const Host = @import("../gpu/host.zig").Host;
const kernels = @import("kernels.zig");
const P = kernels.params;

pub const Result = struct { name: []const u8, max_rel_err: f64, tolerance: f64 };

/// Runs every check, reporting one line per check to `out`. Returns whether
/// all passed.
pub fn run(gpa: Allocator, host: *Host, out: *std.Io.Writer) !bool {
    var ctx: Ctx = .{ .gpa = gpa, .host = host, .rec = .{} };
    defer ctx.rec.deinit(gpa);
    ctx.set = try kernels.Set.create(gpa, &ctx.rec);
    try ctx.flush();

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    ctx.arena = arena_state.allocator();
    ctx.rand = prng.random();

    var ok = true;
    inline for (.{ checkRmsNorm, checkMatmul, checkMatmulChunked, checkSiluMul, checkAdd, checkEmbed }) |f| {
        const r = try f(&ctx);
        const pass = r.max_rel_err <= r.tolerance;
        ok = ok and pass;
        try out.print("{s} {s:<24} max rel err {e:.2} (tolerance {e:.0})\n", .{ if (pass) "PASS" else "FAIL", r.name, r.max_rel_err, r.tolerance });
        _ = arena_state.reset(.retain_capacity);
    }
    return ok;
}

const Ctx = struct {
    gpa: Allocator,
    arena: Allocator = undefined,
    host: *Host,
    rec: cmd.Recorder,
    set: kernels.Set = undefined,
    rand: std.Random = undefined,
    token: u32 = 0,

    fn launcher(c: *Ctx) kernels.Launcher {
        return .{ .gpa = c.gpa, .arena = c.arena, .rec = &c.rec, .set = &c.set };
    }

    fn flush(c: *Ctx) !void {
        var completions: std.ArrayList(cmd.Completion) = .empty;
        defer completions.deinit(c.gpa);
        c.token += 1;
        try c.rec.submit(c.gpa, c.token);
        c.host.execute(c.rec.batch(), &completions) catch |err| {
            if (err == error.GpuError or err == error.DeviceLost) std.log.err("{s}", .{c.host.d.errorMessage()});
            return err;
        };
        c.rec.clear();
    }

    fn random(c: *Ctx, n: usize, scale: f32) ![]f32 {
        const v = try c.arena.alloc(f32, n);
        for (v) |*x| x.* = (c.rand.float(f32) * 2 - 1) * scale;
        return v;
    }

    /// Random bf16 values packed two per u32, and their f32 values.
    fn randomBf16(c: *Ctx, n: usize, scale: f32) !struct { words: []u32, values: []f32 } {
        std.debug.assert(n % 2 == 0);
        const words = try c.arena.alloc(u32, n / 2);
        const values = try c.arena.alloc(f32, n);
        for (values, 0..) |*v, i| {
            const f = (c.rand.float(f32) * 2 - 1) * scale;
            const bits: u16 = @truncate(@as(u32, @bitCast(f)) >> 16);
            v.* = @bitCast(@as(u32, bits) << 16);
            if (i % 2 == 0) words[i / 2] = bits else words[i / 2] |= @as(u32, bits) << 16;
        }
        return .{ .words = words, .values = values };
    }

    fn upload(c: *Ctx, data: anytype) !kernels.Arg {
        const bytes = std.mem.sliceAsBytes(data);
        const size = std.mem.alignForward(u64, bytes.len, 4);
        const buf = try c.rec.createBuffer(c.gpa, size, .{ .storage = true, .copy_dst = true, .copy_src = true });
        try c.rec.writeBuffer(c.gpa, buf, 0, bytes);
        return .{ .buffer = buf, .size = size };
    }

    fn output(c: *Ctx, n: usize) !kernels.Arg {
        const size = n * 4;
        const buf = try c.rec.createBuffer(c.gpa, size, .{ .storage = true, .copy_src = true });
        return .{ .buffer = buf, .size = size };
    }

    /// Reads `arg` back as f32 after running everything recorded so far.
    fn read(c: *Ctx, arg: kernels.Arg) ![]f32 {
        const dst = try c.arena.alloc(f32, @intCast(arg.size / 4));
        c.token += 1;
        try c.rec.readBuffer(c.gpa, c.token, arg.buffer, 0, std.mem.sliceAsBytes(dst));
        try c.flush();
        return dst;
    }
};

/// max_i |got_i - want_i| / max_i |want_i|: error relative to the output scale.
fn relErr(got: []const f32, want: []const f64) f64 {
    std.debug.assert(got.len == want.len);
    var max_err: f64 = 0;
    var scale: f64 = 0;
    for (got, want) |g, w| {
        max_err = @max(max_err, @abs(@as(f64, g) - w));
        scale = @max(scale, @abs(w));
    }
    return max_err / @max(scale, 1e-30);
}

fn checkRmsNorm(c: *Ctx) !Result {
    const rows = 7;
    const dim = 1000;
    const eps = 1e-6;
    const x = try c.random(rows * dim, 3);
    const w = try c.random(dim, 1);
    const y = try c.output(rows * dim);
    try c.launcher().launch(.rmsnorm, P.RmsNorm{ .rows = rows, .dim = dim, .eps = eps }, &.{ try c.upload(x), try c.upload(w), y }, .{ rows, 1, 1 });
    const got = try c.read(y);

    const want = try c.arena.alloc(f64, rows * dim);
    for (0..rows) |r| {
        var ss: f64 = 0;
        for (x[r * dim ..][0..dim]) |v| ss += @as(f64, v) * v;
        const scale = 1 / @sqrt(ss / dim + eps);
        for (0..dim) |i| want[r * dim + i] = x[r * dim + i] * scale * w[i];
    }
    return .{ .name = "rmsnorm", .max_rel_err = relErr(got, want), .tolerance = 1e-6 };
}

fn matmulCase(c: *Ctx, name: []const u8, rows: u32, n: u32, k: u32, split: ?u32) !Result {
    const x = try c.random(rows * k, 1);
    const w = try c.randomBf16(n * k, 1);
    const out = try c.output(rows * n);
    const xa = try c.upload(x);
    const l = c.launcher();
    // Optionally split the weights into two chunks, as large matrices are.
    var col0: u32 = 0;
    for ([_]u32{ split orelse n, n - (split orelse n) }) |cols| {
        if (cols == 0) continue;
        const wa = try c.upload(w.words[col0 * k / 2 ..][0 .. cols * k / 2]);
        try l.launch(.matmul_bf16, P.MatmulBf16{ .rows = rows, .n = cols, .k = k, .col0 = col0, .out_stride = n }, &.{ xa, wa, out }, .{
            std.math.divCeil(u32, cols, kernels.matmul_tile_n) catch unreachable,
            std.math.divCeil(u32, rows, kernels.matmul_tile_t) catch unreachable,
            1,
        });
        col0 += cols;
    }
    const got = try c.read(out);

    const want = try c.arena.alloc(f64, rows * n);
    for (0..rows) |t| for (0..n) |j| {
        var s: f64 = 0;
        for (0..k) |i| s += @as(f64, x[t * k + i]) * w.values[j * k + i];
        want[t * n + j] = s;
    };
    return .{ .name = name, .max_rel_err = relErr(got, want), .tolerance = 1e-5 };
}

fn checkMatmul(c: *Ctx) !Result {
    // 19 tokens and 100 outputs leave partial tiles in both dimensions.
    return matmulCase(c, "matmul_bf16", 19, 100, 1024, null);
}

fn checkMatmulChunked(c: *Ctx) !Result {
    return matmulCase(c, "matmul_bf16 (2 chunks)", 33, 150, 96, 70);
}

fn checkSiluMul(c: *Ctx) !Result {
    const n = 70_001;
    const gate = try c.random(n, 6);
    const up = try c.random(n, 2);
    const ga = try c.upload(gate);
    try c.launcher().launch(.silu_mul, P.Count{ .count = n }, &.{ ga, try c.upload(up) }, kernels.Launcher.elementwise(n));
    const got = try c.read(ga);
    const want = try c.arena.alloc(f64, n);
    for (want, gate, up) |*o, g, u| o.* = @as(f64, g) / (1 + @exp(-@as(f64, g))) * u;
    return .{ .name = "silu_mul", .max_rel_err = relErr(got, want), .tolerance = 1e-6 };
}

fn checkAdd(c: *Ctx) !Result {
    const n = 4099;
    const x = try c.random(n, 1);
    const y = try c.random(n, 1);
    const xa = try c.upload(x);
    try c.launcher().launch(.add, P.Count{ .count = n }, &.{ xa, try c.upload(y) }, kernels.Launcher.elementwise(n));
    const got = try c.read(xa);
    const want = try c.arena.alloc(f64, n);
    for (want, x, y) |*o, a, b| o.* = @as(f32, a + b);
    return .{ .name = "add", .max_rel_err = relErr(got, want), .tolerance = 0 };
}

fn checkEmbed(c: *Ctx) !Result {
    const vocab = 50;
    const dim = 64;
    const split = 30;
    const table = try c.randomBf16(vocab * dim, 1);
    const tokens = [_]u32{ 0, 29, 30, 49, 7, 30, 12 };
    const out = try c.output(tokens.len * dim);
    const ta = try c.upload(&tokens);
    const l = c.launcher();
    for ([_][2]u32{ .{ 0, split }, .{ split, vocab - split } }) |chunk| {
        const words = table.words[chunk[0] * dim / 2 ..][0 .. chunk[1] * dim / 2];
        try l.launch(.embed_bf16, P.EmbedBf16{ .dim = dim, .row0 = chunk[0], .chunk_rows = chunk[1] }, &.{ ta, try c.upload(words), out }, .{ tokens.len, 1, 1 });
    }
    const got = try c.read(out);
    const want = try c.arena.alloc(f64, tokens.len * dim);
    for (tokens, 0..) |tok, t| for (0..dim) |i| {
        want[t * dim + i] = table.values[tok * dim + i];
    };
    return .{ .name = "embed_bf16 (2 chunks)", .max_rel_err = relErr(got, want), .tolerance = 0 };
}
