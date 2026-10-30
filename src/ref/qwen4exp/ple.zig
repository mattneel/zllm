//! Qwen4Exp per-layer embeddings (PLE): the rows of the hashed n-gram table
//! `per_layer_token_embd` a token reads. Each token hashes itself and its
//! two predecessors (newest first; a missing one, or anything before an
//! end-of-sequence token, counts as EOS) into 16 heads, 8 per n-gram order,
//! each a row of its own range of the table. The hash follows
//! agentionai/llama.cpp src/models/qwen4exp.cpp at 9aa51ce4; lifted from
//! gyro-dx, whose rows matched an independent C oracle of that formula.

const std = @import("std");
const gguf = @import("../../gguf/gguf.zig");

/// Heads: (n-gram size − 1) × heads per n-gram, as the released files have.
pub const heads = 16;

pub const Error = error{ BadMetadata, MissingMetadata, MissingTensor, InvalidToken };

pub const Config = struct {
    eos_token: u32,
    vocab: u32,
    multipliers: [3]u64,
    head_offsets: [heads]u64,
    head_vocab_sizes: [heads]u64,
    table_rows: u64,

    /// The 16 table rows of `token` after `previous` (newest first).
    pub fn rows(c: Config, token: u32, previous: [2]?u32) error{InvalidToken}![heads]u64 {
        if (token >= c.vocab) return error.InvalidToken;
        var context = [_]u64{ token, c.eos_token, c.eos_token };
        for (previous, 0..) |p, i| {
            const t = p orelse break;
            if (t >= c.vocab) return error.InvalidToken;
            if (t == c.eos_token) break;
            context[i + 1] = t;
        }
        var out: [heads]u64 = undefined;
        var mixed = context[0] *% c.multipliers[0];
        for (1..3) |gram| {
            mixed ^= context[gram] *% c.multipliers[gram];
            for (0..8) |within| {
                const head = (gram - 1) * 8 + within;
                out[head] = mixed % c.head_vocab_sizes[head] + c.head_offsets[head];
            }
        }
        return out;
    }

    pub fn fromGguf(f: gguf.File) Error!Config {
        const a = "qwen4exp.";
        if (try unsigned(f, a ++ "ple.ngram_size") != 3 or try unsigned(f, a ++ "ple.heads_per_ngram") != 8) return error.BadMetadata;
        const table = f.tensor("per_layer_token_embd.weight") orelse return error.MissingTensor;
        const embd = f.tensor("token_embd.weight") orelse return error.MissingTensor;
        if (table.n_dims != 2 or embd.n_dims != 2) return error.BadMetadata;
        var c: Config = .{
            .eos_token = std.math.cast(u32, try unsigned(f, a ++ "ple.eos_token_id")) orelse return error.BadMetadata,
            .vocab = std.math.cast(u32, embd.dims[1]) orelse return error.BadMetadata,
            .multipliers = undefined,
            .head_offsets = undefined,
            .head_vocab_sizes = undefined,
            .table_rows = table.dims[1],
        };
        if (c.eos_token >= c.vocab) return error.BadMetadata;
        try array(f, a ++ "ple.layer_multipliers", &c.multipliers);
        try array(f, a ++ "ple.head_offsets", &c.head_offsets);
        try array(f, a ++ "ple.head_vocab_sizes", &c.head_vocab_sizes);
        for (c.head_offsets, c.head_vocab_sizes) |off, n| {
            if (n == 0 or off >= c.table_rows or n > c.table_rows - off) return error.BadMetadata;
        }
        return c;
    }
};

/// The token history PLE hashes: the last two tokens, newest first.
pub const History = struct {
    previous: [2]?u32 = .{ null, null },

    /// The rows of `token`, which then joins the history.
    pub fn accept(h: *History, c: Config, token: u32) error{InvalidToken}![heads]u64 {
        const r = try c.rows(token, h.previous);
        // Not `.{ token, h.previous[0] }`: that writes previous[0] before
        // reading it.
        const newest = h.previous[0];
        h.previous = .{ token, newest };
        return r;
    }
};

fn unsigned(f: gguf.File, key: []const u8) Error!u64 {
    return switch (f.get(key) orelse return error.MissingMetadata) {
        .u32 => |x| x,
        .u64 => |x| x,
        .i32 => |x| std.math.cast(u64, x) orelse error.BadMetadata,
        else => error.BadMetadata,
    };
}

fn array(f: gguf.File, key: []const u8, out: []u64) Error!void {
    const v = f.get(key) orelse return error.MissingMetadata;
    if (v != .array or v.array.elem_type != .u64 or v.array.len != out.len) return error.BadMetadata;
    for (out, 0..) |*o, i| o.* = v.array.get(u64, i);
}

/// The released files' configuration.
const released: Config = .{
    .eos_token = 248044,
    .vocab = 248320,
    .multipliers = .{ 23703573157769, 20109073645365, 8052911324071 },
    .head_offsets = .{ 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275 },
    .head_vocab_sizes = .{ 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 },
    .table_rows = 320001536,
};

// Rows from gyro-dx's tests/ple_vectors_generate.py, an isolated C oracle of
// the upstream formula: EOS at either position, missing predecessors, the
// vocabulary's edges, and 64-bit wrap-around.
test "rows match the C oracle" {
    const Case = struct { token: u32, previous: [2]?u32, want: [heads]u64 };
    const cases = [_]Case{
        .{ .token = 1, .previous = .{ null, null }, .want = .{ 16121432, 28938500, 59087997, 73487090, 81148277, 104500129, 120276032, 149373875, 176283436, 184305849, 216528839, 231080079, 257961536, 266068568, 289043455, 305959965 } },
        .{ .token = 12345, .previous = .{ 456, 789 }, .want = .{ 5790556, 34151681, 58558375, 68980953, 86721192, 112682904, 131670637, 143738651, 175870771, 191509697, 203365459, 224597274, 252992302, 261441981, 297604319, 318506071 } },
        .{ .token = 248044, .previous = .{ 456, 789 }, .want = .{ 9802217, 39863005, 59309929, 67482077, 87651392, 115316748, 137698027, 155855185, 169513123, 186401038, 201459525, 224837487, 254405449, 265032678, 282143953, 309465545 } },
        .{ .token = 123, .previous = .{ 248044, 555 }, .want = .{ 19991442, 35773710, 47404574, 63876269, 93312783, 117256139, 133919058, 144199064, 168416363, 188577563, 218298233, 240000618, 242956453, 267138661, 295472052, 319181751 } },
        .{ .token = 123, .previous = .{ null, 555 }, .want = .{ 19991442, 35773710, 47404574, 63876269, 93312783, 117256139, 133919058, 144199064, 168416363, 188577563, 218298233, 240000618, 242956453, 267138661, 295472052, 319181751 } },
        .{ .token = 123, .previous = .{ 555, 248044 }, .want = .{ 11851980, 26389904, 53668609, 71869655, 84623506, 115543611, 131925624, 153771940, 164337395, 185474937, 205277463, 238303451, 252628876, 267312075, 285619420, 317751962 } },
        .{ .token = 248319, .previous = .{ 248318, 248317 }, .want = .{ 15128987, 30260135, 58954572, 68391029, 89082003, 109553187, 120485668, 142150431, 171274154, 191656020, 209760836, 233093504, 245346139, 277795283, 295322020, 303282311 } },
        .{ .token = 0, .previous = .{ 0, 0 }, .want = .{ 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275 } },
    };
    for (cases) |c| try std.testing.expectEqual(c.want, try released.rows(c.token, c.previous));
    var wrap = released;
    wrap.multipliers = .{ std.math.maxInt(u64), std.math.maxInt(u64) - 16, std.math.maxInt(u64) - 32 };
    try std.testing.expectEqual([heads]u64{ 4431907, 24431910, 44431933, 64431966, 84432013, 104432072, 124432135, 144432204, 174518348, 184365542, 212640327, 233035851, 252823303, 275931125, 291038312, 312107704 }, try wrap.rows(248319, .{ 248318, 248317 }));
    try std.testing.expectError(error.InvalidToken, released.rows(248320, .{ null, null }));
}

test "history: newest first, EOS kept, an invalid token not taken" {
    var h: History = .{};
    _ = try h.accept(released, 789);
    _ = try h.accept(released, 456);
    try std.testing.expectEqual(try released.rows(248044, .{ 456, 789 }), try h.accept(released, 248044));
    try std.testing.expectError(error.InvalidToken, h.accept(released, 248320));
    // After EOS the context restarts.
    try std.testing.expectEqual(try released.rows(123, .{ null, null }), try h.accept(released, 123));
}
