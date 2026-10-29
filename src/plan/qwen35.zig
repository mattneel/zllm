//! GPU execution plan for the Qwen 3.5 hybrid model.
//!
//! M1 scope: a teacher-forced forward pass over a batch of tokens from an
//! empty state, mirroring ref/qwen35.zig dispatch for dispatch, with the same
//! activation readbacks. Everything is recorded into the command stream; a
//! host executes it.
//!
//! Layout: matrices stay bf16 and are split by rows into chunks that each fit
//! one storage binding (`maxStorageBufferBindingSize`). Matrix multiplies also
//! split tokens into groups whose inputs and outputs fit one binding, which
//! the 248k-row vocabulary projection needs on adapters with 128 MiB bindings.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const graph = @import("../graph/qwen35.zig");
const cmd = @import("../gpu/cmd.zig");
const kernels = @import("../kernels/kernels.zig");
const Arg = kernels.Arg;
const P = kernels.params;

pub const Hparams = graph.Hparams;

pub const Limits = struct {
    /// maxStorageBufferBindingSize
    max_binding: u64,
    /// minStorageBufferOffsetAlignment
    offset_align: u64,
};

/// A bf16 matrix of `n` rows × `k` (row j = output j's weights), split by
/// rows into chunks.
pub const Matrix = struct {
    n: u32,
    k: u32,
    chunks: []const Chunk,

    pub const Chunk = struct { arg: Arg, row0: u32, rows: u32 };
};

const Attention = struct {
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    q_norm: Arg,
    k_norm: Arg,
};

const DeltaNet = struct {
    wqkv: Matrix,
    wgate: Matrix,
    wbeta: Matrix,
    walpha: Matrix,
    wout: Matrix,
    conv: Arg,
    dt_bias: Arg,
    a: Arg,
    norm: Arg,
};

const Layer = struct {
    attn_norm: Arg,
    post_norm: Arg,
    ffn_gate: Matrix,
    ffn_up: Matrix,
    ffn_down: Matrix,
    mixer: union(graph.LayerKind) {
        delta_net: DeltaNet,
        attention: Attention,
    },
};

/// Weights resident on the GPU.
pub const Model = struct {
    hp: Hparams,
    tok_embd: Matrix,
    output: ?Matrix,
    output_norm: Arg,
    layers: []const Layer,

    pub const UploadError = Hparams.Error || Allocator.Error || error{ MissingTensor, BadShape, UnsupportedTensorType, TensorTooLarge };

    /// Records buffer creation and uploads for every weight. `bytes` (the
    /// GGUF file) must stay valid until the recorded batch has executed;
    /// `arena` owns the returned layout.
    pub fn upload(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, file: gguf.File, bytes: []const u8, limits: Limits) UploadError!Model {
        const hp = try Hparams.fromGguf(file);
        var u: Upload = .{ .arena = arena, .gpa = gpa, .rec = rec, .file = file, .bytes = bytes, .limits = limits };
        const e: u64 = hp.n_embd;
        const ff: u64 = hp.n_ff;

        const layers = try arena.alloc(Layer, hp.n_layer);
        for (layers, hp.kinds(), 0..) |*layer, kind, i| {
            layer.* = .{
                .attn_norm = try u.vector("blk.{d}.attn_norm.weight", i, e),
                .post_norm = try u.vector("blk.{d}.post_attention_norm.weight", i, e),
                .ffn_gate = try u.matrix("blk.{d}.ffn_gate.weight", i, e, ff),
                .ffn_up = try u.matrix("blk.{d}.ffn_up.weight", i, e, ff),
                .ffn_down = try u.matrix("blk.{d}.ffn_down.weight", i, ff, e),
                .mixer = switch (kind) {
                    .attention => blk: {
                        const hd: u64 = hp.head_dim;
                        break :blk .{ .attention = .{
                            .wq = try u.matrix("blk.{d}.attn_q.weight", i, e, 2 * hd * hp.n_head),
                            .wk = try u.matrix("blk.{d}.attn_k.weight", i, e, hd * hp.n_head_kv),
                            .wv = try u.matrix("blk.{d}.attn_v.weight", i, e, hd * hp.n_head_kv),
                            .wo = try u.matrix("blk.{d}.attn_output.weight", i, hd * hp.n_head, e),
                            .q_norm = try u.vector("blk.{d}.attn_q_norm.weight", i, hd),
                            .k_norm = try u.vector("blk.{d}.attn_k_norm.weight", i, hd),
                        } };
                    },
                    .delta_net => blk: {
                        const conv_dim: u64 = 2 * hp.gdn_head_dim * hp.gdn_k_heads + hp.gdn_inner;
                        break :blk .{ .delta_net = .{
                            .wqkv = try u.matrix("blk.{d}.attn_qkv.weight", i, e, conv_dim),
                            .wgate = try u.matrix("blk.{d}.attn_gate.weight", i, e, hp.gdn_inner),
                            .wbeta = try u.matrix("blk.{d}.ssm_beta.weight", i, e, hp.gdn_v_heads),
                            .walpha = try u.matrix("blk.{d}.ssm_alpha.weight", i, e, hp.gdn_v_heads),
                            .wout = try u.matrix("blk.{d}.ssm_out.weight", i, hp.gdn_inner, e),
                            .conv = try u.tensor("blk.{d}.ssm_conv1d.weight", i, &.{ hp.conv_kernel, conv_dim }),
                            .dt_bias = try u.vector("blk.{d}.ssm_dt.bias", i, hp.gdn_v_heads),
                            .a = try u.vector("blk.{d}.ssm_a", i, hp.gdn_v_heads),
                            .norm = try u.vector("blk.{d}.ssm_norm.weight", i, hp.gdn_head_dim),
                        } };
                    },
                },
            };
        }

        return .{
            .hp = hp,
            .tok_embd = try u.matrix("token_embd.weight", null, e, hp.vocab),
            .output = if (file.tensor("output.weight") != null) try u.matrix("output.weight", null, e, hp.vocab) else null,
            .output_norm = try u.vector("output_norm.weight", null, e),
            .layers = layers,
        };
    }
};

const Upload = struct {
    arena: Allocator,
    gpa: Allocator,
    rec: *cmd.Recorder,
    file: gguf.File,
    bytes: []const u8,
    limits: Limits,

    fn info(u: *Upload, comptime fmt: []const u8, layer: ?usize, shape: []const u64, t: gguf.TensorType) !*const gguf.TensorInfo {
        var buf: [96]u8 = undefined;
        const name = if (comptime std.mem.indexOf(u8, fmt, "{d}") != null)
            std.fmt.bufPrint(&buf, fmt, .{layer.?}) catch unreachable
        else
            fmt;
        const ti = u.file.tensor(name) orelse return error.MissingTensor;
        if (!std.mem.eql(u64, ti.shape(), shape)) return error.BadShape;
        if (ti.type != t) return error.UnsupportedTensorType;
        return ti;
    }

    fn buffer(u: *Upload, data: []const u8) !Arg {
        if (data.len > u.limits.max_binding) return error.TensorTooLarge;
        const buf = try u.rec.createBuffer(u.gpa, data.len, .{ .storage = true, .copy_dst = true });
        try u.rec.writeBuffer(u.gpa, buf, 0, data);
        return .{ .buffer = buf, .size = data.len };
    }

    /// An f32 tensor of any shape.
    fn tensor(u: *Upload, comptime fmt: []const u8, layer: ?usize, shape: []const u64) !Arg {
        const ti = try u.info(fmt, layer, shape, .f32);
        return u.buffer(u.file.tensorBytes(u.bytes, ti));
    }

    fn vector(u: *Upload, comptime fmt: []const u8, layer: ?usize, n: u64) !Arg {
        return u.tensor(fmt, layer, &.{n});
    }

    /// A bf16 GGML {k, n} weight, split into chunks of whole rows.
    fn matrix(u: *Upload, comptime fmt: []const u8, layer: ?usize, k: u64, n: u64) !Matrix {
        const ti = try u.info(fmt, layer, &.{ k, n }, .bf16);
        const data = u.file.tensorBytes(u.bytes, ti);
        const row_bytes = k * 2;
        const rows_per_chunk = @min(n, u.limits.max_binding / row_bytes);
        if (rows_per_chunk == 0) return error.TensorTooLarge;
        const count = std.math.divCeil(u64, n, rows_per_chunk) catch unreachable;
        const chunks = try u.arena.alloc(Matrix.Chunk, @intCast(count));
        for (chunks, 0..) |*ch, i| {
            const row0 = i * rows_per_chunk;
            const rows = @min(rows_per_chunk, n - row0);
            ch.* = .{
                .arg = try u.buffer(data[@intCast(row0 * row_bytes)..][0..@intCast(rows * row_bytes)]),
                .row0 = @intCast(row0),
                .rows = @intCast(rows),
            };
        }
        return .{ .n = @intCast(n), .k = @intCast(k), .chunks = chunks };
    }
};

/// Activation buffers and per-layer DeltaNet state for a batch of `tokens`.
pub const Activations = struct {
    tokens: u32,
    token_ids: Arg,
    x: Arg,
    h: Arg,
    mix: Arg,
    gate: Arg,
    up: Arg,
    qg: Arg,
    k: Arg,
    v: Arg,
    att: Arg,
    qkv: Arg,
    conv: Arg,
    z: Arg,
    beta: Arg,
    g: Arg,
    o: Arg,
    logits: Arg,
    rope: Arg,
    /// One recurrent state per layer (zero until used); null for attention layers.
    states: []const ?Arg,

    pub const Error = Allocator.Error || error{TooManyTokens};

    /// `rope_table` (see kernels.ropeTable) must stay valid until execution.
    pub fn create(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, hp: Hparams, tokens: u32, rope_table: []const f32) Error!Activations {
        if (tokens == 0 or tokens > kernels.attention_max_tokens) return error.TooManyTokens;
        const n: u64 = tokens;
        const conv_dim: u64 = 2 * hp.gdn_head_dim * hp.gdn_k_heads + hp.gdn_inner;
        const B = struct {
            gpa: Allocator,
            rec: *cmd.Recorder,
            fn f32s(b: @This(), count: u64) Allocator.Error!Arg {
                const size = count * 4;
                const buf = try b.rec.createBuffer(b.gpa, size, .{ .storage = true, .copy_src = true, .copy_dst = true });
                return .{ .buffer = buf, .size = size };
            }
        };
        const b: B = .{ .gpa = gpa, .rec = rec };

        const states = try arena.alloc(?Arg, hp.n_layer);
        for (states, hp.kinds()) |*s, kind| {
            const d: u64 = hp.gdn_head_dim;
            s.* = if (kind == .delta_net) try b.f32s(hp.gdn_v_heads * d * d) else null;
        }
        const rope = try b.f32s(rope_table.len);
        try rec.writeBuffer(gpa, rope.buffer, 0, std.mem.sliceAsBytes(rope_table));

        return .{
            .tokens = tokens,
            .token_ids = try b.f32s(n),
            .x = try b.f32s(n * hp.n_embd),
            .h = try b.f32s(n * hp.n_embd),
            .mix = try b.f32s(n * hp.n_embd),
            .gate = try b.f32s(n * hp.n_ff),
            .up = try b.f32s(n * hp.n_ff),
            .qg = try b.f32s(n * 2 * hp.head_dim * hp.n_head),
            .k = try b.f32s(n * hp.head_dim * hp.n_head_kv),
            .v = try b.f32s(n * hp.head_dim * hp.n_head_kv),
            .att = try b.f32s(n * hp.head_dim * hp.n_head),
            .qkv = try b.f32s(n * conv_dim),
            .conv = try b.f32s(n * conv_dim),
            .z = try b.f32s(n * hp.gdn_inner),
            .beta = try b.f32s(n * hp.gdn_v_heads),
            .g = try b.f32s(n * hp.gdn_v_heads),
            .o = try b.f32s(n * hp.gdn_inner),
            .logits = try b.f32s(n * hp.vocab),
            .rope = rope,
            .states = states,
        };
    }
};

/// Where readbacks land: `layer_out[i]` gets l_out-i, then the final norm and
/// the logits, all row-major [tokens][dim]. Must outlive execution.
pub const Readback = struct {
    layer_out: []const []f32,
    result_norm: []f32,
    logits: []f32,
};

/// Records the whole forward pass for the token ids in `ids`, followed by
/// reads into `out`.
pub fn recordForward(l: kernels.Launcher, limits: Limits, m: *const Model, a: *const Activations, ids: []const u32, out: Readback) !void {
    const hp = m.hp;
    const n = a.tokens;
    std.debug.assert(ids.len == n);
    const e = hp.n_embd;
    try l.rec.writeBuffer(l.gpa, a.token_ids.buffer, 0, std.mem.sliceAsBytes(ids));

    for (m.tok_embd.chunks) |ch| {
        try l.launch(.embed_bf16, P.EmbedBf16{ .dim = e, .row0 = ch.row0, .chunk_rows = ch.rows }, &.{ a.token_ids, ch.arg, a.x }, .{ n, 1, 1 });
    }

    for (m.layers, 0..) |*layer, il| {
        try l.launch(.rmsnorm, P.RmsNorm{ .rows = n, .dim = e, .eps = hp.rms_eps }, &.{ a.x, layer.attn_norm, a.h }, .{ n, 1, 1 });
        switch (layer.mixer) {
            .attention => |*w| try recordAttention(l, limits, hp, w, a),
            .delta_net => |*w| try recordDeltaNet(l, limits, hp, w, a, a.states[il].?),
        }
        try add(l, a.x, a.mix, n * e);

        try l.launch(.rmsnorm, P.RmsNorm{ .rows = n, .dim = e, .eps = hp.rms_eps }, &.{ a.x, layer.post_norm, a.h }, .{ n, 1, 1 });
        try matmul(l, limits, a.gate, a.h, &layer.ffn_gate, n);
        try matmul(l, limits, a.up, a.h, &layer.ffn_up, n);
        try l.launch(.silu_mul, P.Count{ .count = n * hp.n_ff }, &.{ a.gate, a.up }, kernels.Launcher.elementwise(n * hp.n_ff));
        try matmul(l, limits, a.mix, a.gate, &layer.ffn_down, n);
        try add(l, a.x, a.mix, n * e);

        try l.rec.readBuffer(l.gpa, @intCast(il), a.x.buffer, 0, std.mem.sliceAsBytes(out.layer_out[il]));
    }

    try l.launch(.rmsnorm, P.RmsNorm{ .rows = n, .dim = e, .eps = hp.rms_eps }, &.{ a.x, m.output_norm, a.h }, .{ n, 1, 1 });
    try l.rec.readBuffer(l.gpa, hp.n_layer, a.h.buffer, 0, std.mem.sliceAsBytes(out.result_norm));
    try matmul(l, limits, a.logits, a.h, if (m.output) |*o| o else &m.tok_embd, n);
    try l.rec.readBuffer(l.gpa, hp.n_layer + 1, a.logits.buffer, 0, std.mem.sliceAsBytes(out.logits));
}

fn add(l: kernels.Launcher, x: Arg, y: Arg, count: u32) !void {
    try l.launch(.add, P.Count{ .count = count }, &.{ x, y }, kernels.Launcher.elementwise(count));
}

fn recordAttention(l: kernels.Launcher, limits: Limits, hp: Hparams, w: *const Attention, a: *const Activations) !void {
    const n = a.tokens;
    const hd = hp.head_dim;
    try matmul(l, limits, a.qg, a.h, &w.wq, n);
    try matmul(l, limits, a.k, a.h, &w.wk, n);
    try matmul(l, limits, a.v, a.h, &w.wv, n);
    try l.launch(.qk_norm_rope, P.QkNormRope{
        .heads = hp.n_head,
        .head_dim = hd,
        .n_rot = hp.n_rot,
        .row_stride = 2 * hd * hp.n_head,
        .head_stride = 2 * hd,
        .pos0 = 0,
        .eps = hp.rms_eps,
    }, &.{ a.qg, w.q_norm, a.rope }, .{ hp.n_head, n, 1 });
    try l.launch(.qk_norm_rope, P.QkNormRope{
        .heads = hp.n_head_kv,
        .head_dim = hd,
        .n_rot = hp.n_rot,
        .row_stride = hd * hp.n_head_kv,
        .head_stride = hd,
        .pos0 = 0,
        .eps = hp.rms_eps,
    }, &.{ a.k, w.k_norm, a.rope }, .{ hp.n_head_kv, n, 1 });
    try l.launch(.attention, P.Attention{
        .tokens = n,
        .heads = hp.n_head,
        .kv_heads = hp.n_head_kv,
        .head_dim = hd,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd))),
    }, &.{ a.qg, a.k, a.v, a.att }, .{ hp.n_head, n, 1 });
    try matmul(l, limits, a.mix, a.att, &w.wo, n);
}

fn recordDeltaNet(l: kernels.Launcher, limits: Limits, hp: Hparams, w: *const DeltaNet, a: *const Activations, state: Arg) !void {
    const n = a.tokens;
    const d = hp.gdn_head_dim;
    const conv_dim = 2 * d * hp.gdn_k_heads + hp.gdn_inner;
    try matmul(l, limits, a.qkv, a.h, &w.wqkv, n);
    try matmul(l, limits, a.z, a.h, &w.wgate, n);
    try matmul(l, limits, a.beta, a.h, &w.wbeta, n);
    try matmul(l, limits, a.g, a.h, &w.walpha, n);
    try l.launch(.gates, P.Gates{ .count = n * hp.gdn_v_heads, .heads = hp.gdn_v_heads }, &.{ a.beta, a.g, w.dt_bias, w.a }, kernels.Launcher.elementwise(n * hp.gdn_v_heads));
    try l.launch(.conv_silu, P.ConvSilu{ .tokens = n, .channels = conv_dim, .taps = hp.conv_kernel }, &.{ a.qkv, w.conv, a.conv }, kernels.Launcher.elementwise(n * conv_dim));
    try l.launch(.l2norm_heads, P.L2NormHeads{ .heads = 2 * hp.gdn_k_heads, .dim = d, .row_stride = conv_dim, .eps = hp.rms_eps }, &.{a.conv}, .{ 2 * hp.gdn_k_heads, n, 1 });
    try l.launch(.delta_rule, P.DeltaRule{
        .tokens = n,
        .k_heads = hp.gdn_k_heads,
        .v_heads = hp.gdn_v_heads,
        .dim = d,
        .conv_stride = conv_dim,
        .eps = hp.rms_eps,
    }, &.{ a.conv, a.g, a.beta, a.z, w.norm, state, a.o }, .{ hp.gdn_v_heads, 1, 1 });
    try matmul(l, limits, a.mix, a.o, &w.wout, n);
}

/// out[t][j] = x[t] · w[j] for t < rows, over every weight chunk and token
/// group whose bindings fit `limits`.
fn matmul(l: kernels.Launcher, limits: Limits, out: Arg, x: Arg, w: *const Matrix, rows: u32) !void {
    std.debug.assert(w.k % kernels.matmul_tile_k == 0);
    const in_row: u64 = @as(u64, w.k) * 4;
    const out_row: u64 = @as(u64, w.n) * 4;
    // Token groups start at binding offsets aligned for both x and out.
    var step: u64 = 1;
    while ((step * in_row) % limits.offset_align != 0 or (step * out_row) % limits.offset_align != 0) step += 1;
    const fit = limits.max_binding / @max(in_row, out_row);
    const group: u32 = @intCast(@max(step, fit / step * step));
    std.debug.assert(group * @max(in_row, out_row) <= limits.max_binding);

    var t0: u32 = 0;
    while (t0 < rows) : (t0 += group) {
        const tn = @min(group, rows - t0);
        const xa: Arg = .{ .buffer = x.buffer, .offset = x.offset + t0 * in_row, .size = tn * in_row };
        const oa: Arg = .{ .buffer = out.buffer, .offset = out.offset + t0 * out_row, .size = tn * out_row };
        for (w.chunks) |ch| {
            try l.launch(.matmul_bf16, P.MatmulBf16{ .rows = tn, .n = ch.rows, .k = w.k, .col0 = ch.row0, .out_stride = w.n }, &.{ xa, ch.arg, oa }, .{
                std.math.divCeil(u32, ch.rows, kernels.matmul_tile_n) catch unreachable,
                std.math.divCeil(u32, tn, kernels.matmul_tile_t) catch unreachable,
                1,
            });
        }
    }
}
