//! GPU execution plan for the Qwen 3.5 hybrid model.
//!
//! `Model` uploads the weights; `Program` builds everything a step needs
//! once (buffers, persistent state, prebuilt launches) and records steps:
//! m tokens at position pos0, optionally projecting the last rows to logits
//! and sampling the next token on the GPU. Prefill is a step with many
//! tokens, decode a step with one.
//!
//! Layout: matrices stay bf16 and are split by rows into chunks that each fit
//! one storage binding (`maxStorageBufferBindingSize`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const graph = @import("../graph/qwen35.zig");
const cmd = @import("../gpu/cmd.zig");
const kernels = @import("../kernels/kernels.zig");
const Arg = kernels.Arg;
const P = kernels.params;
const sampler = @import("../sampler/sampler.zig");

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

pub const KvFormat = enum(u32) { f32 = 0, f16 = 1 };

pub const Config = struct {
    /// Most tokens one step processes (one prefill chunk).
    max_tokens: u32 = 512,
    /// Positions the KV cache and the RoPE table hold.
    context: u32 = 8192,
    /// Rows of logits one head pass computes: 1 for generation; more lets a
    /// check read every position's logits in fewer passes.
    logit_rows: u32 = 1,
    kv: KvFormat = .f16,
    /// DeltaNet checkpoint slots.
    checkpoints: u32 = 4,
};

/// One entry of the step body: a launch, or the end of a layer (where a
/// check may read the residual stream back).
pub const Item = union(enum) {
    launch: kernels.Launch,
    matmul: Matmul,
    attention: Attend,
    layer_end: u32,
};

/// An attention layer's attention, prebuilt both ways: per query for
/// prefill, split over blocks of cache positions for decoding.
pub const Attend = struct {
    full: kernels.Launch,
    split: kernels.Launch,
    combine: kernels.Launch,

    fn record(a: Attend, gpa: Allocator, rec: *cmd.Recorder, e: kernels.Extent, path: Path) Allocator.Error!void {
        if (path == .decode) {
            try a.split.record(gpa, rec, e);
            try a.combine.record(gpa, rec, e);
        } else {
            try a.full.record(gpa, rec, e);
        }
    }
};

/// Which kernels a step runs: the tiled prefill kernels, or the decode
/// kernels (matrix-vector multiplies, split attention) for steps of at most
/// `kernels.decode_max_rows` tokens. Each is bitwise independent of how a
/// sequence is split into steps, but the two round differently.
pub const Path = enum {
    prefill,
    decode,

    pub fn of(rows: u32) Path {
        return if (rows <= kernels.decode_max_rows) .decode else .prefill;
    }
};

/// One weight chunk's multiply, prebuilt both ways: tiled for prefill,
/// matrix-vector for decode.
pub const Matmul = struct {
    tiled: kernels.Launch,
    vector: kernels.Launch,

    pub fn pick(mm: Matmul, path: Path) kernels.Launch {
        return if (path == .decode) mm.vector else mm.tiled;
    }
};

/// Everything a step needs, built once: activation buffers sized for
/// `max_tokens`, the KV caches, the DeltaNet state and its checkpoints, the
/// sampler's buffers, and every launch with its bind group. A step then
/// writes one uniform (and its token ids) and replays the launches with
/// grids sized to its token count.
///
/// Records keep pointers into the Program (staging for uploads), so it must
/// not move while a recorded batch is pending.
pub const Program = struct {
    hp: Hparams,
    cfg: Config,
    step: Arg,
    token_ids: Arg,
    x: Arg,
    h: Arg,
    lm_in: Arg,
    logits: Arg,
    /// Every DeltaNet layer's recurrent state and conv history, back to back.
    recurrent: Arg,
    checkpoints: []const cmd.Buffer,
    sample_params: Arg,
    bias: Arg,
    prompt_ids: Arg,
    prompt_mask: Arg,
    counts: Arg,
    /// The sampler's state; the drawn token and its stop flag come first.
    sample_state: Arg,

    body: []const Item,
    head: []const Matmul,
    mark: kernels.Launch,
    /// One launch per sampler phase and round.
    sample: std.EnumArray(kernels.SamplePhase, [4]kernels.Launch),
    /// The phases the current request's draws run: the first `sample_len`.
    sample_steps: [kernels.max_sample_steps]kernels.SampleStep = undefined,
    sample_len: usize = 0,

    // Upload staging, alive until the batch that writes it has executed.
    step_value: P.Step = .{ .m = 0, .pos0 = 0, .u = 0 },
    sample_value: P.Sample = undefined,
    token_stage: []u32,
    prompt_stage: []u32,
    bias_stage: [][2]u32,

    pub const BuildError = Allocator.Error || error{ BadConfig, TooLarge };

    /// Records creation of every buffer and bind group. `arena` owns the
    /// result and the parameter staging, and must outlive the Program.
    pub fn build(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *const kernels.Set, m: *const Model, cfg: Config, limits: Limits) BuildError!Program {
        const hp = m.hp;
        if (cfg.max_tokens == 0 or cfg.logit_rows == 0 or cfg.context < cfg.max_tokens or cfg.checkpoints == 0) return error.BadConfig;
        if (hp.conv_kernel > kernels.max_conv_taps or hp.head_dim > kernels.max_head_dim or hp.gdn_head_dim > kernels.max_head_dim) return error.BadConfig;
        if (hp.n_head % hp.n_head_kv != 0 or hp.n_head / hp.n_head_kv > kernels.attn_max_group) return error.BadConfig;
        const e: u64 = hp.n_embd;
        const cap: u64 = cfg.max_tokens;
        const conv_dim: u64 = 2 * hp.gdn_head_dim * hp.gdn_k_heads + hp.gdn_inner;
        const kv_dim: u64 = hp.head_dim * hp.n_head_kv;

        var a: Alloc = .{ .gpa = gpa, .rec = rec, .limits = limits };
        const x = try a.storage(cap * e);
        const h = try a.storage(cap * e);
        const mix = try a.storage(cap * e);
        const gate = try a.storage(cap * hp.n_ff);
        const up = try a.storage(cap * hp.n_ff);
        const qg = try a.storage(cap * 2 * hp.head_dim * hp.n_head);
        const k = try a.storage(cap * kv_dim);
        const v = try a.storage(cap * kv_dim);
        const att = try a.storage(cap * hp.head_dim * hp.n_head);
        const qkv = try a.storage(cap * conv_dim);
        const conv = try a.storage(cap * conv_dim);
        const z = try a.storage(cap * hp.gdn_inner);
        const beta = try a.storage(cap * hp.gdn_v_heads);
        const g = try a.storage(cap * hp.gdn_v_heads);
        const o = try a.storage(cap * hp.gdn_inner);
        const token_ids = try a.storage(cap);
        const lm_in = try a.storage(@as(u64, cfg.logit_rows) * e);
        const logits = try a.storage(@as(u64, cfg.logit_rows) * hp.vocab);

        const rope_table = try kernels.ropeTable(arena, cfg.context, hp.n_rot, hp.rope_base);
        const rope = try a.storage(rope_table.len);
        try rec.writeBuffer(gpa, rope.buffer, 0, std.mem.sliceAsBytes(rope_table));
        const step = try a.buffer(@sizeOf(P.Step), .{ .uniform = true, .copy_dst = true });

        // Recurrent state: per DeltaNet layer, the state then the conv
        // history, each at an aligned offset so it can be bound alone.
        const state_bytes: u64 = 4 * @as(u64, hp.gdn_v_heads) * hp.gdn_head_dim * hp.gdn_head_dim;
        const hist_bytes: u64 = 4 * @as(u64, hp.conv_kernel - 1) * conv_dim;
        const al = limits.offset_align;
        const per_layer = std.mem.alignForward(u64, state_bytes, al) + std.mem.alignForward(u64, hist_bytes, al);
        var n_delta: u64 = 0;
        for (hp.kinds()) |kind| n_delta += @intFromBool(kind == .delta_net);
        const recurrent = try a.buffer(@max(per_layer * n_delta, 4), .{ .storage = true, .copy_src = true, .copy_dst = true });
        const checkpoints = try arena.alloc(cmd.Buffer, cfg.checkpoints);
        for (checkpoints) |*cp| cp.* = (try a.buffer(recurrent.size, .{ .copy_src = true, .copy_dst = true })).buffer;

        const kv_bytes: u64 = if (cfg.kv == .f16) 2 else 4;
        const kv_layer = @as(u64, cfg.context) * kv_dim * kv_bytes;

        const sample_params = try a.buffer(@sizeOf(P.Sample), .{ .uniform = true, .copy_dst = true });
        const bias = try a.storage(2 * kernels.max_bias);
        const prompt_ids = try a.storage(cfg.context);
        const prompt_mask = try a.storage(hp.vocab);
        const counts = try a.storage(hp.vocab);
        const work = try a.storage(hp.vocab);
        // Decode attention's per-block partials: (max, sum, output) per
        // (token, head, block of cache positions).
        const max_blocks: u32 = std.math.divCeil(u32, cfg.context, kernels.attn_block) catch unreachable;
        const attn_part = try a.storage(@as(u64, kernels.decode_max_rows) * hp.n_head * max_blocks * (hp.head_dim + 2));
        const sample_state = try a.storage(kernels.sampleStateWords(hp.vocab));

        var b: kernels.Builder = .{ .gpa = gpa, .arena = arena, .rec = rec, .set = set };
        defer b.deinit();
        var body: std.ArrayList(Item) = .empty;
        const capn: u32 = cfg.max_tokens;
        const norm: P.RmsNorm = .{ .rows = capn, .dim = hp.n_embd, .eps = hp.rms_eps };

        for (m.tok_embd.chunks) |ch| {
            try body.append(arena, .{ .launch = try b.add(.embed_bf16, P.EmbedBf16{ .dim = hp.n_embd, .row0 = ch.row0, .chunk_rows = ch.rows }, &.{ token_ids, ch.arg, x }, .rows_x) });
        }
        var delta_i: u64 = 0;
        for (m.layers, 0..) |*layer, il| {
            try body.append(arena, .{ .launch = try b.add(.rmsnorm, norm, &.{ x, layer.attn_norm, h }, .rows_x) });
            switch (layer.mixer) {
                .attention => |*w| {
                    const kc = try a.buffer(kv_layer, .{ .storage = true });
                    const vc = try a.buffer(kv_layer, .{ .storage = true });
                    try matmul(&b, arena, &body, qg, h, &w.wq, capn);
                    try matmul(&b, arena, &body, k, h, &w.wk, capn);
                    try matmul(&b, arena, &body, v, h, &w.wv, capn);
                    const hd = hp.head_dim;
                    try body.append(arena, .{ .launch = try b.add(.qk_norm_rope, P.QkNormRope{
                        .heads = hp.n_head,
                        .head_dim = hd,
                        .n_rot = hp.n_rot,
                        .row_stride = 2 * hd * hp.n_head,
                        .head_stride = 2 * hd,
                        .eps = hp.rms_eps,
                    }, &.{ qg, w.q_norm, rope, step }, .{ .rows_y = hp.n_head }) });
                    try body.append(arena, .{ .launch = try b.add(.qk_norm_rope, P.QkNormRope{
                        .heads = hp.n_head_kv,
                        .head_dim = hd,
                        .n_rot = hp.n_rot,
                        .row_stride = hd * hp.n_head_kv,
                        .head_stride = hd,
                        .eps = hp.rms_eps,
                    }, &.{ k, w.k_norm, rope, step }, .{ .rows_y = hp.n_head_kv }) });
                    const half: u32 = @backingInt(cfg.kv);
                    try body.append(arena, .{ .launch = try b.add(.kv_append, P.KvAppend{ .width = @intCast(kv_dim), .f16 = half }, &.{ k, v, kc, vc, step }, .{ .elems = @intCast(kv_dim / 2) }) });
                    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
                    const decode: P.AttnDecode = .{ .heads = hp.n_head, .kv_heads = hp.n_head_kv, .head_dim = hd, .f16 = half, .scale = scale, .max_blocks = max_blocks };
                    try body.append(arena, .{ .attention = .{
                        .full = try b.add(.attention, P.Attention{
                            .heads = hp.n_head,
                            .kv_heads = hp.n_head_kv,
                            .head_dim = hd,
                            .f16 = half,
                            .scale = scale,
                        }, &.{ qg, kc, vc, att, step }, .{ .rows_y = hp.n_head }),
                        .split = try b.add(.attn_split, decode, &.{ qg, kc, vc, attn_part, step }, .{ .key_blocks = .{ .x = hp.n_head_kv, .block = kernels.attn_block } }),
                        .combine = try b.add(.attn_combine, decode, &.{ qg, attn_part, att, step }, .{ .rows_y = hp.n_head }),
                    } });
                    try matmul(&b, arena, &body, mix, att, &w.wo, capn);
                },
                .delta_net => |*w| {
                    const base = delta_i * per_layer;
                    delta_i += 1;
                    const state: Arg = .{ .buffer = recurrent.buffer, .offset = base, .size = state_bytes };
                    const hist: Arg = .{ .buffer = recurrent.buffer, .offset = base + std.mem.alignForward(u64, state_bytes, al), .size = hist_bytes };
                    const d = hp.gdn_head_dim;
                    try matmul(&b, arena, &body, qkv, h, &w.wqkv, capn);
                    try matmul(&b, arena, &body, z, h, &w.wgate, capn);
                    try matmul(&b, arena, &body, beta, h, &w.wbeta, capn);
                    try matmul(&b, arena, &body, g, h, &w.walpha, capn);
                    try body.append(arena, .{ .launch = try b.add(.gates, P.Gates{ .count = capn * hp.gdn_v_heads, .heads = hp.gdn_v_heads }, &.{ beta, g, w.dt_bias, w.a }, .{ .elems = hp.gdn_v_heads }) });
                    try body.append(arena, .{ .launch = try b.add(.conv_silu, P.ConvSilu{ .channels = @intCast(conv_dim), .taps = hp.conv_kernel }, &.{ qkv, w.conv, hist, conv, step }, .{ .fixed = kernels.Launcher.elementwise(conv_dim) }) });
                    try body.append(arena, .{ .launch = try b.add(.l2norm_heads, P.L2NormHeads{ .heads = 2 * hp.gdn_k_heads, .dim = d, .row_stride = @intCast(conv_dim), .eps = hp.rms_eps }, &.{conv}, .{ .rows_y = 2 * hp.gdn_k_heads }) });
                    try body.append(arena, .{ .launch = try b.add(.delta_rule, P.DeltaRule{
                        .k_heads = hp.gdn_k_heads,
                        .v_heads = hp.gdn_v_heads,
                        .dim = d,
                        .conv_stride = @intCast(conv_dim),
                        .eps = hp.rms_eps,
                    }, &.{ conv, g, beta, state, o, step }, .{ .fixed = .{ hp.gdn_v_heads, std.math.divCeil(u32, d, kernels.delta_cols) catch unreachable, 1 } }) });
                    try body.append(arena, .{ .launch = try b.add(.gated_norm, P.GatedNorm{ .v_heads = hp.gdn_v_heads, .dim = d, .eps = hp.rms_eps }, &.{ o, z, w.norm }, .{ .rows_y = hp.gdn_v_heads }) });
                    try matmul(&b, arena, &body, mix, o, &w.wout, capn);
                },
            }
            try body.append(arena, .{ .launch = try b.add(.add, P.Count{ .count = capn * hp.n_embd }, &.{ x, mix }, .{ .elems = hp.n_embd }) });
            try body.append(arena, .{ .launch = try b.add(.rmsnorm, norm, &.{ x, layer.post_norm, h }, .rows_x) });
            try matmul(&b, arena, &body, gate, h, &layer.ffn_gate, capn);
            try matmul(&b, arena, &body, up, h, &layer.ffn_up, capn);
            try body.append(arena, .{ .launch = try b.add(.silu_mul, P.Count{ .count = capn * hp.n_ff }, &.{ gate, up }, .{ .elems = hp.n_ff }) });
            try matmul(&b, arena, &body, mix, gate, &layer.ffn_down, capn);
            try body.append(arena, .{ .launch = try b.add(.add, P.Count{ .count = capn * hp.n_embd }, &.{ x, mix }, .{ .elems = hp.n_embd }) });
            try body.append(arena, .{ .layer_end = @intCast(il) });
        }
        try body.append(arena, .{ .launch = try b.add(.rmsnorm, norm, &.{ x, m.output_norm, h }, .rows_x) });

        var head: std.ArrayList(Item) = .empty;
        try matmul(&b, arena, &head, logits, lm_in, if (m.output) |*out| out else &m.tok_embd, cfg.logit_rows);
        const head_launches = try arena.alloc(Matmul, head.items.len);
        for (head.items, head_launches) |item, *l| l.* = item.matmul;

        const mark = try b.addWith(.mark_tokens, sample_params, &.{ prompt_ids, prompt_mask }, .{ .elems = 1 });
        var sample: std.EnumArray(kernels.SamplePhase, [4]kernels.Launch) = undefined;
        for (std.enums.values(kernels.SamplePhase)) |phase| {
            const groups: u32 = if (phase.wide()) kernels.sampleGroups(hp.vocab) else 1;
            const rounds: u32 = if (phase == .hist or phase == .digit) 4 else 1;
            for (sample.getPtr(phase), 0..) |*l, r| {
                // Phases with one round reuse round 0's launch.
                if (r >= rounds) {
                    l.* = sample.get(phase)[0];
                    continue;
                }
                l.* = try b.add(.sample, P.SamplePhase{ .phase = @backingInt(phase), .round = @intCast(r) }, &.{
                    logits, bias, prompt_mask, counts, work, sample_state, token_ids, step, sample_params,
                }, .{ .fixed = .{ groups, 1, 1 } });
            }
        }
        try b.finish();

        return .{
            .hp = hp,
            .cfg = cfg,
            .step = step,
            .token_ids = token_ids,
            .x = x,
            .h = h,
            .lm_in = lm_in,
            .logits = logits,
            .recurrent = recurrent,
            .checkpoints = checkpoints,
            .sample_params = sample_params,
            .bias = bias,
            .prompt_ids = prompt_ids,
            .prompt_mask = prompt_mask,
            .counts = counts,
            .sample_state = sample_state,
            .body = body.items,
            .head = head_launches,
            .mark = mark,
            .sample = sample,
            .token_stage = try arena.alloc(u32, cfg.max_tokens),
            .prompt_stage = try arena.alloc(u32, cfg.context),
            .bias_stage = try arena.alloc([2]u32, kernels.max_bias),
        };
    }

    /// Zeroes the DeltaNet state: the next step starts a sequence at
    /// position 0. (Cache rows are only read at positions a step has written.)
    pub fn recordReset(p: *const Program, gpa: Allocator, rec: *cmd.Recorder) Allocator.Error!void {
        try rec.clearBuffer(gpa, p.recurrent.buffer, 0, p.recurrent.size);
    }

    pub fn recordSave(p: *const Program, gpa: Allocator, rec: *cmd.Recorder, slot: u32) Allocator.Error!void {
        try rec.copyBuffer(gpa, p.recurrent.buffer, 0, p.checkpoints[slot], 0, p.recurrent.size);
    }

    pub fn recordRestore(p: *const Program, gpa: Allocator, rec: *cmd.Recorder, slot: u32) Allocator.Error!void {
        try rec.copyBuffer(gpa, p.checkpoints[slot], 0, p.recurrent.buffer, 0, p.recurrent.size);
    }

    /// Sets up the sampler for a request: its parameters, the prompt's token
    /// set for the repetition penalty, and zero output counts.
    pub fn recordRequest(p: *Program, gpa: Allocator, rec: *cmd.Recorder, params: sampler.Params, prompt: []const u32, stops: []const u32) Allocator.Error!void {
        std.debug.assert(prompt.len <= p.cfg.context);
        p.sample_value = params.gpu(p.hp.vocab, @intCast(prompt.len), stops);
        p.sample_len = kernels.sampleSteps(p.sample_value, &p.sample_steps).len;
        try rec.writeBuffer(gpa, p.sample_params.buffer, 0, std.mem.asBytes(&p.sample_value));
        const biases = params.biasWords(p.bias_stage);
        if (biases.len > 0) try rec.writeBuffer(gpa, p.bias.buffer, 0, std.mem.sliceAsBytes(biases));
        @memcpy(p.prompt_stage[0..prompt.len], prompt);
        if (prompt.len > 0) try rec.writeBuffer(gpa, p.prompt_ids.buffer, 0, std.mem.sliceAsBytes(p.prompt_stage[0..prompt.len]));
        try rec.clearBuffer(gpa, p.counts.buffer, 0, p.counts.size);
        try rec.clearBuffer(gpa, p.prompt_mask.buffer, 0, p.prompt_mask.size);
        if (prompt.len > 0) try p.mark.record(gpa, rec, .{ .rows = @intCast(prompt.len) });
    }

    pub const Step = struct {
        /// Position of the step's first token.
        pos0: u32,
        /// Token ids to process; null: the one token the previous step sampled.
        tokens: ?[]const u32,
        /// Project this many of the step's last rows to logits.
        logit_rows: u32 = 0,
        /// Sample from the last row with this draw (needs logit_rows == 1),
        /// reading back (token, is_stop) into `sampled`.
        draw: ?f32 = null,
        sampled: ?*[2]u32 = null,
        /// Activation readbacks for checks.
        readback: ?Readback = null,
        /// The kernels to use; by default the decode kernels for steps of at
        /// most `kernels.decode_max_rows` tokens, else the prefill kernels.
        path: ?Path = null,
    };

    /// Where a step's readbacks land, row-major [rows][dim]: `layer_out[i]`
    /// gets l_out-i and `result_norm` the final norm (m rows each), `logits`
    /// the step's logit rows. Each is optional. Must outlive execution.
    pub const Readback = struct {
        layer_out: []const []f32 = &.{},
        result_norm: ?[]f32 = null,
        logits: ?[]f32 = null,
    };

    pub fn recordStep(p: *Program, gpa: Allocator, rec: *cmd.Recorder, s: Step) Allocator.Error!void {
        const m: u32 = if (s.tokens) |t| @intCast(t.len) else 1;
        const path = s.path orelse Path.of(m);
        std.debug.assert(path == .prefill or m <= kernels.decode_max_rows);
        const e: u64 = p.hp.n_embd;
        std.debug.assert(m > 0 and m <= p.cfg.max_tokens and s.pos0 + m <= p.cfg.context);
        std.debug.assert(s.logit_rows <= m and (s.draw == null or s.logit_rows == 1));

        p.step_value = .{ .m = m, .pos0 = s.pos0, .u = s.draw orelse 0 };
        try rec.writeBuffer(gpa, p.step.buffer, 0, std.mem.asBytes(&p.step_value));
        if (s.tokens) |t| {
            @memcpy(p.token_stage[0..t.len], t);
            try rec.writeBuffer(gpa, p.token_ids.buffer, 0, std.mem.sliceAsBytes(p.token_stage[0..t.len]));
        }

        const ext: kernels.Extent = .{ .rows = m, .keys = s.pos0 + m };
        for (p.body) |item| switch (item) {
            .launch => |l| try l.record(gpa, rec, ext),
            .matmul => |mm| try mm.pick(path).record(gpa, rec, ext),
            .attention => |at| try at.record(gpa, rec, ext, path),
            .layer_end => |il| if (s.readback) |r| if (r.layer_out.len > 0) {
                try rec.readBuffer(gpa, read_token, p.x.buffer, 0, std.mem.sliceAsBytes(r.layer_out[il][0 .. m * e]));
            },
        };
        if (s.readback) |r| if (r.result_norm) |dst| try rec.readBuffer(gpa, read_token, p.h.buffer, 0, std.mem.sliceAsBytes(dst[0 .. m * e]));

        // Logit rows in passes of at most cfg.logit_rows.
        const vocab: u64 = p.hp.vocab;
        var r0 = m - s.logit_rows;
        while (r0 < m) {
            const n = @min(p.cfg.logit_rows, m - r0);
            try rec.copyBuffer(gpa, p.h.buffer, r0 * e * 4, p.lm_in.buffer, 0, n * e * 4);
            for (p.head) |mm| try mm.pick(if (path == .decode and n <= kernels.decode_max_rows) .decode else .prefill).record(gpa, rec, .{ .rows = n });
            if (s.readback) |r| if (r.logits) |dst| {
                const at = (r0 - (m - s.logit_rows)) * vocab;
                try rec.readBuffer(gpa, read_token, p.logits.buffer, 0, std.mem.sliceAsBytes(dst[at..][0 .. n * vocab]));
            };
            r0 += n;
        }

        if (s.draw != null) {
            std.debug.assert(p.sample_len > 0);
            for (p.sample_steps[0..p.sample_len]) |ss| try p.sample.get(ss.phase)[ss.round].record(gpa, rec, .{ .rows = 1 });
            try rec.readBuffer(gpa, read_token, p.sample_state.buffer, 0, std.mem.asBytes(s.sampled.?));
        }
    }
};

/// Completion token of every read a Program records.
pub const read_token = 0xffff_fffe;

/// Records buffer creation, checking each fits one binding.
const Alloc = struct {
    gpa: Allocator,
    rec: *cmd.Recorder,
    limits: Limits,

    fn buffer(a: *Alloc, size: u64, usage: cmd.Usage) Program.BuildError!Arg {
        if (usage.storage and size > a.limits.max_binding) return error.TooLarge;
        return .{ .buffer = try a.rec.createBuffer(a.gpa, size, usage), .size = size };
    }

    /// `count` f32 or u32 values.
    fn storage(a: *Alloc, count: u64) Program.BuildError!Arg {
        return a.buffer(count * 4, .{ .storage = true, .copy_src = true, .copy_dst = true });
    }
};

/// out[t][j] = x[t] · w[j] for the rows of a step, over every weight chunk.
fn matmul(b: *kernels.Builder, arena: Allocator, list: *std.ArrayList(Item), out: Arg, x: Arg, w: *const Matrix, cap: u32) Allocator.Error!void {
    std.debug.assert(w.k % kernels.matmul_tile_k == 0);
    for (w.chunks) |ch| {
        const tiled = try b.add(.matmul_bf16, P.MatmulBf16{ .rows = cap, .n = ch.rows, .k = w.k, .col0 = ch.row0, .out_stride = w.n }, &.{ x, ch.arg, out }, .{ .tiles_y = .{
            .x = std.math.divCeil(u32, ch.rows, kernels.matmul_tile_n) catch unreachable,
            .tile = kernels.matmul_tile_t,
        } });
        const vector = try b.add(.matvec_bf16, P.MatvecBf16{ .n = ch.rows, .k = w.k, .col0 = ch.row0, .out_stride = w.n }, &.{ x, ch.arg, out, ch.arg }, .{
            .rows_y = std.math.divCeil(u32, ch.rows, kernels.matvec_rows) catch unreachable,
        });
        try list.append(arena, .{ .matmul = .{ .tiled = tiled, .vector = vector } });
    }
}
