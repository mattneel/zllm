//! The Qwen4Exp (Qwen3.8-Flash-Next) step program: its weights on the GPU
//! and every launch of a step, built once (src/ref/qwen4exp/engine.zig is
//! the graph it follows). This first version runs the decode kernels only,
//! in steps of at most kernels.decode_max_rows tokens, and caps the context
//! at the sparse attention's budget: up to 2,048 + 3 cells the indexer
//! selects every cell, so dense attention is the same computation.
//!
//! The per-layer embedding (PLE) table stays in the file: each step hashes
//! its tokens' histories (ple.History) and records `write_file`s of their
//! 16 rows each, which the host serves from the file (a mapping natively,
//! the picked File in the browser). The engine therefore passes every
//! step's tokens: a sampled token is read back before the next step.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const graph = @import("../graph/qwen4exp.zig");
const cmd = @import("../gpu/cmd.zig");
const kernels = @import("../kernels/kernels.zig");
const gyro = @import("../ref/gyro/gyro.zig");
const ple = @import("../ref/qwen4exp/ple.zig");
const sampler = @import("../sampler/sampler.zig");
const common = @import("common.zig");
const Arg = kernels.Arg;
const P = kernels.params;
const Matrix = common.Matrix;
const Upload = common.Upload;
const Alloc = common.Alloc;
const Mm = common.Mm;
const weightKernel = common.weightKernel;

pub const Hparams = graph.Hparams;

/// Contexts the program allows: the sparse attention's cell budget, below
/// which it is dense (the indexer is not built yet).
pub const max_context = 2048;
/// PLE rows a token reads.
pub const ple_heads = ple.heads;

const Hc = struct {
    norm: Arg,
    down: Matrix,
    up: Matrix,
    inject: ?Matrix,
};

const Linear = struct {
    /// qkv | z | β | α, stacked by rows.
    proj: Matrix,
    out: Matrix,
    conv: Arg,
    dt_bias: Arg,
    a: Arg,
    norm: Arg,
};

const Full = struct {
    /// Q (with the output gates) | K | V.
    qkv: Matrix,
    wo: Matrix,
    q_norm: Arg,
    k_norm: Arg,
};

const Ple = struct {
    key: Matrix,
    value: Matrix,
    norm_key: Arg,
    norm_query: Arg,
    norm_conv: Arg,
    conv: Arg,
};

const Layer = struct {
    hc_attn: Hc,
    hc_ffn: Hc,
    mixer: union(enum) { linear: Linear, full: Full },
    ple: ?Ple,
    router: Arg,
    shared_gate: Arg,
    /// The routed experts, Gyro-coded, [n_expert × rows × cols] each.
    gate_exps: Arg,
    up_exps: Arg,
    down_exps: Arg,
    /// Shared expert: gate | up, then down.
    sh_gu: Matrix,
    sh_down: Matrix,
};

pub const Model = struct {
    hp: Hparams,
    tok_embd: Matrix,
    output: Matrix,
    hc_head: Hc,
    layers: []const Layer,
    /// Sign vectors of the expert input rotations: widths n_embd and n_ff_exp.
    signs_embd: Arg,
    signs_ff: Arg,
    codebook: Arg,
    ple: PleTable,

    pub const UploadError = graph.Error || ple.Error || common.UploadError;

    /// Calls f(context, w) for every Q8_0 multiply (for tuning and scratch
    /// sizing).
    pub fn eachMatrix(m: *const Model, context: anytype, comptime f: fn (@TypeOf(context), *const Matrix) void) void {
        f(context, &m.output);
        for ([_]*const Hc{&m.hc_head}) |h| eachHc(h, context, f);
        for (m.layers) |*l| {
            eachHc(&l.hc_attn, context, f);
            eachHc(&l.hc_ffn, context, f);
            switch (l.mixer) {
                .linear => |*w| {
                    f(context, &w.proj);
                    f(context, &w.out);
                },
                .full => |*w| {
                    f(context, &w.qkv);
                    f(context, &w.wo);
                },
            }
            if (l.ple) |*p| {
                f(context, &p.key);
                f(context, &p.value);
            }
            f(context, &l.sh_gu);
            f(context, &l.sh_down);
        }
    }

    fn eachHc(h: *const Hc, context: anytype, comptime f: fn (@TypeOf(context), *const Matrix) void) void {
        f(context, &h.down);
        f(context, &h.up);
        if (h.inject) |*i| f(context, i);
    }

    /// Records the upload of every weight but the PLE table. Only TQ2_T
    /// experts are supported on the GPU so far.
    pub fn upload(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *const kernels.Set, file: gguf.File, limits: common.Limits) UploadError!Model {
        const hp = try Hparams.fromGguf(file);
        var u: Upload = .{ .arena = arena, .gpa = gpa, .rec = rec, .set = set, .file = file, .limits = limits };
        const e: u64 = hp.n_embd;
        const hd: u64 = hp.hcDim();
        const layers = try arena.alloc(Layer, hp.n_layer);
        for (layers, 0..) |*l, i| {
            const fe: u64 = hp.n_ff_exp;
            const ne: u64 = hp.n_expert;
            l.* = .{
                .hc_attn = try uploadHc(&u, hp, i, "attn"),
                .hc_ffn = try uploadHc(&u, hp, i, "ffn"),
                .mixer = if (hp.recurrent(i)) .{ .linear = .{
                    .proj = try u.stacked(i, e, &.{
                        .{ "blk.{d}.attn_qkv.weight", hp.convChannels() },
                        .{ "blk.{d}.attn_gate.weight", hp.ssm_inner },
                        .{ "blk.{d}.ssm_beta.weight", hp.ssm_v_heads },
                        .{ "blk.{d}.ssm_alpha.weight", hp.ssm_v_heads },
                    }),
                    .out = try u.matrix("blk.{d}.ssm_out.weight", i, hp.ssm_inner, e),
                    .conv = try u.tensor("blk.{d}.ssm_conv1d.weight", i, &.{ hp.ssm_conv, hp.convChannels() }),
                    .dt_bias = try u.vector("blk.{d}.ssm_dt.bias", i, hp.ssm_v_heads),
                    .a = try u.vector("blk.{d}.ssm_a", i, hp.ssm_v_heads),
                    .norm = try u.vector("blk.{d}.ssm_norm.weight", i, hp.ssm_state),
                } } else .{ .full = .{
                    .qkv = try u.stacked(i, e, &.{
                        .{ "blk.{d}.attn_q.weight", 2 * @as(u64, hp.head_dim) * hp.n_head },
                        .{ "blk.{d}.attn_k.weight", @as(u64, hp.head_dim) * hp.n_head_kv },
                        .{ "blk.{d}.attn_v.weight", @as(u64, hp.head_dim) * hp.n_head_kv },
                    }),
                    .wo = try u.matrix("blk.{d}.attn_output.weight", i, @as(u64, hp.head_dim) * hp.n_head, e),
                    .q_norm = try u.vector("blk.{d}.attn_q_norm.weight", i, hp.head_dim),
                    .k_norm = try u.vector("blk.{d}.attn_k_norm.weight", i, hp.head_dim),
                } },
                .ple = if (i == hp.ple_layer) .{
                    .key = try u.matrix("blk.{d}.ple_key.weight", i, e, hd),
                    .value = try u.matrix("blk.{d}.ple_value.weight", i, e, e),
                    .norm_key = try u.vector("blk.{d}.ple_norm_key.weight", i, hd),
                    .norm_query = try u.vector("blk.{d}.ple_norm_query.weight", i, hd),
                    .norm_conv = try u.vector("blk.{d}.ple_norm_conv.weight", i, hd),
                    .conv = try u.tensor("blk.{d}.ple_conv1d.weight", i, &.{ hp.ple_conv, hd }),
                } else null,
                .router = try u.tensor("blk.{d}.ffn_gate_inp.weight", i, &.{ e, ne }),
                .shared_gate = try u.vector("blk.{d}.ffn_gate_inp_shexp.weight", i, e),
                .gate_exps = try u.raw("blk.{d}.ffn_gate_exps.weight", i, &.{ e, fe, ne }, .tq2_t),
                .up_exps = try u.raw("blk.{d}.ffn_up_exps.weight", i, &.{ e, fe, ne }, .tq2_t),
                .down_exps = try u.raw("blk.{d}.ffn_down_exps.weight", i, &.{ fe, e, ne }, .tq2_t),
                .sh_gu = try u.stacked(i, e, &.{ .{ "blk.{d}.ffn_gate_shexp.weight", hp.n_ff_shexp }, .{ "blk.{d}.ffn_up_shexp.weight", hp.n_ff_shexp } }),
                .sh_down = try u.matrix("blk.{d}.ffn_down_shexp.weight", i, hp.n_ff_shexp, e),
            };
        }
        const signs = try rotationSigns(arena, file, hp);
        const table = try PleTable.find(file, hp);
        return .{
            .hp = hp,
            .tok_embd = try u.matrix("token_embd.weight", null, e, hp.vocab),
            .output = try u.matrix("output.weight", null, e, hp.vocab),
            .hc_head = .{
                .norm = try u.vector("output_hc_norm.weight", null, hd),
                .down = try u.matrix("output_hc_down.weight", null, hd, hp.hc_rank),
                .up = try u.matrix("output_hc_up.weight", null, hp.hc_rank, hd),
                .inject = null,
            },
            .layers = layers,
            .signs_embd = try upload32(gpa, rec, signs.embd),
            .signs_ff = try upload32(gpa, rec, signs.ff),
            .codebook = try uploadBytes(gpa, rec, gyro.codebook_bytes),
            .ple = table,
        };
    }
};

/// Where each token's PLE rows lie in the file. A row is read with the 2
/// bytes after it, `stride` in all, as GPU writes are whole words; another
/// tensor's data must follow the table so that stays inside the file.
const PleTable = struct {
    config: ple.Config,
    /// Absolute file offset of row 0.
    at: u64,
    row_bytes: u64,
    stride: u32,

    fn find(file: gguf.File, hp: Hparams) Model.UploadError!PleTable {
        const t = file.tensor("per_layer_token_embd.weight") orelse return error.MissingTensor;
        if (t.type != .q8_0 or t.n_dims != 2 or t.dims[0] != hp.ple_head_dim) return error.Unsupported;
        const row_bytes = t.byteSize() / t.dims[1];
        const stride = std.mem.alignForward(u64, row_bytes, 4);
        const end = t.offset + t.byteSize();
        const followed = for (file.tensors) |other| {
            if (other.offset >= end + (stride - row_bytes)) break true;
        } else false;
        if (!followed) return error.Unsupported;
        return .{ .config = try ple.Config.fromGguf(file), .at = file.data_offset + t.offset, .row_bytes = row_bytes, .stride = @intCast(stride) };
    }
};

fn uploadHc(u: *Upload, hp: Hparams, i: usize, comptime which: []const u8) Model.UploadError!Hc {
    const hd: u64 = hp.hcDim();
    return .{
        .norm = try u.vector("blk.{d}.hc_" ++ which ++ "_norm.weight", i, hd),
        .down = try u.matrix("blk.{d}.hc_" ++ which ++ "_down.weight", i, hd, hp.hc_rank),
        .up = try u.matrix("blk.{d}.hc_" ++ which ++ "_up.weight", i, hp.hc_rank, hd),
        .inject = try u.matrix("blk.{d}.hc_" ++ which ++ "_inject.weight", i, hd, hp.hc),
    };
}

/// The prism.hadamard sign vectors of widths n_embd and n_ff_exp. Every
/// expert tensor must be listed as rotated (the program always rotates), and
/// nothing else.
fn rotationSigns(arena: Allocator, file: gguf.File, hp: Hparams) Model.UploadError!struct { embd: []f32, ff: []f32 } {
    const names = file.get("prism.hadamard.weight_names") orelse return error.MissingMetadata;
    if (names != .array or names.array.elem_type != .string or names.array.len != 3 * @as(u64, hp.n_layer)) return error.Unsupported;
    var it = names.array.iterator();
    while (it.nextString()) |n| {
        const expert = std.mem.endsWith(u8, n, ".ffn_gate_exps.weight") or std.mem.endsWith(u8, n, ".ffn_up_exps.weight") or std.mem.endsWith(u8, n, ".ffn_down_exps.weight");
        if (!expert) return error.Unsupported;
    }
    const block = file.get("prism.hadamard.block_size") orelse return error.MissingMetadata;
    const block_ok = switch (block) {
        .u32 => |x| x == 128,
        .i32 => |x| x == 128,
        else => false,
    };
    if (!block_ok) return error.Unsupported;
    const widths = file.get("prism.hadamard.sign_widths") orelse return error.MissingMetadata;
    const values = file.get("prism.hadamard.sign_values") orelse return error.MissingMetadata;
    if (widths != .array or values != .array or values.array.elem_type != .i32) return error.BadMetadata;
    var out: struct { embd: []f32, ff: []f32 } = .{ .embd = &.{}, .ff = &.{} };
    var off: usize = 0;
    for (0..@intCast(widths.array.len)) |k| {
        const w: usize = widths.array.get(u32, k);
        if (off + w > values.array.len) return error.BadMetadata;
        const s = try arena.alloc(f32, w);
        for (s, 0..) |*v, j| v.* = @floatFromInt(@as(i32, @bitCast(values.array.get(u32, off + j))));
        if (w == hp.n_embd) out.embd = s;
        if (w == hp.n_ff_exp) out.ff = s;
        off += w;
    }
    if (out.embd.len == 0 or out.ff.len == 0) return error.Unsupported;
    return .{ .embd = out.embd, .ff = out.ff };
}

fn upload32(gpa: Allocator, rec: *cmd.Recorder, data: []const f32) Allocator.Error!Arg {
    return uploadBytes(gpa, rec, std.mem.sliceAsBytes(data));
}

/// `data` must stay alive until the batch executes.
fn uploadBytes(gpa: Allocator, rec: *cmd.Recorder, data: []const u8) Allocator.Error!Arg {
    const size = std.mem.alignForward(u64, data.len, 4);
    const buf = try rec.createBuffer(gpa, size, .{ .storage = true, .copy_dst = true });
    try rec.writeBuffer(gpa, buf, 0, data);
    return .{ .buffer = buf, .size = size };
}

pub const Config = struct {
    /// Positions the KV caches and the RoPE table hold, at most max_context.
    context: u32 = max_context,
    kv: common.KvFormat = .f16,
    /// Recurrent-state checkpoint slots.
    checkpoints: u32 = 3,
    tuning: common.Tuning = .{},
};

/// One launch of the step body, and how its grid scales with the step's m
/// tokens: by `mul` rows per token.
const Item = struct { launch: kernels.Launch, mul: u32 = 1 };

pub const Program = struct {
    hp: Hparams,
    cfg: Config,
    step: Arg,
    token_ids: Arg,
    /// The residual streams, [token][hc · n_embd].
    res: Arg,
    /// A block's input, [token][n_embd].
    cur: Arg,
    /// The PLE rows as read from the file, [token][16 rows], `table.stride`
    /// bytes apart.
    ple_rows: Arg,
    logits: Arg,
    /// Every DeltaNet layer's state and conv history, then the PLE block's
    /// convolution history: the state a checkpoint copies.
    recurrent: Arg,
    checkpoints: []const cmd.Buffer,
    table: PleTable,
    /// The tokens the next step's PLE rows hash after, and each checkpoint
    /// slot's.
    history: ple.History = .{},
    saved: []ple.History,
    body: []const Item,
    /// The token embeddings: the body's first `embeds` items.
    embeds: usize,
    /// Per layer, the body index just past its router and past its end:
    /// where a Readback copies the chosen experts and the streams.
    route_end: []const usize,
    layer_end: []const usize,
    /// The router's choices, [token][n_expert_used] expert ids.
    ids: Arg,
    head: []const Item,
    token_stage: []u32,
    step_value: P.Step = .{ .m = 0, .pos0 = 0, .u = 0 },
    // The sampler, as Qwen 3.5's program runs it.
    sample_params: Arg,
    bias: Arg,
    prompt_ids: Arg,
    prompt_mask: Arg,
    counts: Arg,
    sample_state: Arg,
    mark: kernels.Launch,
    sample: std.EnumArray(kernels.SamplePhase, [4]kernels.Launch),
    sample_steps: [kernels.max_sample_steps]kernels.SampleStep = undefined,
    sample_len: usize = 0,
    sample_value: P.Sample = undefined,
    prompt_stage: []u32,
    bias_stage: [][2]u32,

    pub const BuildError = common.BuildError;

    /// Records the creation of every buffer and bind group. `arena` owns the
    /// result and must outlive the Program.
    pub fn build(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *kernels.Set, m: *const Model, cfg: Config, limits: common.Limits) BuildError!Program {
        const hp = m.hp;
        if (cfg.context == 0 or cfg.context > max_context) return error.BadConfig;
        if (hp.head_dim > kernels.max_head_dim or hp.ssm_state > kernels.max_delta_head) return error.BadConfig;
        if (hp.n_head % hp.n_head_kv != 0 or hp.n_head / hp.n_head_kv > kernels.attn_max_group_wide or hp.n_embd > 2560 or hp.n_ff_exp > 2560) return error.BadConfig;
        const cap: u32 = kernels.decode_max_rows;
        const e: u64 = hp.n_embd;
        const hd: u64 = hp.hcDim();
        const used: u64 = hp.n_expert_used;
        const ff: u64 = hp.n_ff_exp;
        var a: Alloc = .{ .gpa = gpa, .rec = rec, .limits = limits };

        const step = try a.buffer(@sizeOf(P.Step), .{ .uniform = true, .copy_dst = true });
        const token_ids = try a.storage(cap);
        const res = try a.storage(cap * hd);
        const cur = try a.storage(cap * e);
        const xn = try a.storage(cap * hd);
        const lo = try a.storage(cap * @as(u64, hp.hc_rank));
        const gate = try a.storage(cap * hd);
        const inject = try a.storage(cap * @as(u64, hp.hc));
        const block = try a.storage(cap * e);
        const logits = try a.storage(cap * @as(u64, hp.vocab));
        // Linear attention.
        const conv_dim: u64 = hp.convChannels();
        const proj_row: u64 = conv_dim + hp.ssm_inner + 2 * @as(u64, hp.ssm_v_heads);
        const proj = try a.storage(cap * proj_row);
        const conv = try a.storage(cap * conv_dim);
        const g = try a.storage(cap * hp.ssm_v_heads);
        const beta = try a.storage(cap * hp.ssm_v_heads);
        const o = try a.storage(cap * hp.ssm_inner);
        // Full attention.
        const hdim: u64 = hp.head_dim;
        const kv_dim = hdim * hp.n_head_kv;
        const qkv_row = 2 * hdim * hp.n_head + 2 * kv_dim;
        const qkv = try a.storage(cap * qkv_row);
        const att = try a.storage(cap * hdim * hp.n_head);
        const max_blocks: u32 = std.math.divCeil(u32, cfg.context, kernels.attn_block) catch unreachable;
        const attn_part = try a.storage(cap * hp.n_head * @as(u64, max_blocks) * (hdim + 2));
        // Experts.
        const ids = try a.storage(cap * used);
        const weights = try a.storage(cap * used);
        const shared_logit = try a.storage(cap);
        const xh = try a.storage(cap * e);
        const gu = try a.storage(cap * used * 2 * ff);
        const act = try a.storage(cap * used * ff);
        const down = try a.storage(cap * used * e);
        const sh_act = try a.storage(cap * @as(u64, hp.n_ff_shexp));
        const sh_out = try a.storage(cap * e);
        // PLE.
        const ple_rows = try a.buffer(cap * ple_heads * @as(u64, m.ple.stride), .{ .storage = true, .copy_dst = true });
        const ple_index = try a.storage(cap * ple_heads);
        const ple_emb = try a.storage(cap * e);
        const ple_key = try a.storage(cap * hd);
        const ple_value = try a.storage(cap * e);
        const ple_gated = try a.storage(cap * hd);
        const ple_normed = try a.storage(cap * hd);
        const index = try arena.alloc(u32, cap * ple_heads);
        for (index, 0..) |*v, i| v.* = @intCast(i);
        try rec.writeBuffer(gpa, ple_index.buffer, 0, std.mem.sliceAsBytes(index));

        const rope_table = try kernels.ropeTable(arena, cfg.context, hp.n_rot, hp.rope_base);
        const rope = try a.storage(rope_table.len);
        try rec.writeBuffer(gpa, rope.buffer, 0, std.mem.sliceAsBytes(rope_table));

        // Recurrent state: per DeltaNet layer, the state then the conv
        // history, each at an aligned offset so it can be bound alone.
        const al = limits.offset_align;
        const state_bytes: u64 = 4 * @as(u64, hp.ssm_v_heads) * hp.ssm_state * hp.ssm_state;
        const hist_bytes: u64 = 4 * @as(u64, hp.ssm_conv - 1) * conv_dim;
        const hist_at = std.mem.alignForward(u64, state_bytes, al);
        const per_layer = hist_at + std.mem.alignForward(u64, hist_bytes, al);
        var n_linear: u64 = 0;
        for (0..hp.n_layer) |il| n_linear += @intFromBool(hp.recurrent(il));
        const ple_hist_bytes: u64 = 4 * @as(u64, hp.ple_conv - 1) * hp.ple_dilation * hd;
        const recurrent = try a.buffer(per_layer * n_linear + ple_hist_bytes, .{ .storage = true, .copy_dst = true, .copy_src = true });
        const ple_hist: Arg = .{ .buffer = recurrent.buffer, .offset = per_layer * n_linear, .size = ple_hist_bytes };
        const checkpoints = try arena.alloc(cmd.Buffer, cfg.checkpoints);
        for (checkpoints) |*cp| cp.* = (try a.buffer(recurrent.size, .{ .copy_src = true, .copy_dst = true })).buffer;
        // No step keeps a row: every layer binds the same unwritten stand-in.
        const kept = try a.buffer(16, .{ .storage = true });

        const sample_params = try a.buffer(@sizeOf(P.Sample), .{ .uniform = true, .copy_dst = true });
        const bias = try a.storage(2 * kernels.max_bias);
        const prompt_ids = try a.storage(cfg.context);
        const prompt_mask = try a.storage(hp.vocab);
        const counts = try a.storage(hp.vocab);
        const work = try a.storage(hp.vocab);
        const sample_state = try a.storage(kernels.sampleStateWords(hp.vocab));
        const kv_bytes: u64 = if (cfg.kv == .f16) 2 else 4;
        const kv_layer = @as(u64, cfg.context) * kv_dim * kv_bytes;

        var b: kernels.Builder = .{ .gpa = gpa, .arena = arena, .rec = rec, .set = set };
        defer b.deinit();
        const mm: Mm = .{
            .b = &b,
            .arena = arena,
            .cap = cap,
            .eps = hp.eps,
            // The decode multiplies use neither.
            .partials = try a.storage(1),
            .act = .{ null, null },
            .tuning = cfg.tuning,
        };
        var body: std.ArrayList(Item) = .empty;
        const add = struct {
            fn launches(list: *std.ArrayList(Item), al_: Allocator, ls: []const kernels.Launch) Allocator.Error!void {
                for (ls) |l| try list.append(al_, .{ .launch = l });
            }
        }.launches;

        for (m.tok_embd.chunks) |ch| {
            try body.append(arena, .{ .launch = try b.add(weightKernel(.embed, ch.type), P.EmbedBf16{ .dim = hp.n_embd, .row0 = ch.row0, .chunk_rows = ch.rows }, &.{ token_ids, ch.arg, cur }, .rows_x) });
        }
        // (recordStep copies each token's embedding into its hc streams.)
        const embed_end = body.items.len;

        const hcn: P.Hc = .{ .dim = hp.n_embd, .hc = hp.hc };
        const col_groups: u32 = std.math.divCeil(u32, hp.n_embd, 256) catch unreachable;
        const Mix = struct {
            /// The streams' norms, the low-rank gate, the mix into `cur`,
            /// and (with an inject matrix) the stream weights.
            fn build(list: *std.ArrayList(Item), bb: *kernels.Builder, m_: Mm, al_: Allocator, h: *const Hc, hp_: Hparams, bufs: anytype) Program.BuildError!void {
                try list.append(al_, .{ .launch = try bb.add(.rmsnorm, P.RmsNorm{ .rows = 0, .dim = hp_.n_embd, .eps = hp_.eps, .w_rows = hp_.hc }, &.{ bufs.res, h.norm, bufs.xn }, .rows_x), .mul = hp_.hc });
                try add(list, al_, try m_.vector(bufs.lo, bufs.xn, &h.down, .{}));
                try list.append(al_, .{ .launch = try bb.add(.hc_act, P.HcAct{ .count = kernels.decode_max_rows * hp_.hc_rank, .hc = hp_.hc }, &.{bufs.lo}, .{ .elems = hp_.hc_rank }) });
                try add(list, al_, try m_.vector(bufs.gate, bufs.lo, &h.up, .{}));
                try list.append(al_, .{ .launch = try bb.add(.hc_mix, P.Hc{ .dim = hp_.n_embd, .hc = hp_.hc }, &.{ bufs.xn, bufs.gate, bufs.cur }, .{ .rows_y = bufs.col_groups }) });
                if (h.inject) |*inj| try add(list, al_, try m_.vector(bufs.inject, bufs.xn, inj, .{}));
            }
        };
        const bufs = .{ .res = res, .xn = xn, .lo = lo, .gate = gate, .cur = cur, .inject = inject, .col_groups = col_groups };

        var lin_i: u64 = 0;
        const route_end = try arena.alloc(usize, hp.n_layer);
        const layer_end = try arena.alloc(usize, hp.n_layer);
        for (m.layers, route_end, layer_end) |*layer, *re, *le| {
            if (layer.ple) |*pw| {
                // The token's 16 table rows, decoded as a 16-row Q8_0 table.
                try body.append(arena, .{ .launch = try b.add(.embed_q8_0, P.EmbedBf16{ .dim = hp.ple_head_dim, .row0 = 0, .chunk_rows = cap * ple_heads, .stride = m.ple.stride }, &.{ ple_index, ple_rows, ple_emb }, .rows_x), .mul = ple_heads });
                try add(&body, arena, try mm.vector(ple_key, ple_emb, &pw.key, .{}));
                try add(&body, arena, try mm.vector(ple_value, ple_emb, &pw.value, .{}));
                try body.append(arena, .{ .launch = try b.add(.ple_mix, P.PleMix{ .dim = hp.n_embd, .hc = hp.hc, .eps = hp.eps }, &.{ ple_key, res, ple_value, pw.norm_key, pw.norm_query, pw.norm_conv, ple_gated, ple_normed }, .{ .rows_y = hp.hc }) });
                try body.append(arena, .{ .launch = try b.add(.ple_conv, P.PleConv{ .channels = @intCast(hd), .taps = hp.ple_conv, .dilation = hp.ple_dilation }, &.{ ple_gated, ple_normed, pw.conv, ple_hist, res, step }, .{ .fixed = .{ std.math.divCeil(u32, @intCast(hd), 256) catch unreachable, 1, 1 } }) });
            }
            try Mix.build(&body, &b, mm, arena, &layer.hc_attn, hp, bufs);
            switch (layer.mixer) {
                .linear => |*w| {
                    const base = lin_i * per_layer;
                    lin_i += 1;
                    const state: Arg = .{ .buffer = recurrent.buffer, .offset = base, .size = state_bytes };
                    const hist: Arg = .{ .buffer = recurrent.buffer, .offset = base + hist_at, .size = hist_bytes };
                    const d = hp.ssm_state;
                    try add(&body, arena, try mm.vector(proj, cur, &w.proj, .{}));
                    try body.append(arena, .{ .launch = try b.add(.delta_prep, P.DeltaPrep{
                        .channels = @intCast(conv_dim),
                        .taps = hp.ssm_conv,
                        .key_channels = 2 * d * hp.ssm_k_heads,
                        .head = d,
                        .row = @intCast(proj_row),
                        .v_heads = hp.ssm_v_heads,
                        .beta_off = @intCast(conv_dim + hp.ssm_inner),
                        .alpha_off = @intCast(conv_dim + hp.ssm_inner + hp.ssm_v_heads),
                        .eps = hp.eps,
                    }, &.{ proj, w.conv, hist, conv, w.dt_bias, w.a, g, beta }, .{ .rows_y = @intCast(conv_dim / d + 1) }) });
                    try body.append(arena, .{ .launch = try b.add(.delta_rule, P.DeltaRule{
                        .k_heads = hp.ssm_k_heads,
                        .v_heads = hp.ssm_v_heads,
                        .dim = d,
                        .conv_stride = @intCast(conv_dim),
                        .taps = hp.ssm_conv,
                        .row = @intCast(proj_row),
                    }, &.{ conv, g, beta, state, o, step, proj, hist, kept }, .{ .fixed = .{ hp.ssm_v_heads, std.math.divCeil(u32, d, kernels.delta_cols) catch unreachable, 1 } }) });
                    try body.append(arena, .{ .launch = try b.add(.gated_norm, P.GatedNorm{
                        .v_heads = hp.ssm_v_heads,
                        .dim = d,
                        .eps = hp.eps,
                        .z_row = @intCast(proj_row),
                        .z_off = @intCast(conv_dim),
                        .sigmoid = 1,
                    }, &.{ o, proj, w.norm }, .{ .rows_y = hp.ssm_v_heads }) });
                    try add(&body, arena, try mm.vector(block, o, &w.out, .{}));
                },
                .full => |*w| {
                    const kc = try a.buffer(kv_layer, .{ .storage = true });
                    const vc = try a.buffer(kv_layer, .{ .storage = true });
                    const half: u32 = @backingInt(cfg.kv);
                    const q_row: u32 = @intCast(qkv_row);
                    try add(&body, arena, try mm.vector(qkv, cur, &w.qkv, .{}));
                    try body.append(arena, .{ .launch = try b.add(.rope_append, P.RopeAppend{
                        .heads = hp.n_head,
                        .kv_heads = hp.n_head_kv,
                        .head_dim = hp.head_dim,
                        .n_rot = hp.n_rot,
                        .row = q_row,
                        .k_off = @intCast(2 * hdim * hp.n_head),
                        .v_off = @intCast(2 * hdim * hp.n_head + kv_dim),
                        .f16 = half,
                        .eps = hp.eps,
                    }, &.{ qkv, w.q_norm, w.k_norm, rope, kc, vc, step }, .{ .rows_y = hp.n_head + hp.n_head_kv }) });
                    const params: P.Attention = .{
                        .heads = hp.n_head,
                        .kv_heads = hp.n_head_kv,
                        .head_dim = hp.head_dim,
                        .f16 = half,
                        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hp.head_dim))),
                        .max_blocks = max_blocks,
                        .q_row = q_row,
                        .row0 = 0,
                    };
                    try body.append(arena, .{ .launch = try b.add(kernels.attnSplit(hp.n_head / hp.n_head_kv), params, &.{ qkv, kc, vc, attn_part, step }, .{ .key_blocks = .{ .x = hp.n_head_kv, .block = kernels.attn_block, .row0 = 0, .max_rows = kernels.attn_rows } }) });
                    try body.append(arena, .{ .launch = try b.add(.attn_combine, params, &.{ qkv, attn_part, att, step }, .{ .rows_y_from = .{ .x = hp.n_head, .row0 = 0, .max_rows = kernels.attn_rows } }) });
                    try add(&body, arena, try mm.vector(block, att, &w.wo, .{}));
                },
            }
            try body.append(arena, .{ .launch = try b.add(.hc_combine, hcn, &.{ block, inject, res }, .{ .rows_y = col_groups }) });

            // The experts.
            try Mix.build(&body, &b, mm, arena, &layer.hc_ffn, hp, bufs);
            try body.append(arena, .{ .launch = try b.add(.moe_route, P.MoeRoute{ .n_expert = hp.n_expert, .n_used = hp.n_expert_used, .cols = hp.n_embd, .x_stride = hp.n_embd, .scale = hp.expert_weights_scale }, &.{ layer.router, layer.shared_gate, cur, ids, weights, shared_logit }, .rows_x) });
            re.* = body.items.len;
            try body.append(arena, .{ .launch = try b.add(.rotate128, P.Rotate128{ .width = hp.n_embd, .x_stride = hp.n_embd, .y_stride = hp.n_embd }, &.{ cur, m.signs_embd, xh }, .{ .rows_y = hp.n_embd / 128 }) });
            try body.append(arena, .{ .launch = try b.add(.gyro_matvec, P.GyroMatvec{
                .rows = hp.n_ff_exp,
                .cols = hp.n_embd,
                .n_used = hp.n_expert_used,
                .x_per_slot = 0,
                .x_stride = hp.n_embd,
                .out_stride = 2 * hp.n_ff_exp,
                .out_z = hp.n_ff_exp,
            }, &.{ layer.gate_exps, layer.up_exps, xh, gu, ids, m.codebook }, .{ .tiles_y = .{ .x = std.math.divCeil(u32, hp.n_ff_exp, 8) catch unreachable, .tile = 1, .z = 2 } }), .mul = hp.n_expert_used });
            try body.append(arena, .{ .launch = try b.add(.rotate128, P.Rotate128{ .width = hp.n_ff_exp, .x_stride = 2 * hp.n_ff_exp, .y_stride = hp.n_ff_exp, .silu_pair = 1 }, &.{ gu, m.signs_ff, act }, .{ .rows_y = hp.n_ff_exp / 128 }), .mul = hp.n_expert_used });
            try body.append(arena, .{ .launch = try b.add(.gyro_matvec, P.GyroMatvec{
                .rows = hp.n_embd,
                .cols = hp.n_ff_exp,
                .n_used = hp.n_expert_used,
                .x_per_slot = 1,
                .x_stride = hp.n_ff_exp,
                .out_stride = hp.n_embd,
                .out_z = 0,
            }, &.{ layer.down_exps, layer.down_exps, act, down, ids, m.codebook }, .{ .tiles_y = .{ .x = std.math.divCeil(u32, hp.n_embd, 8) catch unreachable, .tile = 1, .z = 1 } }), .mul = hp.n_expert_used });
            try add(&body, arena, try mm.vector(sh_act, cur, &layer.sh_gu, .{ .silu_pair = hp.n_ff_shexp }));
            try add(&body, arena, try mm.vector(sh_out, sh_act, &layer.sh_down, .{}));
            try body.append(arena, .{ .launch = try b.add(.moe_combine, P.MoeCombine{ .n_used = hp.n_expert_used, .dim = hp.n_embd }, &.{ down, weights, sh_out, shared_logit, block }, .{ .rows_y = col_groups }) });
            try body.append(arena, .{ .launch = try b.add(.hc_combine, hcn, &.{ block, inject, res }, .{ .rows_y = col_groups }) });
            le.* = body.items.len;
        }

        var head: std.ArrayList(Item) = .empty;
        try Mix.build(&head, &b, mm, arena, &m.hc_head, hp, bufs);
        try add(&head, arena, try mm.vector(logits, cur, &m.output, .{}));
        const mark = try b.addWith(.mark_tokens, sample_params, &.{ prompt_ids, prompt_mask }, .{ .elems = 1 });
        const sampler_args: common.SamplerArgs = .{ .bias = bias, .prompt_mask = prompt_mask, .counts = counts, .work = work, .state = sample_state, .next = token_ids, .params = sample_params };
        const sample = try sampler_args.launches(&b, hp.vocab, logits, step);
        try b.finish();

        return .{
            .hp = hp,
            .cfg = cfg,
            .step = step,
            .token_ids = token_ids,
            .res = res,
            .cur = cur,
            .ple_rows = ple_rows,
            .logits = logits,
            .recurrent = recurrent,
            .checkpoints = checkpoints,
            .table = m.ple,
            .saved = try arena.alloc(ple.History, cfg.checkpoints),
            .body = body.items,
            .embeds = embed_end,
            .route_end = route_end,
            .layer_end = layer_end,
            .ids = ids,
            .head = head.items,
            .token_stage = try arena.alloc(u32, cap),
            .sample_params = sample_params,
            .bias = bias,
            .prompt_ids = prompt_ids,
            .prompt_mask = prompt_mask,
            .counts = counts,
            .sample_state = sample_state,
            .mark = mark,
            .sample = sample,
            .prompt_stage = try arena.alloc(u32, cfg.context),
            .bias_stage = try arena.alloc([2]u32, kernels.max_bias),
        };
    }

    /// Zeroes the recurrent state (the PLE convolution's too) and the token
    /// history: the next step starts a sequence at position 0.
    pub fn recordReset(p: *Program, gpa: Allocator, rec: *cmd.Recorder) Allocator.Error!void {
        try rec.clearBuffer(gpa, p.recurrent.buffer, 0, p.recurrent.size);
        p.history = .{};
    }

    pub fn recordSave(p: *Program, gpa: Allocator, rec: *cmd.Recorder, slot: u32) Allocator.Error!void {
        try rec.copyBuffer(gpa, p.recurrent.buffer, 0, p.checkpoints[slot], 0, p.recurrent.size);
        p.saved[slot] = p.history;
    }

    pub fn recordRestore(p: *Program, gpa: Allocator, rec: *cmd.Recorder, slot: u32) Allocator.Error!void {
        try rec.copyBuffer(gpa, p.checkpoints[slot], 0, p.recurrent.buffer, 0, p.recurrent.size);
        p.history = p.saved[slot];
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
        pos0: u32,
        tokens: []const u32,
        /// 0, or every row: project the rows to logits.
        logit_rows: u32 = 0,
        /// Sample from the row with this draw (a one-token step with its
        /// logits), reading back (token, is_stop) into `sampled`.
        draw: ?f32 = null,
        sampled: ?*[2]u32 = null,
        /// The rows' logits, read back here ([tokens][vocab]), if given.
        logits: ?[]f32 = null,
        readback: ?*const Readback = null,
    };

    /// Activation readbacks for checks, per layer: the streams after it
    /// ([tokens][hc · n_embd]) and the experts its router chose
    /// ([tokens][n_expert_used]).
    pub const Readback = struct {
        streams: []const []f32,
        experts: []const []u32,
    };

    pub const StepError = Allocator.Error || error{InvalidToken};

    pub fn recordStep(p: *Program, gpa: Allocator, rec: *cmd.Recorder, s: Step) StepError!void {
        const m: u32 = @intCast(s.tokens.len);
        const e: usize = p.hp.n_embd;
        const hc: usize = p.hp.hc;
        std.debug.assert(m > 0 and m <= kernels.decode_max_rows and s.pos0 + m <= p.cfg.context);
        std.debug.assert(s.logit_rows == 0 or s.logit_rows == m);
        std.debug.assert(s.draw == null or (m == 1 and s.logit_rows == 1));
        p.step_value = .{ .m = m, .pos0 = s.pos0, .u = s.draw orelse 0 };
        try rec.writeBuffer(gpa, p.step.buffer, 0, std.mem.asBytes(&p.step_value));
        @memcpy(p.token_stage[0..m], s.tokens);
        try rec.writeBuffer(gpa, p.token_ids.buffer, 0, std.mem.sliceAsBytes(p.token_stage[0..m]));
        // Each token's PLE rows, straight from the file.
        const t = p.table;
        for (s.tokens, 0..) |token, i| {
            for (try p.history.accept(t.config, token), 0..) |row, head| {
                try rec.writeFile(gpa, p.ple_rows.buffer, (i * ple_heads + head) * t.stride, t.at + row * t.row_bytes, t.stride);
            }
        }

        const ext = struct {
            fn of(item: Item, m_: u32, keys: u32) kernels.Extent {
                return .{ .rows = m_ * item.mul, .keys = keys };
            }
        }.of;
        const keys = s.pos0 + m;
        // The embeddings, then into each of the token's streams.
        for (p.body[0..p.embeds]) |item| try item.launch.record(gpa, rec, ext(item, m, keys));
        for (0..m) |r| for (0..hc) |c| {
            try rec.copyBuffer(gpa, p.cur.buffer, r * e * 4, p.res.buffer, (r * hc + c) * e * 4, e * 4);
        };
        var il: usize = 0;
        for (p.body[p.embeds..], p.embeds..) |item, at| {
            try item.launch.record(gpa, rec, ext(item, m, keys));
            const r = s.readback orelse continue;
            if (at + 1 == p.route_end[il]) {
                try rec.readBuffer(gpa, common.read_token, p.ids.buffer, 0, std.mem.sliceAsBytes(r.experts[il][0 .. m * p.hp.n_expert_used]));
            }
            if (at + 1 == p.layer_end[il]) {
                try rec.readBuffer(gpa, common.read_token, p.res.buffer, 0, std.mem.sliceAsBytes(r.streams[il][0 .. m * hc * e]));
                il += 1;
            }
        }
        if (s.logit_rows == 0) return;
        for (p.head) |item| try item.launch.record(gpa, rec, ext(item, m, keys));
        if (s.logits) |dst| {
            const vocab: usize = p.hp.vocab;
            try rec.readBuffer(gpa, common.read_token, p.logits.buffer, 0, std.mem.sliceAsBytes(dst[0 .. m * vocab]));
        }
        if (s.draw != null) {
            std.debug.assert(p.sample_len > 0);
            for (p.sample_steps[0..p.sample_len]) |ss| try p.sample.get(ss.phase)[ss.round].record(gpa, rec, .{ .rows = 1 });
            try rec.readBuffer(gpa, common.read_token, p.sample_state.buffer, 0, std.mem.asBytes(s.sampled.?));
        }
    }
};
