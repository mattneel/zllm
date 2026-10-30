//! GPU execution plan for the Qwen 3.5 hybrid model.
//!
//! `Model` uploads the weights; `Program` builds everything a step needs
//! once (buffers, persistent state, prebuilt launches) and records steps:
//! m tokens at position pos0, optionally projecting the last rows to logits
//! and sampling the next token on the GPU. Prefill is a step with many
//! tokens, decode a step with one.
//!
//! Layout: matrices stay as stored (BF16, F16 or quantized) and are split by rows into chunks that each fit
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
const common = @import("common.zig");
const Limits = common.Limits;
const Matrix = common.Matrix;
const KvFormat = common.KvFormat;
const Tuning = common.Tuning;
const Upload = common.Upload;
const Alloc = common.Alloc;
const Mm = common.Mm;
const SamplerArgs = common.SamplerArgs;
const weightKernel = common.weightKernel;
const concat = common.concat;
const read_token = common.read_token;

pub const Hparams = graph.Hparams;

const Attention = struct {
    /// Q (with the output gates) | K | V, stacked by rows.
    wqkv: Matrix,
    wo: Matrix,
    q_norm: Arg,
    k_norm: Arg,
};

const DeltaNet = struct {
    /// qkv | z (the output gate) | β | α, stacked by rows.
    wproj: Matrix,
    wout: Matrix,
    conv: Arg,
    dt_bias: Arg,
    a: Arg,
    norm: Arg,
};

const Layer = struct {
    attn_norm: Arg,
    post_norm: Arg,
    /// gate | up, stacked by rows.
    ffn_gu: Matrix,
    ffn_down: Matrix,
    mixer: union(graph.LayerKind) {
        delta_net: DeltaNet,
        attention: Attention,
    },
};

/// The multi-token prediction block (src/ref/qwen35.zig `Mtp`).
const Mtp = struct {
    enorm: Arg,
    hnorm: Arg,
    /// [enorm(embedding), hnorm(hidden)] → n_embd.
    eh_proj: Matrix,
    /// A full-attention block.
    block: Layer,
    head_norm: Arg,
};

/// Weights resident on the GPU.
pub const Model = struct {
    hp: Hparams,
    tok_embd: Matrix,
    output: ?Matrix,
    output_norm: Arg,
    layers: []const Layer,
    /// The MTP block, for models that have one.
    mtp: ?Mtp,

    pub const UploadError = Hparams.Error || common.UploadError;

    /// Calls f(context, w) for every multiplied weight (not the embedding
    /// table).
    pub fn eachMatrix(m: *const Model, context: anytype, comptime f: fn (@TypeOf(context), *const Matrix) void) void {
        if (m.output) |*o| f(context, o);
        for (m.layers) |*l| eachLayerMatrix(l, context, f);
        if (m.mtp) |*t| {
            f(context, &t.eh_proj);
            eachLayerMatrix(&t.block, context, f);
        }
    }

    fn eachLayerMatrix(l: *const Layer, context: anytype, comptime f: fn (@TypeOf(context), *const Matrix) void) void {
        f(context, &l.ffn_gu);
        f(context, &l.ffn_down);
        switch (l.mixer) {
            .attention => |*w| {
                f(context, &w.wqkv);
                f(context, &w.wo);
            },
            .delta_net => |*w| {
                f(context, &w.wproj);
                f(context, &w.wout);
            },
        }
    }

    /// Records buffer creation and an upload from the model file
    /// (`write_file`) for every weight; the host reads the file. `arena` owns
    /// the returned layout.
    pub fn upload(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *const kernels.Set, file: gguf.File, limits: Limits) UploadError!Model {
        const hp = try Hparams.fromGguf(file);
        var u: Upload = .{ .arena = arena, .gpa = gpa, .rec = rec, .set = set, .file = file, .limits = limits };
        const e: u64 = hp.n_embd;
        const ff: u64 = hp.n_ff;

        const layers = try arena.alloc(Layer, hp.n_layer);
        for (layers, hp.kinds(), 0..) |*layer, kind, i| {
            layer.* = .{
                .attn_norm = try u.vector("blk.{d}.attn_norm.weight", i, e),
                .post_norm = try u.vector("blk.{d}.post_attention_norm.weight", i, e),
                .ffn_gu = try u.stacked(i, e, &.{ .{ "blk.{d}.ffn_gate.weight", ff }, .{ "blk.{d}.ffn_up.weight", ff } }),
                .ffn_down = try u.matrix("blk.{d}.ffn_down.weight", i, ff, e),
                .mixer = switch (kind) {
                    .attention => .{ .attention = try uploadAttention(&u, hp, i) },
                    .delta_net => blk: {
                        const conv_dim: u64 = 2 * hp.gdn_head_dim * hp.gdn_k_heads + hp.gdn_inner;
                        break :blk .{ .delta_net = .{
                            .wproj = try u.stacked(i, e, &.{
                                .{ "blk.{d}.attn_qkv.weight", conv_dim },
                                .{ "blk.{d}.attn_gate.weight", hp.gdn_inner },
                                .{ "blk.{d}.ssm_beta.weight", hp.gdn_v_heads },
                                .{ "blk.{d}.ssm_alpha.weight", hp.gdn_v_heads },
                            }),
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

        const mtp: ?Mtp = if (hp.n_layer_nextn == 1) blk: {
            const i: usize = hp.n_layer;
            break :blk .{
                .enorm = try u.vector("blk.{d}.nextn.enorm.weight", i, e),
                .hnorm = try u.vector("blk.{d}.nextn.hnorm.weight", i, e),
                .eh_proj = try u.matrix("blk.{d}.nextn.eh_proj.weight", i, 2 * e, e),
                .block = .{
                    .attn_norm = try u.vector("blk.{d}.attn_norm.weight", i, e),
                    .post_norm = try u.vector("blk.{d}.post_attention_norm.weight", i, e),
                    .ffn_gu = try u.stacked(i, e, &.{ .{ "blk.{d}.ffn_gate.weight", ff }, .{ "blk.{d}.ffn_up.weight", ff } }),
                    .ffn_down = try u.matrix("blk.{d}.ffn_down.weight", i, ff, e),
                    .mixer = .{ .attention = try uploadAttention(&u, hp, i) },
                },
                // The model's output norm where the block has no own.
                .head_norm = if (u.has("blk.{d}.nextn.shared_head_norm.weight", i))
                    try u.vector("blk.{d}.nextn.shared_head_norm.weight", i, e)
                else
                    try u.vector("output_norm.weight", null, e),
            };
        } else null;

        return .{
            .hp = hp,
            .mtp = mtp,
            .tok_embd = try u.matrix("token_embd.weight", null, e, hp.vocab),
            .output = if (file.tensor("output.weight") != null) try u.matrix("output.weight", null, e, hp.vocab) else null,
            .output_norm = try u.vector("output_norm.weight", null, e),
            .layers = layers,
        };
    }
};

/// Layer `i`'s gated full attention.
fn uploadAttention(u: *Upload, hp: Hparams, i: usize) Model.UploadError!Attention {
    const e: u64 = hp.n_embd;
    const hd: u64 = hp.head_dim;
    return .{
        .wqkv = try u.stacked(i, e, &.{
            .{ "blk.{d}.attn_q.weight", 2 * hd * hp.n_head },
            .{ "blk.{d}.attn_k.weight", hd * hp.n_head_kv },
            .{ "blk.{d}.attn_v.weight", hd * hp.n_head_kv },
        }),
        .wo = try u.matrix("blk.{d}.attn_output.weight", i, hd * hp.n_head, e),
        .q_norm = try u.vector("blk.{d}.attn_q_norm.weight", i, hd),
        .k_norm = try u.vector("blk.{d}.attn_k_norm.weight", i, hd),
    };
}

/// Whether a program for `m` can be built with `mtp`: the model has an MTP
/// block and a logits row the storage offset alignment divides (the verify
/// step's second sampler binds row 1).
pub fn canSpeculate(m: *const Model, limits: Limits) bool {
    return m.mtp != null and (@as(u64, m.hp.vocab) * 4) % limits.offset_align == 0;
}

/// Whether speculating pays off for `m`, by default: a speculative round
/// costs a two-position step (about one step: decoding a weight serves
/// both positions) plus a draft, which runs the MTP block and the whole
/// output head. Where the draft's weights are at most an eighth of the main
/// model's, rounds cost little more than steps and acceptance rates seen in
/// practice (70% and up) make them pay; in a small model the head alone is
/// a large share (the 0.8B's is about 30%), and drafts cost more than they
/// save.
pub fn speculationPays(m: *const Model) bool {
    const t = m.mtp orelse return false;
    const Bytes = struct {
        n: u64 = 0,
        fn add(s: *@This(), w: *const Matrix) void {
            for (w.chunks) |ch| s.n += ch.arg.size;
        }
    };
    var main: Bytes = .{};
    for (m.layers) |*l| Model.eachLayerMatrix(l, &main, Bytes.add);
    const head = if (m.output) |*o| o else &m.tok_embd;
    main.add(head);
    var draft: Bytes = .{};
    draft.add(&t.eh_proj);
    Model.eachLayerMatrix(&t.block, &draft, Bytes.add);
    draft.add(head);
    return draft.n * 8 <= main.n;
}

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
    /// Decode-multiply variants per weight shape (tuning); the default
    /// variant elsewhere.
    tuning: Tuning = .{},
    /// Build the MTP block's step (recordMtp) and the two-position verify
    /// step (recordVerify); needs a model with an MTP block, logit_rows ≥ 2,
    /// and a logits row size the storage offset alignment divides.
    mtp: bool = false,
};

/// One entry of the step body.
pub const Item = union(enum) {
    /// The same launch on both kernel paths.
    launch: kernels.Launch,
    /// Different launches per kernel path.
    fork: Fork,
    /// The final RMSNorm into `h`: the prefill head reads it; the decode head
    /// folds it into its projection, so decode runs it only for a readback.
    final_norm: kernels.Launch,
    /// The end of a layer, where a check may read the residual stream back.
    layer_end: u32,
};

/// Launches that differ by kernel path: the prefill kernels unfused around
/// tiled matrix multiplies, the decode kernels fused into matrix-vector
/// multiplies.
pub const Fork = struct {
    prefill: []const kernels.Launch,
    decode: []const kernels.Launch,

    pub fn pick(f: Fork, path: Path) []const kernels.Launch {
        return if (path == .decode) f.decode else f.prefill;
    }
};

/// Which kernels a step runs: the prefill kernels, or the decode kernels
/// (fused matrix-vector multiplies, split attention) for steps of at most
/// `kernels.decode_max_rows` tokens. Each is bitwise independent of how a
/// sequence is split into steps, but the two round differently.
pub const Path = enum {
    prefill,
    decode,

    pub fn of(rows: u32) Path {
        return if (rows <= kernels.decode_max_rows) .decode else .prefill;
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
    /// `recurrent` as after the kept row of the last step that kept one.
    kept: Arg,
    checkpoints: []const cmd.Buffer,
    sample_params: Arg,
    bias: Arg,
    prompt_ids: Arg,
    prompt_mask: Arg,
    counts: Arg,
    /// The sampler's state; the drawn token and its stop flag come first.
    sample_state: Arg,

    body: []const Item,
    /// The vocabulary projection: on the prefill path from `lm_in` (rows
    /// copied from `h`), on the decode path from `x` with the final norm.
    head: Fork,
    mark: kernels.Launch,
    /// One launch per sampler phase and round.
    sample: std.EnumArray(kernels.SamplePhase, [4]kernels.Launch),
    /// The phases the current request's draws run: the first `sample_len`.
    sample_steps: [kernels.max_sample_steps]kernels.SampleStep = undefined,
    sample_len: usize = 0,
    /// Tokens per tile of the prefill multiplies: a step's partial tile
    /// costs as much as a whole one (kernels.matmulTiles).
    prefill_tile: u32,
    mtp: ?MtpProgram = null,
    /// speculationPays for the model.
    speculation_pays: bool = false,

    // Upload staging, alive until the batch that writes it has executed.
    step_value: P.Step = .{ .m = 0, .pos0 = 0, .u = 0 },
    sample_value: P.Sample = undefined,
    token_stage: []u32,
    prompt_stage: []u32,
    bias_stage: [][2]u32,

    pub const BuildError = common.BuildError;

    /// Records creation of every buffer and bind group. `arena` owns the
    /// result and the parameter staging, and must outlive the Program.
    pub fn build(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *kernels.Set, m: *const Model, cfg: Config, limits: Limits) BuildError!Program {
        const hp = m.hp;
        if (cfg.max_tokens == 0 or cfg.logit_rows == 0 or cfg.context < cfg.max_tokens or cfg.checkpoints == 0) return error.BadConfig;
        if (hp.head_dim > kernels.max_head_dim or hp.gdn_head_dim > kernels.max_delta_head or hp.n_embd % 4 != 0) return error.BadConfig;
        if (hp.n_head % hp.n_head_kv != 0 or hp.n_head / hp.n_head_kv > kernels.attn_max_group) return error.BadConfig;
        const e: u64 = hp.n_embd;
        const cap: u64 = cfg.max_tokens;
        const conv_dim: u64 = 2 * hp.gdn_head_dim * hp.gdn_k_heads + hp.gdn_inner;
        const kv_dim: u64 = hp.head_dim * hp.n_head_kv;

        var a: Alloc = .{ .gpa = gpa, .rec = rec, .limits = limits };
        const hd: u64 = hp.head_dim;
        // Row widths of the fused projections.
        const qkv_row: u64 = 2 * hd * hp.n_head + 2 * kv_dim;
        const proj_row: u64 = conv_dim + hp.gdn_inner + 2 * @as(u64, hp.gdn_v_heads);
        const x = try a.storage(cap * e);
        const h = try a.storage(cap * e);
        const gu = try a.storage(cap * 2 * hp.n_ff);
        const act = try a.storage(cap * hp.n_ff);
        const qkv = try a.storage(cap * qkv_row);
        const att = try a.storage(cap * hd * hp.n_head);
        const proj = try a.storage(cap * proj_row);
        const conv = try a.storage(cap * conv_dim);
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
        // The recurrent state as after a step's kept row (Step.keep), laid
        // out as `recurrent`; without MTP no step keeps a row, and every
        // layer binds the same unwritten stand-in.
        const kept = try a.buffer(if (cfg.mtp) recurrent.size else 16, .{ .storage = true, .copy_src = true });

        const kv_bytes: u64 = if (cfg.kv == .f16) 2 else 4;
        const kv_layer = @as(u64, cfg.context) * kv_dim * kv_bytes;

        const sample_params = try a.buffer(@sizeOf(P.Sample), .{ .uniform = true, .copy_dst = true });
        const bias = try a.storage(2 * kernels.max_bias);
        const prompt_ids = try a.storage(cfg.context);
        const prompt_mask = try a.storage(hp.vocab);
        const counts = try a.storage(hp.vocab);
        const work = try a.storage(hp.vocab);
        // Attention's per-block partials: (max, sum, output) per (token,
        // head, block of cache positions), for one chunk of a step's rows.
        const max_blocks: u32 = std.math.divCeil(u32, cfg.context, kernels.attn_block) catch unreachable;
        const attn_chunk: u64 = @min(cap, kernels.attn_rows);
        const attn_part = try a.storage(attn_chunk * hp.n_head * max_blocks * (hp.head_dim + 2));
        const sample_state = try a.storage(kernels.sampleStateWords(hp.vocab));

        var b: kernels.Builder = .{ .gpa = gpa, .arena = arena, .rec = rec, .set = set };
        defer b.deinit();
        var body: std.ArrayList(Item) = .empty;
        const capn: u32 = cfg.max_tokens;
        const norm: P.RmsNorm = .{ .rows = capn, .dim = hp.n_embd, .eps = hp.rms_eps };
        const mm: Mm = .{
            .b = &b,
            .arena = arena,
            .cap = capn,
            .eps = hp.rms_eps,
            .partials = try a.storage(Mm.partialFloats(m, cap)),
            .act = .{ try Mm.actScratch(&a, m, cap, 256), try Mm.actScratch(&a, m, cap, 32) },
            .tuning = cfg.tuning,
        };

        const lb: LayerBuilder = .{
            .a = &a,
            .mm = mm,
            .hp = hp,
            .kv_layer = kv_layer,
            .kv = cfg.kv,
            .max_blocks = max_blocks,
            .x = x,
            .h = h,
            .gu = gu,
            .act = act,
            .qkv = qkv,
            .att = att,
            .attn_part = attn_part,
        };
        for (m.tok_embd.chunks) |ch| {
            try body.append(arena, .{ .launch = try b.add(weightKernel(.embed, ch.type), P.EmbedBf16{ .dim = hp.n_embd, .row0 = ch.row0, .chunk_rows = ch.rows }, &.{ token_ids, ch.arg, x }, .rows_x) });
        }
        var delta_i: u64 = 0;
        for (m.layers, 0..) |*layer, il| {
            switch (layer.mixer) {
                .attention => |*w| try lb.attention(&body, layer.attn_norm, w, rope, step),
                .delta_net => |*w| {
                    const base = delta_i * per_layer;
                    delta_i += 1;
                    const state: Arg = .{ .buffer = recurrent.buffer, .offset = base, .size = state_bytes };
                    const hist_at = std.mem.alignForward(u64, state_bytes, al);
                    const hist: Arg = .{ .buffer = recurrent.buffer, .offset = base + hist_at, .size = hist_bytes };
                    const kept_layer: Arg = if (cfg.mtp) .{ .buffer = kept.buffer, .offset = base, .size = hist_at + hist_bytes } else kept;
                    const d = hp.gdn_head_dim;
                    try body.append(arena, .{ .fork = .{
                        .prefill = try mm.tiled(proj, .{ .rmsnorm = .{ .x = x, .w = layer.attn_norm, .out = h } }, &w.wproj, false),
                        .decode = try mm.vector(proj, x, &w.wproj, .{ .norm = layer.attn_norm }),
                    } });
                    try body.append(arena, .{ .launch = try b.add(.delta_prep, P.DeltaPrep{
                        .channels = @intCast(conv_dim),
                        .taps = hp.conv_kernel,
                        .key_channels = 2 * d * hp.gdn_k_heads,
                        .head = d,
                        .row = @intCast(proj_row),
                        .v_heads = hp.gdn_v_heads,
                        .beta_off = @intCast(conv_dim + hp.gdn_inner),
                        .alpha_off = @intCast(conv_dim + hp.gdn_inner + hp.gdn_v_heads),
                        .eps = hp.rms_eps,
                    }, &.{ proj, w.conv, hist, conv, w.dt_bias, w.a, g, beta }, .{ .rows_y = @intCast(conv_dim / d + 1) }) });
                    try body.append(arena, .{ .launch = try b.add(.delta_rule, P.DeltaRule{
                        .k_heads = hp.gdn_k_heads,
                        .v_heads = hp.gdn_v_heads,
                        .dim = d,
                        .conv_stride = @intCast(conv_dim),
                        .taps = hp.conv_kernel,
                        .row = @intCast(proj_row),
                        .kept_hist = @intCast(hist_at / 4),
                    }, &.{ conv, g, beta, state, o, step, proj, hist, kept_layer }, .{ .fixed = .{ hp.gdn_v_heads, std.math.divCeil(u32, d, kernels.delta_cols) catch unreachable, 1 } }) });
                    try body.append(arena, .{ .launch = try b.add(.gated_norm, P.GatedNorm{
                        .v_heads = hp.gdn_v_heads,
                        .dim = d,
                        .eps = hp.rms_eps,
                        .z_row = @intCast(proj_row),
                        .z_off = @intCast(conv_dim),
                    }, &.{ o, proj, w.norm }, .{ .rows_y = hp.gdn_v_heads }) });
                    try body.append(arena, .{ .fork = .{
                        .prefill = try mm.tiled(x, .{ .plain = o }, &w.wout, true),
                        .decode = try mm.vector(x, o, &w.wout, .{ .accumulate = true }),
                    } });
                },
            }
            try lb.ffn(&body, layer);
            try body.append(arena, .{ .layer_end = @intCast(il) });
        }
        try body.append(arena, .{ .final_norm = try b.add(.rmsnorm, norm, &.{ x, m.output_norm, h }, .rows_x) });

        const lm = if (m.output) |*out| out else &m.tok_embd;
        const head: Fork = .{
            .prefill = try mm.tiled(logits, .{ .plain = lm_in }, lm, false),
            .decode = try mm.vector(logits, x, lm, .{ .norm = m.output_norm }),
        };

        var mtp: ?MtpProgram = if (!cfg.mtp) null else if (m.mtp) |*t| blk: {
            if (cfg.logit_rows < 2 or (@as(u64, hp.vocab) * 4) % limits.offset_align != 0) return error.BadConfig;
            const mtp_step = try a.buffer(@sizeOf(P.Step), .{ .uniform = true, .copy_dst = true });
            const mtp_tokens = try a.storage(cap);
            const e1 = try a.storage(cap * e);
            const cat = try a.storage(cap * 2 * e);
            // The block's cache row i holds position i + 1 (position 0 has
            // no hidden state before it): its RoPE table starts at 1.
            const shifted = try kernels.ropeTable(arena, cfg.context + 1, hp.n_rot, hp.rope_base);
            const mtp_rope = try a.storage(rope_table.len);
            try rec.writeBuffer(gpa, mtp_rope.buffer, 0, std.mem.sliceAsBytes(shifted[hp.n_rot / 2 * 2 ..]));

            var pre: std.ArrayList(kernels.Launch) = .empty;
            for (m.tok_embd.chunks) |ch| {
                try pre.append(arena, try b.add(weightKernel(.embed, ch.type), P.EmbedBf16{ .dim = hp.n_embd, .row0 = ch.row0, .chunk_rows = ch.rows }, &.{ mtp_tokens, ch.arg, x }, .rows_x));
            }
            try pre.append(arena, try b.add(.rmsnorm, norm, &.{ x, t.enorm, e1 }, .rows_x));
            try pre.append(arena, try b.add(.rmsnorm, norm, &.{ h, t.hnorm, x }, .rows_x));
            var mbody: std.ArrayList(Item) = .empty;
            try mbody.append(arena, .{ .fork = .{
                .prefill = try mm.tiled(x, .{ .plain = cat }, &t.eh_proj, false),
                .decode = try mm.vector(x, cat, &t.eh_proj, .{}),
            } });
            try lb.attention(&mbody, t.block.attn_norm, &t.block.mixer.attention, mtp_rope, mtp_step);
            try lb.ffn(&mbody, &t.block);
            break :blk .{
                .step = mtp_step,
                .tokens = mtp_tokens,
                .e1 = e1,
                .cat = cat,
                .pre = pre.items,
                .body = mbody.items,
                .head_norm = try b.add(.rmsnorm, norm, &.{ x, t.head_norm, h }, .rows_x),
                .head = .{ .prefill = head.prefill, .decode = try mm.vector(logits, x, lm, .{ .norm = t.head_norm }) },
                .draft_head = try mm.vector(logits, lm_in, lm, .{ .norm = t.head_norm }),
                .draft = try b.add(.draft, P.Draft{ .vocab = hp.vocab, .row = 0, .slot = 1 }, &.{ logits, token_ids }, .{ .fixed = .{ 1, 1, 1 } }),
                .verify_step = try a.buffer(@sizeOf(P.Step), .{ .uniform = true, .copy_dst = true }),
                .token_stage = try arena.alloc(u32, cfg.max_tokens),
            };
        } else return error.BadConfig;

        const mark = try b.addWith(.mark_tokens, sample_params, &.{ prompt_ids, prompt_mask }, .{ .elems = 1 });
        const sampler_args: SamplerArgs = .{ .bias = bias, .prompt_mask = prompt_mask, .counts = counts, .work = work, .state = sample_state, .next = token_ids, .params = sample_params };
        const sample = try sampler_args.launches(&b, hp.vocab, logits, step);
        if (mtp) |*t| {
            const row_bytes = @as(u64, hp.vocab) * 4;
            t.verify_sample = try sampler_args.launches(&b, hp.vocab, .{ .buffer = logits.buffer, .offset = row_bytes, .size = row_bytes }, t.verify_step);
        }
        try b.finish();

        return .{
            .hp = hp,
            .cfg = cfg,
            .prefill_tile = prefillTile(m),
            .mtp = mtp,
            .speculation_pays = speculationPays(m),
            .step = step,
            .token_ids = token_ids,
            .x = x,
            .h = h,
            .lm_in = lm_in,
            .logits = logits,
            .recurrent = recurrent,
            .kept = kept,
            .checkpoints = checkpoints,
            .sample_params = sample_params,
            .bias = bias,
            .prompt_ids = prompt_ids,
            .prompt_mask = prompt_mask,
            .counts = counts,
            .sample_state = sample_state,
            .body = body.items,
            .head = head,
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

    /// Undoes a verify step whose draft was rejected: rolls the DeltaNet
    /// state back to its kept row (the positions after it are dropped;
    /// attention needs nothing, as the next step overwrites their cache
    /// rows) and uncounts its second sample, the sampler's last draw.
    pub fn recordRollback(p: *const Program, gpa: Allocator, rec: *cmd.Recorder) Allocator.Error!void {
        std.debug.assert(p.cfg.mtp);
        try rec.copyBuffer(gpa, p.kept.buffer, 0, p.recurrent.buffer, 0, p.recurrent.size);
        try p.sample.get(.unbump)[0].record(gpa, rec, .{ .rows = 1 });
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
        /// Leave every row's final hidden state in `h` for recordMtp.
        hidden: bool = false,
        /// Keep the DeltaNet state as after this row, for recordRollback
        /// (needs cfg.mtp).
        keep: ?u32 = null,
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
        const e: usize = p.hp.n_embd;
        std.debug.assert(m > 0 and m <= p.cfg.max_tokens and s.pos0 + m <= p.cfg.context);
        std.debug.assert(s.logit_rows <= m and (s.draw == null or s.logit_rows == 1));

        std.debug.assert(s.keep == null or (p.cfg.mtp and s.keep.? < m));
        p.step_value = .{ .m = m, .pos0 = s.pos0, .u = s.draw orelse 0, .keep = if (s.keep) |r| r + 1 else 0 };
        try rec.writeBuffer(gpa, p.step.buffer, 0, std.mem.asBytes(&p.step_value));
        if (s.tokens) |t| {
            @memcpy(p.token_stage[0..t.len], t);
            try rec.writeBuffer(gpa, p.token_ids.buffer, 0, std.mem.sliceAsBytes(p.token_stage[0..t.len]));
        }

        // The decode head projects x itself, with the final norm fused, when
        // it covers every row of the step in one pass; otherwise rows of the
        // final norm are copied to lm_in for the prefill head.
        const fused_head = path == .decode and s.logit_rows == m and m <= p.cfg.logit_rows;
        const want_norm = s.hidden or (s.readback != null and s.readback.?.result_norm != null);
        const ext: kernels.Extent = .{ .rows = m, .keys = s.pos0 + m };
        for (p.body) |item| switch (item) {
            .launch => |l| try l.record(gpa, rec, ext),
            .fork => |f| for (f.pick(path)) |l| try l.record(gpa, rec, ext),
            .final_norm => |l| if (path == .prefill or want_norm or (s.logit_rows > 0 and !fused_head)) try l.record(gpa, rec, ext),
            .layer_end => |il| if (s.readback) |r| if (r.layer_out.len > 0) {
                try rec.readBuffer(gpa, read_token, p.x.buffer, 0, std.mem.sliceAsBytes(r.layer_out[il][0 .. m * e]));
            },
        };
        if (s.readback) |r| if (r.result_norm) |dst| try rec.readBuffer(gpa, read_token, p.h.buffer, 0, std.mem.sliceAsBytes(dst[0 .. m * e]));

        try p.recordHead(gpa, rec, p.head, m, s.logit_rows, fused_head, if (s.readback) |r| r.logits else null);

        if (s.draw != null) {
            std.debug.assert(p.sample_len > 0);
            for (p.sample_steps[0..p.sample_len]) |ss| try p.sample.get(ss.phase)[ss.round].record(gpa, rec, .{ .rows = 1 });
            try rec.readBuffer(gpa, read_token, p.sample_state.buffer, 0, std.mem.asBytes(s.sampled.?));
        }
    }

    /// Projects a step's last `logit_rows` of its m rows to logits, in
    /// passes of at most cfg.logit_rows: `fused` from x with the norm folded
    /// in, else from the final norm's rows in `h`.
    fn recordHead(p: *const Program, gpa: Allocator, rec: *cmd.Recorder, head: Fork, m: u32, logit_rows: u32, fused: bool, readback: ?[]f32) Allocator.Error!void {
        const e: usize = p.hp.n_embd;
        const vocab: usize = p.hp.vocab;
        var r0 = m - logit_rows;
        while (r0 < m) {
            const n = @min(p.cfg.logit_rows, m - r0);
            if (fused) {
                for (head.decode) |l| try l.record(gpa, rec, .{ .rows = n });
            } else {
                try rec.copyBuffer(gpa, p.h.buffer, r0 * e * 4, p.lm_in.buffer, 0, n * e * 4);
                for (head.prefill) |l| try l.record(gpa, rec, .{ .rows = n });
            }
            if (readback) |dst| {
                const at = (r0 - (m - logit_rows)) * vocab;
                try rec.readBuffer(gpa, read_token, p.logits.buffer, 0, std.mem.sliceAsBytes(dst[at..][0 .. n * vocab]));
            }
            r0 += n;
        }
    }

    pub const MtpStep = struct {
        /// Main position of the first row's hidden state: the block's cache
        /// row, at position pos0 + 1.
        pos0: u32,
        /// For each row, the token after its hidden state's position.
        tokens: []const u32,
        logit_rows: u32 = 0,
        path: ?Path = null,
        /// Where the logit rows land, row-major.
        logits: ?[]f32 = null,
        /// Write the last row's draft (its argmax) to token_ids[1] for
        /// recordVerify; its logits overwrite logits row 0 after any logit
        /// rows' readback.
        draft: bool = false,
    };

    /// Runs the MTP block over rows of the previous step's final hidden
    /// states (a step with `hidden`, or the final norm read back): row r
    /// drafts the token two after position pos0 + r.
    pub fn recordMtp(p: *Program, gpa: Allocator, rec: *cmd.Recorder, s: MtpStep) Allocator.Error!void {
        const t = &p.mtp.?;
        const m: u32 = @intCast(s.tokens.len);
        const path = s.path orelse Path.of(m);
        std.debug.assert(m > 0 and m <= p.cfg.max_tokens and s.pos0 + m <= p.cfg.context and s.logit_rows <= m);
        std.debug.assert(path == .prefill or m <= kernels.decode_max_rows);
        const e: u64 = p.hp.n_embd;

        t.step_value = .{ .m = m, .pos0 = s.pos0, .u = 0 };
        try rec.writeBuffer(gpa, t.step.buffer, 0, std.mem.asBytes(&t.step_value));
        @memcpy(t.token_stage[0..m], s.tokens);
        try rec.writeBuffer(gpa, t.tokens.buffer, 0, std.mem.sliceAsBytes(t.token_stage[0..m]));

        const ext: kernels.Extent = .{ .rows = m, .keys = s.pos0 + m };
        for (t.pre) |l| try l.record(gpa, rec, ext);
        // Row r of cat: [enorm(embedding) (e1), hnorm(hidden) (x)].
        for (0..m) |r| {
            try rec.copyBuffer(gpa, t.e1.buffer, r * e * 4, t.cat.buffer, r * 2 * e * 4, e * 4);
            try rec.copyBuffer(gpa, p.x.buffer, r * e * 4, t.cat.buffer, (r * 2 + 1) * e * 4, e * 4);
        }
        for (t.body) |item| switch (item) {
            .launch => |l| try l.record(gpa, rec, ext),
            .fork => |f| for (f.pick(path)) |l| try l.record(gpa, rec, ext),
            .final_norm, .layer_end => unreachable,
        };
        if (s.logit_rows > 0) {
            const fused = path == .decode and s.logit_rows == m and m <= p.cfg.logit_rows;
            if (!fused) try t.head_norm.record(gpa, rec, ext);
            try p.recordHead(gpa, rec, t.head, m, s.logit_rows, fused, s.logits);
        }
        if (s.draft) try p.recordDraft(gpa, rec, m);
    }

    /// The draft from the MTP step's last row: only its logits are needed.
    fn recordDraft(p: *Program, gpa: Allocator, rec: *cmd.Recorder, m: u32) Allocator.Error!void {
        const t = &p.mtp.?;
        const e: u64 = p.hp.n_embd;
        try rec.copyBuffer(gpa, p.x.buffer, (m - 1) * e * 4, p.lm_in.buffer, 0, e * 4);
        for (t.draft_head) |l| try l.record(gpa, rec, .{ .rows = 1 });
        try t.draft.record(gpa, rec, .{ .rows = 1 });
    }

    pub const Verify = struct {
        /// Position of the token; the draft goes at pos0 + 1.
        pos0: u32,
        token: u32,
        /// The draws for the samples after the token and after the draft.
        draws: [2]f32,
        /// Each sample read back as (token, is_stop).
        sampled: *[2][2]u32,
        /// The draft read back.
        draft: *u32,
        /// Each position's logits, read back.
        logits: [2]?[]f32 = .{ null, null },
    };

    /// A speculative step: the token and the draft recordMtp left in
    /// token_ids[1], through the decode kernels at pos0 and pos0 + 1,
    /// keeping the DeltaNet state after the token (recordRollback restores
    /// it when the draft is rejected), with each final hidden state left in
    /// `h` for recordMtp. Samples both positions, in order, so the second
    /// sample sees the first in the sampler's output counts.
    pub fn recordVerify(p: *Program, gpa: Allocator, rec: *cmd.Recorder, v: Verify) Allocator.Error!void {
        const t = &p.mtp.?;
        std.debug.assert(p.sample_len > 0 and v.pos0 + 2 <= p.cfg.context);
        p.step_value = .{ .m = 2, .pos0 = v.pos0, .u = v.draws[0], .keep = 1 };
        try rec.writeBuffer(gpa, p.step.buffer, 0, std.mem.asBytes(&p.step_value));
        t.verify_value = .{ .m = 2, .pos0 = v.pos0, .u = v.draws[1] };
        try rec.writeBuffer(gpa, t.verify_step.buffer, 0, std.mem.asBytes(&t.verify_value));
        p.token_stage[0] = v.token;
        try rec.writeBuffer(gpa, p.token_ids.buffer, 0, std.mem.sliceAsBytes(p.token_stage[0..1]));

        const ext: kernels.Extent = .{ .rows = 2, .keys = v.pos0 + 2 };
        for (p.body) |item| switch (item) {
            .launch, .final_norm => |l| try l.record(gpa, rec, ext),
            .fork => |f| for (f.decode) |l| try l.record(gpa, rec, ext),
            .layer_end => {},
        };
        for (p.head.decode) |l| try l.record(gpa, rec, .{ .rows = 2 });
        const vocab: usize = p.hp.vocab;
        for (v.logits, 0..) |dst, r| if (dst) |row| {
            try rec.readBuffer(gpa, read_token, p.logits.buffer, r * vocab * 4, std.mem.sliceAsBytes(row[0..vocab]));
        };
        try rec.readBuffer(gpa, read_token, p.token_ids.buffer, 4, std.mem.asBytes(v.draft));
        for ([_]*const std.EnumArray(kernels.SamplePhase, [4]kernels.Launch){ &p.sample, &t.verify_sample }, v.sampled) |set, *dst| {
            for (p.sample_steps[0..p.sample_len]) |ss| try set.get(ss.phase)[ss.round].record(gpa, rec, .{ .rows = 1 });
            try rec.readBuffer(gpa, read_token, p.sample_state.buffer, 0, std.mem.asBytes(dst));
        }
    }
};

/// The MTP block's step: its own step uniform, token ids, cache and
/// RoPE table, over the main program's activation buffers.
pub const MtpProgram = struct {
    step: Arg,
    tokens: Arg,
    e1: Arg,
    cat: Arg,
    /// The embedding, then enorm into e1 and hnorm of `h` into x.
    pre: []const kernels.Launch,
    /// eh_proj from cat into x, then the attention block.
    body: []const Item,
    head_norm: kernels.Launch,
    head: Fork,
    /// The head over one row of `lm_in` (the step's last, copied from x),
    /// with the head norm, into logits row 0.
    draft_head: []const kernels.Launch,
    /// The argmax of logits row 0 into token_ids[1], where the verify step
    /// embeds the draft.
    draft: kernels.Launch,
    /// The verify step's second draw (`u`; the rest unused).
    verify_step: Arg,
    /// The sampler over logits row 1, with the second draw.
    verify_sample: std.EnumArray(kernels.SamplePhase, [4]kernels.Launch) = undefined,
    step_value: P.Step = .{ .m = 0, .pos0 = 0, .u = 0 },
    verify_value: P.Step = .{ .m = 0, .pos0 = 0, .u = 0 },
    token_stage: []u32,
};

/// Builds a decoder layer's launches over the step's activation buffers.
const LayerBuilder = struct {
    a: *Alloc,
    mm: Mm,
    hp: Hparams,
    kv_layer: u64,
    kv: KvFormat,
    max_blocks: u32,
    x: Arg,
    h: Arg,
    gu: Arg,
    act: Arg,
    qkv: Arg,
    att: Arg,
    attn_part: Arg,

    /// Gated full attention with a new KV cache, positioned by `rope` and
    /// `step`.
    fn attention(lb: LayerBuilder, body: *std.ArrayList(Item), attn_norm: Arg, w: *const Attention, rope: Arg, step: Arg) Program.BuildError!void {
        const hp = lb.hp;
        const mm = lb.mm;
        const b = mm.b;
        const arena = mm.arena;
        const hd: u64 = hp.head_dim;
        const kv_dim: u64 = hd * hp.n_head_kv;
        const qkv_row: u64 = 2 * hd * hp.n_head + 2 * kv_dim;
        const kc = try lb.a.buffer(lb.kv_layer, .{ .storage = true });
        const vc = try lb.a.buffer(lb.kv_layer, .{ .storage = true });
        const half: u32 = @backingInt(lb.kv);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const q_row: u32 = @intCast(qkv_row);
        try body.append(arena, .{ .fork = .{
            .prefill = try mm.tiled(lb.qkv, .{ .rmsnorm = .{ .x = lb.x, .w = attn_norm, .out = lb.h } }, &w.wqkv, false),
            .decode = try mm.vector(lb.qkv, lb.x, &w.wqkv, .{ .norm = attn_norm }),
        } });
        try body.append(arena, .{ .launch = try b.add(.rope_append, P.RopeAppend{
            .heads = hp.n_head,
            .kv_heads = hp.n_head_kv,
            .head_dim = hp.head_dim,
            .n_rot = hp.n_rot,
            .row = q_row,
            .k_off = @intCast(2 * hd * hp.n_head),
            .v_off = @intCast(2 * hd * hp.n_head + kv_dim),
            .f16 = half,
            .eps = hp.rms_eps,
        }, &.{ lb.qkv, w.q_norm, w.k_norm, rope, kc, vc, step }, .{ .rows_y = hp.n_head + hp.n_head_kv }) });
        // Split attention over blocks of cache positions, in chunks of
        // attn_rows tokens; a step records the chunks its rows reach.
        var attn: std.ArrayList(kernels.Launch) = .empty;
        var row0: u32 = 0;
        while (row0 < mm.cap) : (row0 += kernels.attn_rows) {
            const params: P.Attention = .{ .heads = hp.n_head, .kv_heads = hp.n_head_kv, .head_dim = hp.head_dim, .f16 = half, .scale = scale, .max_blocks = lb.max_blocks, .q_row = q_row, .row0 = row0 };
            try attn.append(arena, try b.add(.attn_split, params, &.{ lb.qkv, kc, vc, lb.attn_part, step }, .{ .key_blocks = .{ .x = hp.n_head_kv, .block = kernels.attn_block, .row0 = row0, .max_rows = kernels.attn_rows } }));
            try attn.append(arena, try b.add(.attn_combine, params, &.{ lb.qkv, lb.attn_part, lb.att, step }, .{ .rows_y_from = .{ .x = hp.n_head, .row0 = row0, .max_rows = kernels.attn_rows } }));
        }
        try body.append(arena, .{ .fork = .{
            .prefill = try concat(arena, &.{ attn.items, try mm.tiled(lb.x, .{ .plain = lb.att }, &w.wo, true) }),
            .decode = try concat(arena, &.{ attn.items[0..2], try mm.vector(lb.x, lb.att, &w.wo, .{ .accumulate = true }) }),
        } });
    }

    /// The SiLU feed-forward block, added to x.
    fn ffn(lb: LayerBuilder, body: *std.ArrayList(Item), layer: *const Layer) Program.BuildError!void {
        const mm = lb.mm;
        const ff = lb.hp.n_ff;
        try body.append(mm.arena, .{ .fork = .{
            .prefill = try concat(mm.arena, &.{
                try mm.tiled(lb.gu, .{ .rmsnorm = .{ .x = lb.x, .w = layer.post_norm, .out = lb.h } }, &layer.ffn_gu, false),
                try mm.tiled(lb.x, .{ .silu = .{ .gu = lb.gu, .width = ff, .out = lb.act } }, &layer.ffn_down, true),
            }),
            .decode = try concat(mm.arena, &.{
                try mm.vector(lb.act, lb.x, &layer.ffn_gu, .{ .norm = layer.post_norm, .silu_pair = ff }),
                try mm.vector(lb.x, lb.act, &layer.ffn_down, .{ .accumulate = true }),
            }),
        } });
    }
};

/// The largest token tile among the model's prefill multiplies.
fn prefillTile(m: *const Model) u32 {
    const Tile = struct {
        tokens: u32 = 1,
        fn add(s: *@This(), w: *const Matrix) void {
            for (w.chunks) |ch| s.tokens = @max(s.tokens, kernels.matmulTiles(ch.type)[1]);
        }
    };
    var tile: Tile = .{};
    m.eachMatrix(&tile, Tile.add);
    return tile.tokens;
}
