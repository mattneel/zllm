//! Execution plans: weight layout, activation buffers and recorded steps,
//! and the cache of what tuning found for them. `Model` and `Program` pick
//! the architecture's own from the GGUF metadata; the engine drives either
//! through `Program`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const cmd = @import("../gpu/cmd.zig");
const kernels = @import("../kernels/kernels.zig");
const sampler = @import("../sampler/sampler.zig");
const graph = @import("../graph/qwen4exp.zig");

pub const common = @import("common.zig");
pub const qwen35 = @import("qwen35.zig");
pub const qwen4exp = @import("qwen4exp.zig");
pub const cache = @import("cache.zig");

/// A model's weights on the GPU. `qwen35` also holds the dense
/// architectures (graph/dense.zig), as models of attention layers only.
pub const Model = union(enum) {
    qwen35: qwen35.Model,
    qwen4exp: qwen4exp.Model,

    pub const UploadError = qwen35.Model.UploadError || qwen4exp.Model.UploadError;

    /// Records the upload of the weights of the architecture `file` names.
    pub fn upload(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *const kernels.Set, file: gguf.File, limits: common.Limits) UploadError!Model {
        const arch = file.get("general.architecture");
        if (arch != null and arch.? == .string and std.mem.eql(u8, arch.?.string, graph.arch)) {
            return .{ .qwen4exp = try qwen4exp.Model.upload(arena, gpa, rec, set, file, limits) };
        }
        return .{ .qwen35 = try qwen35.Model.uploadAny(arena, gpa, rec, set, file, limits) };
    }

    /// Calls f(context, w) for every multiply's weights (tuning).
    pub fn eachMatrix(m: *const Model, context: anytype, comptime f: fn (@TypeOf(context), *const common.Matrix) void) void {
        switch (m.*) {
            inline else => |*w| w.eachMatrix(context, f),
        }
    }

    /// Whether a program for this model can be built with its MTP block
    /// (qwen35.canSpeculate).
    pub fn canSpeculate(m: *const Model, limits: common.Limits) bool {
        return switch (m.*) {
            .qwen35 => |*w| qwen35.canSpeculate(w, limits),
            .qwen4exp => false,
        };
    }

    /// Whether speculating with the MTP block speeds decoding up
    /// (qwen35.speculationPays).
    pub fn speculationPays(m: *const Model) bool {
        return switch (m.*) {
            .qwen35 => |*w| qwen35.speculationPays(w),
            .qwen4exp => false,
        };
    }
};

/// What a program is built for; each architecture takes what applies to it.
pub const Config = struct {
    /// Positions the KV cache holds; Qwen4Exp caps it at its max_context.
    context: u32,
    kv: common.KvFormat = .f16,
    /// Readback rows reserved for calibration; ordinary generation needs two.
    logit_rows: u32 = 2,
    tuning: common.Tuning = .{},
    /// Qwen 3.5: build the MTP block's steps (qwen35.Config.mtp).
    mtp: bool = false,
    /// The fusion pass (fuse.lower); off for the unfused reference.
    fuse: bool = true,
};

/// A step as the engine records it (each program's own Step has more for
/// checks).
pub const Step = struct {
    pos0: u32,
    /// Token ids to process; null: the one token the previous step sampled,
    /// for programs that take it on the GPU (`chainsSamples`).
    tokens: ?[]const u32,
    /// Project this many of the step's last rows to logits.
    logit_rows: u32 = 0,
    /// Sample from the last row with this draw, reading back (token,
    /// is_stop) into `sampled`.
    draw: ?f32 = null,
    sampled: ?*[2]u32 = null,
    /// The logit rows, read back here.
    logits: ?[]f32 = null,
    /// Kernel path, shared by Qwen 3.5 and Qwen4Exp.
    path: ?qwen35.Path = null,
    /// Qwen 3.5: leave the final hidden states for its MTP block.
    hidden: bool = false,
};

/// A step program, whichever architecture. Must not move once the engine
/// points at it.
pub const Program = union(enum) {
    qwen35: qwen35.Program,
    qwen4exp: qwen4exp.Program,

    pub const BuildError = common.BuildError;
    pub const StepError = Allocator.Error || error{InvalidToken};
    pub const MemoryError = BuildError || cmd.DecodeError || error{Overflow};
    /// Conservative D3D12 placement granularity, applied to each buffer.
    pub const memory_alignment: u64 = 64 * 1024;

    /// Resident bytes of every buffer `build` creates, excluding the model's
    /// already-uploaded weights and backend pipeline/command metadata.
    /// Uses the real allocation path, including KV, recurrent checkpoints,
    /// sampler, scratch and packed uniform pages; no GPU work is executed.
    /// The temporary CPU plan and any lazily selected kernel variants are
    /// discarded, leaving the caller's kernel set unchanged.
    pub fn memoryBytes(gpa: Allocator, set: *const kernels.Set, m: *const Model, cfg: Config, limits: common.Limits) MemoryError!u64 {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var rec: cmd.Recorder = .{};
        defer rec.deinit(gpa);
        var planning_set = set.*;
        _ = try build(arena.allocator(), gpa, &rec, &planning_set, m, cfg, limits);
        return cmd.bufferBytes(gpa, rec.batch(), memory_alignment);
    }

    pub fn build(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *kernels.Set, m: *const Model, cfg: Config, limits: common.Limits) BuildError!Program {
        return switch (m.*) {
            .qwen35 => |*w| .{ .qwen35 = try .build(arena, gpa, rec, set, w, .{
                .context = cfg.context,
                .max_tokens = @min(512, cfg.context),
                .kv = cfg.kv,
                .mtp = cfg.mtp,
                .logit_rows = cfg.logit_rows,
                .tuning = cfg.tuning,
            }, limits) },
            .qwen4exp => |*w| .{ .qwen4exp = try .build(arena, gpa, rec, set, w, .{
                .context = @min(cfg.context, qwen4exp.max_context),
                .max_tokens = @min(512, cfg.context, qwen4exp.max_context),
                .logit_rows = cfg.logit_rows,
                .kv = cfg.kv,
                .tuning = cfg.tuning,
                .fuse = cfg.fuse,
            }, limits) },
        };
    }

    pub fn architecture(p: *const Program) []const u8 {
        return switch (p.*) {
            .qwen35 => |*x| x.arch,
            .qwen4exp => graph.arch,
        };
    }

    pub fn vocab(p: *const Program) u32 {
        return switch (p.*) {
            inline else => |*x| x.hp.vocab,
        };
    }

    pub fn layers(p: *const Program) u32 {
        return switch (p.*) {
            inline else => |*x| x.hp.n_layer,
        };
    }

    pub fn embedding(p: *const Program) u32 {
        return switch (p.*) {
            inline else => |*x| x.hp.n_embd,
        };
    }

    pub fn context(p: *const Program) u32 {
        return switch (p.*) {
            inline else => |*x| x.cfg.context,
        };
    }

    pub fn kv(p: *const Program) common.KvFormat {
        return switch (p.*) {
            inline else => |*x| x.cfg.kv,
        };
    }

    /// Most tokens one step processes.
    pub fn maxTokens(p: *const Program) u32 {
        return switch (p.*) {
            inline else => |*x| x.cfg.max_tokens,
        };
    }

    pub fn checkpoints(p: *const Program) u32 {
        return switch (p.*) {
            inline else => |*x| x.cfg.checkpoints,
        };
    }

    /// Tokens of a prefill tile, which costs as much part-filled.
    pub fn prefillTile(p: *const Program) u32 {
        return switch (p.*) {
            inline else => |*x| x.prefill_tile,
        };
    }

    /// Whether the program has the MTP block's steps (speculation).
    pub fn mtp(p: *const Program) bool {
        return switch (p.*) {
            .qwen35 => |*x| x.mtp != null,
            .qwen4exp => false,
        };
    }

    pub fn speculationPays(p: *const Program) bool {
        return switch (p.*) {
            .qwen35 => |*x| x.speculation_pays,
            .qwen4exp => false,
        };
    }

    /// Whether a decode step can take its token from the previous step's
    /// sample on the GPU, unseen by the host (pipelined decoding). Qwen4Exp
    /// cannot: the host picks each token's PLE rows.
    pub fn chainsSamples(p: *const Program) bool {
        return p.* == .qwen35;
    }

    pub fn recordReset(p: *Program, gpa: Allocator, rec: *cmd.Recorder) Allocator.Error!void {
        return switch (p.*) {
            inline else => |*x| x.recordReset(gpa, rec),
        };
    }

    pub fn recordSave(p: *Program, gpa: Allocator, rec: *cmd.Recorder, slot: u32) Allocator.Error!void {
        return switch (p.*) {
            inline else => |*x| x.recordSave(gpa, rec, slot),
        };
    }

    pub fn recordRestore(p: *Program, gpa: Allocator, rec: *cmd.Recorder, slot: u32) Allocator.Error!void {
        return switch (p.*) {
            inline else => |*x| x.recordRestore(gpa, rec, slot),
        };
    }

    pub fn recordRequest(p: *Program, gpa: Allocator, rec: *cmd.Recorder, params: sampler.Params, prompt: []const u32, stops: []const u32) Allocator.Error!void {
        return switch (p.*) {
            inline else => |*x| x.recordRequest(gpa, rec, params, prompt, stops),
        };
    }

    pub fn recordStep(p: *Program, gpa: Allocator, rec: *cmd.Recorder, s: Step) StepError!void {
        switch (p.*) {
            .qwen35 => |*x| try x.recordStep(gpa, rec, .{
                .pos0 = s.pos0,
                .tokens = s.tokens,
                .logit_rows = s.logit_rows,
                .draw = s.draw,
                .sampled = s.sampled,
                .readback = if (s.logits) |l| .{ .logits = l } else null,
                .path = s.path,
                .hidden = s.hidden,
            }),
            .qwen4exp => |*x| try x.recordStep(gpa, rec, .{
                .pos0 = s.pos0,
                .tokens = s.tokens.?,
                .logit_rows = s.logit_rows,
                .draw = s.draw,
                .sampled = s.sampled,
                .logits = s.logits,
                .path = s.path,
            }),
        }
    }

    // Speculation: only programs with the MTP block (`mtp`).

    pub fn recordRollback(p: *Program, gpa: Allocator, rec: *cmd.Recorder) Allocator.Error!void {
        return p.qwen35.recordRollback(gpa, rec);
    }

    pub fn recordMtp(p: *Program, gpa: Allocator, rec: *cmd.Recorder, s: qwen35.Program.MtpStep) Allocator.Error!void {
        return p.qwen35.recordMtp(gpa, rec, s);
    }

    pub fn recordVerify(p: *Program, gpa: Allocator, rec: *cmd.Recorder, s: qwen35.Program.Verify) Allocator.Error!void {
        return p.qwen35.recordVerify(gpa, rec, s);
    }
};

test "program capacity and tile expose each architecture's built values" {
    var p: Program = .{ .qwen4exp = undefined };
    p.qwen4exp.cfg = .{ .context = 8192, .max_tokens = 129 };
    p.qwen4exp.prefill_tile = 64;
    try std.testing.expectEqual(@as(u32, 129), p.maxTokens());
    try std.testing.expectEqual(@as(u32, 64), p.prefillTile());
    p = .{ .qwen35 = undefined };
    p.qwen35.cfg = .{ .context = 8192, .max_tokens = 257 };
    p.qwen35.prefill_tile = 32;
    try std.testing.expectEqual(@as(u32, 257), p.maxTokens());
    try std.testing.expectEqual(@as(u32, 32), p.prefillTile());
}

test "memory planning matches real allocations for dense hybrid and MTP layouts" {
    const gpa = std.testing.allocator;
    const arg: kernels.Arg = .{ .buffer = @fromBackingInt(10000), .size = 256 * 256 * 2 };
    const matrix: common.Matrix = .{
        .n = 256,
        .k = 256,
        .chunks = &.{.{ .arg = arg, .row0 = 0, .rows = 256, .type = .bf16 }},
    };
    const gate_up: common.Matrix = .{
        .n = 512,
        .k = 256,
        .chunks = &.{.{ .arg = .{ .buffer = @fromBackingInt(10001), .size = 512 * 256 * 2 }, .row0 = 0, .rows = 512, .type = .bf16 }},
    };
    var m: Model = .{ .qwen35 = .{
        .hp = .{
            .vocab = 256,
            .n_embd = 256,
            .n_ff = 256,
            .n_layer = 1,
            .n_layer_nextn = 0,
            .rms_eps = 1e-6,
            .n_head = 1,
            .n_head_kv = 1,
            .head_dim = 32,
            .n_rot = 32,
            .rope_base = 10000,
            .rope_sections = @splat(0),
            .conv_kernel = 1,
            .gdn_head_dim = 0,
            .gdn_k_heads = 0,
            .gdn_v_heads = 0,
            .gdn_inner = 0,
            .layer_kinds = @splat(.attention),
        },
        .attn = .{ .gated = false, .qk_norm = false },
        .tok_embd = matrix,
        .output = matrix,
        .output_norm = arg,
        .layers = &.{.{
            .attn_norm = arg,
            .post_norm = arg,
            .ffn_gu = gate_up,
            .ffn_down = matrix,
            .mixer = .{ .attention = .{ .wqkv = matrix, .wo = matrix, .q_norm = arg, .k_norm = arg } },
        }},
        .mtp = null,
    } };
    var set: kernels.Set = .{ .pipelines = undefined, .backend = .webgpu };
    for (std.enums.values(kernels.Kernel)) |k| set.pipelines.set(k, @fromBackingInt(0));
    const limits: common.Limits = .{ .max_binding = 1 << 30, .offset_align = 256 };
    var cfg: Config = .{ .context = 4096, .kv = .f32 };
    const attention = m.qwen35.layers[0];
    var hybrid_layers = [_]@TypeOf(attention){ attention, attention };
    var previous_bytes: u64 = 0;
    for (0..3) |layout| {
        if (layout == 1) {
            m.qwen35.hp.n_layer = 2;
            m.qwen35.hp.layer_kinds[1] = .delta_net;
            m.qwen35.hp.conv_kernel = 4;
            m.qwen35.hp.gdn_head_dim = 32;
            m.qwen35.hp.gdn_k_heads = 1;
            m.qwen35.hp.gdn_v_heads = 1;
            m.qwen35.hp.gdn_inner = 32;
            hybrid_layers[1].mixer = .{ .delta_net = .{ .wproj = matrix, .wout = matrix, .conv = arg, .dt_bias = arg, .a = arg, .norm = arg } };
            m.qwen35.layers = &hybrid_layers;
        } else if (layout == 2) {
            m.qwen35.mtp = .{ .enorm = arg, .hnorm = arg, .eh_proj = matrix, .block = attention, .head_norm = arg };
            cfg.mtp = true;
        }
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        var rec: cmd.Recorder = .{};
        defer rec.deinit(gpa);
        _ = try Program.build(arena.allocator(), gpa, &rec, &set, &m, cfg, limits);
        const bytes = try Program.memoryBytes(gpa, &set, &m, cfg, limits);
        try std.testing.expectEqual(try cmd.bufferBytes(gpa, rec.batch(), Program.memory_alignment), bytes);
        try std.testing.expect(bytes > previous_bytes);
        previous_bytes = bytes;
        const cache_count: u64 = if (cfg.mtp) 4 else 2;
        var smaller = cfg;
        smaller.kv = .f16;
        const half_bytes = try Program.memoryBytes(gpa, &set, &m, smaller, limits);
        try std.testing.expectEqual(cache_count * 262144, bytes - half_bytes);
        smaller.kv = .fp8;
        const fp8_bytes = try Program.memoryBytes(gpa, &set, &m, smaller, limits);
        try std.testing.expectEqual(cache_count * 65536, half_bytes - fp8_bytes);
        try std.testing.expectEqual(@as(usize, 0), set.n_variants);
    }
    try std.testing.expectError(error.BadConfig, Program.memoryBytes(gpa, &set, &m, .{ .context = 0 }, limits));
}
