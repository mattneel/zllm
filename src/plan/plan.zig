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

/// A model's weights on the GPU.
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
        return .{ .qwen35 = try qwen35.Model.upload(arena, gpa, rec, set, file, limits) };
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
    tuning: common.Tuning = .{},
    /// Qwen 3.5: build the MTP block's steps (qwen35.Config.mtp).
    mtp: bool = false,
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
    /// Qwen 3.5's kernel path (qwen35.Program.Step.path).
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

    pub fn build(arena: Allocator, gpa: Allocator, rec: *cmd.Recorder, set: *kernels.Set, m: *const Model, cfg: Config, limits: common.Limits) BuildError!Program {
        return switch (m.*) {
            .qwen35 => |*w| .{ .qwen35 = try .build(arena, gpa, rec, set, w, .{
                .context = cfg.context,
                .max_tokens = @min(512, cfg.context),
                .kv = cfg.kv,
                .mtp = cfg.mtp,
                .logit_rows = 2,
                .tuning = cfg.tuning,
            }, limits) },
            .qwen4exp => |*w| .{ .qwen4exp = try .build(arena, gpa, rec, set, w, .{
                .context = @min(cfg.context, qwen4exp.max_context),
                .kv = cfg.kv,
                .tuning = cfg.tuning,
            }, limits) },
        };
    }

    pub fn architecture(p: *const Program) []const u8 {
        return switch (p.*) {
            .qwen35 => "qwen35",
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
            .qwen35 => |*x| x.cfg.max_tokens,
            .qwen4exp => kernels.decode_max_rows,
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
            .qwen35 => |*x| x.prefill_tile,
            .qwen4exp => 1,
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
