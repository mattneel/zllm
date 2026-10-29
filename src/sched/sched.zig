//! The engine: turns generation requests into GPU submissions, one step
//! each, and their results into tokens and text. It never blocks; the host
//! drives it:
//!
//!     try engine.start(request);
//!     while (true) switch (try engine.next()) {
//!         .submit => |words| { execute words; try engine.complete(elapsed_ns); },
//!         .token => |t| show t.text,
//!         .done => |d| break,
//!     }
//!
//! Prefill chunks are the largest token count whose predicted cost fits the
//! submission budget. The prediction is linear, fixed + per_token · m: the
//! fixed part is measured on one-token steps (decode, and each prompt's last
//! token), the per-token part on longer prefill steps. They also end
//! at checkpoint positions: every `state.checkpoint_interval` tokens and just
//! before the prompt's last token, which then runs as its own step and is
//! sampled. That checkpoint lets a later request with the same prompt, or one
//! that only changes what follows it, restart one token before the end.
//! Decode is one token per submission, and only the sampled token id (and
//! its stop flag) comes back.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cmd = @import("../gpu/cmd.zig");
const plan = @import("../plan/qwen35.zig");
const sampler = @import("../sampler/sampler.zig");
const state = @import("../state/state.zig");
const Tokenizer = @import("../tokenizer/tokenizer.zig").Tokenizer;
const Stream = @import("../tokenizer/stream.zig").Stream;

/// Target GPU time per submission (spec §8).
pub const budget_ns = 150 * std.time.ns_per_ms;
/// First prefill chunk, before any step has been measured.
pub const first_chunk = 64;

pub const Request = struct {
    prompt: []const u32,
    max_tokens: u32,
    sampling: sampler.Params = .{},
    /// Tokens that end generation, checked on the GPU (not part of the text).
    stop_tokens: []const u32 = &.{},
    /// Strings that end generation, checked on the CPU (not part of the text).
    stop: []const []const u8 = &.{},
    /// For checks: each sampled step's logits (row i at i · vocab), at least
    /// max_tokens · vocab values.
    logits: ?[]f32 = null,
};

pub const Finish = enum {
    /// A stop token or stop string.
    stop,
    /// max_tokens, or the context is full.
    length,
};

pub const Usage = struct {
    prompt_tokens: u32 = 0,
    /// Prompt tokens served from the resident sequence instead of prefill.
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    prefill_steps: u32 = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
};

pub const Event = union(enum) {
    /// Execute this batch, then call `complete`. Valid until then.
    submit: []const u32,
    /// A sampled token and the text it released (valid until the next call).
    token: struct { id: u32, text: []const u8 },
    done: struct { finish: Finish, usage: Usage },
};

pub const Error = Allocator.Error || Tokenizer.DecodeError || sampler.Params.Error || error{ Busy, EmptyPrompt, PromptTooLong, TooManyStops, NotSubmitted };

pub const Engine = struct {
    gpa: Allocator,
    program: *plan.Program,
    tok: *const Tokenizer,
    seq: state.Sequence,
    rec: cmd.Recorder = .{},
    phase: Phase = .idle,
    /// Measured cost of a one-token step (the fixed part), 0 until seen.
    step_ns: u64 = 0,
    /// Measured cost per additional prefill token, 0 until seen.
    token_ns: u64 = 0,
    submits: u32 = 0,

    // Per request.
    req: Request = undefined,
    draws: sampler.Draws = undefined,
    stream: Stream = .{ .stops = &.{} },
    text: std.ArrayList(u8) = .empty,
    usage: Usage = .{},
    /// Prompt position of the next prefill chunk.
    prefill_pos: u32 = 0,
    /// Readback of the last sampling step: (token, is_stop).
    sampled: [2]u32 = .{ 0, 0 },
    pending: ?Pending = null,
    finish: ?Finish = null,

    const Phase = enum { idle, prefill, decode, emit, done };
    const Pending = struct { tokens: u32, samples: bool, prefill: bool };

    /// The engine keeps pointers to itself in recorded batches: it must not
    /// move while a batch is pending.
    pub fn init(gpa: Allocator, program: *plan.Program, tok: *const Tokenizer) Allocator.Error!Engine {
        return .{ .gpa = gpa, .program = program, .tok = tok, .seq = try .init(gpa, program.cfg.checkpoints) };
    }

    pub fn deinit(e: *Engine) void {
        e.seq.deinit(e.gpa);
        e.rec.deinit(e.gpa);
        e.stream.deinit(e.gpa);
        e.text.deinit(e.gpa);
        e.* = undefined;
    }

    /// The resident sequence: every token whose state the GPU holds.
    pub fn resident(e: *const Engine) []const u32 {
        return e.seq.tokens.items;
    }

    /// `req` and the memory it points to must stay valid until `.done`.
    pub fn start(e: *Engine, req: Request) Error!void {
        if (e.phase != .idle and e.phase != .done) return error.Busy;
        const cfg = e.program.cfg;
        if (req.prompt.len == 0) return error.EmptyPrompt;
        if (req.prompt.len > cfg.context) return error.PromptTooLong;
        if (req.stop_tokens.len > @import("../kernels/kernels.zig").max_stops) return error.TooManyStops;
        try req.sampling.validate(e.program.hp.vocab);

        e.rec.clear();
        const r = e.seq.plan(req.prompt);
        switch (r.from) {
            .current => {},
            .checkpoint => |slot| try e.program.recordRestore(e.gpa, &e.rec, slot),
            .reset => try e.program.recordReset(e.gpa, &e.rec),
        }
        e.seq.rewind(r);
        try e.program.recordRequest(e.gpa, &e.rec, req.sampling, req.prompt, req.stop_tokens);

        e.req = req;
        e.draws = .init(req.sampling.seed);
        e.stream.deinit(e.gpa);
        e.stream = .{ .stops = req.stop };
        e.usage = .{ .prompt_tokens = @intCast(req.prompt.len), .cached_tokens = r.start };
        e.prefill_pos = r.start;
        e.pending = null;
        e.finish = null;
        e.phase = .prefill;
    }

    pub fn next(e: *Engine) Error!Event {
        if (e.pending != null) return error.NotSubmitted;
        switch (e.phase) {
            .idle => return error.NotSubmitted,
            .prefill => {
                try e.recordPrefill();
                return .{ .submit = e.rec.batch() };
            },
            .decode => {
                try e.recordDecode();
                return .{ .submit = e.rec.batch() };
            },
            .emit => {
                e.phase = if (e.finish == null) .decode else .done;
                return .{ .token = .{ .id = e.sampled[0], .text = e.text.items } };
            },
            .done => {
                e.phase = .idle;
                return .{ .done = .{ .finish = e.finish.?, .usage = e.usage } };
            },
        }
    }

    /// Reports that the last submitted batch finished, `elapsed_ns` after it
    /// was submitted.
    pub fn complete(e: *Engine, elapsed_ns: u64) Error!void {
        const p = e.pending orelse return error.NotSubmitted;
        e.pending = null;
        e.rec.clear();
        if (p.prefill) {
            e.usage.prefill_ns += elapsed_ns;
            e.usage.prefill_steps += 1;
        } else {
            e.usage.decode_ns += elapsed_ns;
        }
        if (p.tokens == 1) {
            e.step_ns = ema(e.step_ns, elapsed_ns);
        } else {
            e.token_ns = ema(e.token_ns, @max(1, (elapsed_ns -| e.step_ns) / p.tokens));
        }
        if (!p.samples) return;

        const token = e.sampled[0];
        e.usage.completion_tokens += 1;
        e.text.clearRetainingCapacity();
        if (e.sampled[1] != 0) {
            try e.stream.flush(e.gpa, &e.text);
            e.finish = .stop;
        } else if (try e.stream.push(e.gpa, e.tok, token, &e.text)) {
            e.finish = .stop;
        } else if (e.usage.completion_tokens == e.req.max_tokens or e.seq.len() == e.program.cfg.context) {
            try e.stream.flush(e.gpa, &e.text);
            e.finish = .length;
        }
        e.phase = .emit;
    }

    /// Tokens for the next prefill step: as many as the budget allows, up to
    /// the next checkpoint position and the end of the prompt.
    fn chunkSize(e: *const Engine) u32 {
        const pos = e.prefill_pos;
        const left: u32 = @intCast(e.req.prompt.len - pos);
        var m: u32 = first_chunk;
        if (e.token_ns != 0) {
            const fixed: u64 = @min(e.step_ns, budget_ns / 2);
            m = @intCast(@max(1, @min(std.math.maxInt(u32), (budget_ns - fixed) / e.token_ns)));
        }
        m = @min(m, e.program.cfg.max_tokens, left);
        const boundary = (pos / state.checkpoint_interval + 1) * state.checkpoint_interval;
        m = @min(m, boundary - pos);
        const before_last: u32 = @intCast(e.req.prompt.len - 1);
        return if (pos < before_last) @min(m, before_last - pos) else m;
    }

    fn recordPrefill(e: *Engine) Error!void {
        const pos0 = e.prefill_pos;
        const m = e.chunkSize();
        const tokens = e.req.prompt[pos0..][0..m];
        const last = pos0 + m == e.req.prompt.len;
        try e.program.recordStep(e.gpa, &e.rec, .{
            .pos0 = pos0,
            .tokens = tokens,
            .logit_rows = @intFromBool(last),
            .draw = if (last) e.draws.next() else null,
            .sampled = if (last) &e.sampled else null,
            .readback = if (last) e.logitsReadback() else null,
        });
        try e.seq.append(e.gpa, tokens);
        e.prefill_pos += m;
        if (e.seq.len() + 1 == e.req.prompt.len or e.seq.len() % state.checkpoint_interval == 0) try e.save();
        try e.submitBatch(.{ .tokens = m, .samples = last, .prefill = true });
        if (last) e.phase = .decode;
    }

    /// Processes the token just sampled and samples the next.
    fn recordDecode(e: *Engine) Error!void {
        const pos0 = e.seq.len();
        try e.program.recordStep(e.gpa, &e.rec, .{
            .pos0 = pos0,
            .tokens = null,
            .logit_rows = 1,
            .draw = e.draws.next(),
            .sampled = &e.sampled,
            .readback = e.logitsReadback(),
        });
        try e.seq.append(e.gpa, &.{e.sampled[0]});
        if (e.seq.len() % state.checkpoint_interval == 0) try e.save();
        try e.submitBatch(.{ .tokens = 1, .samples = true, .prefill = false });
    }

    fn logitsReadback(e: *const Engine) ?plan.Program.Readback {
        const all = e.req.logits orelse return null;
        const vocab = e.program.hp.vocab;
        return .{ .logits = all[e.usage.completion_tokens * vocab ..][0..vocab] };
    }

    /// Drops the resident sequence and its checkpoints: the next request
    /// starts from a zeroed state.
    pub fn forget(e: *Engine) void {
        e.seq.rewind(.{ .from = .reset, .start = 0 });
    }

    /// Smooths measurements: the mean of the previous estimate and the new one.
    fn ema(old: u64, new: u64) u64 {
        return if (old == 0) new else (old + new) / 2;
    }

    fn save(e: *Engine) Allocator.Error!void {
        if (e.seq.checkpointSlot()) |slot| try e.program.recordSave(e.gpa, &e.rec, slot);
    }

    fn submitBatch(e: *Engine, p: Pending) Allocator.Error!void {
        e.submits += 1;
        try e.rec.submit(e.gpa, e.submits);
        e.pending = p;
    }
};
