//! The engine: turns generation requests into GPU submissions, one step
//! each, and their results into tokens and text. It never blocks; the host
//! drives it:
//!
//!     while (try engine.calibration()) |words| { execute words; engine.calibrated(ns); }
//!     try engine.start(request);
//!     while (true) switch (try engine.next()) {
//!         .submit => |words| { execute words; try engine.complete(elapsed_ns); },
//!         .token => |t| show t.text,
//!         .wait => unreachable, // only when batches are still outstanding
//!         .done => |d| break,
//!     }
//!
//! A host that executes asynchronously can keep two batches in flight: on
//! `.submit` it starts the batch and calls `next` again without completing
//! it; on `.wait` it completes the oldest outstanding batch. Decoding then
//! records the next step before the previous step's token is read back:
//! the sampler feeds each token to the next step on the GPU, so only the
//! host-side bookkeeping waits. This hides readback latency, which is about
//! 3 ms in Chrome. When a step finishes the request (a stop or max_tokens),
//! the step recorded after it has already processed its token; that token
//! stays resident, and the later step's sample is discarded.
//!
//! `cancel` ends a request between batches: the batch in flight completes
//! (and a token it sampled is still emitted), then `.done` reports `abort`.
//! Every token a completed step processed stays resident, so the state is as
//! consistent as after any other finish. Text the stream was holding back
//! (an incomplete UTF-8 sequence or a possible stop string) is dropped.
//!
//! Prefill chunks are the largest token count whose predicted cost fits the
//! submission budget. The prediction is linear, fixed + per_token · m: the
//! fixed part is measured on one-token steps (calibration before the first
//! request, decode, and each prompt's last token), the per-token part on
//! longer prefill steps, counting their tokens in whole tiles of the prefill
//! multiplies (a partial tile costs as much as a whole one). Where even one
//! tile overshoots the budget (a large model), a chunk is one tile as long
//! as its prediction stays under the hard cap. Chunks also end
//! at checkpoint positions: every `state.checkpoint_interval` tokens and just
//! before the prompt's last token, which then runs as its own step and is
//! sampled. That checkpoint lets a later request with the same prompt, or one
//! that only changes what follows it, restart one token before the end.
//! Decode is one token per submission, and only the sampled token id (and
//! its stop flag) comes back. A program that cannot take a step's token
//! from the previous step's sample on the GPU (Qwen4Exp, whose host picks
//! each token's PLE rows: plan.Program.chainsSamples) decodes one batch at
//! a time, each step processing the token read back.
//!
//! Speculative decoding, with a program built with the model's MTP block:
//! each round is one batch that runs the MTP block over the last step's
//! positions (drafting the token after next), then verifies the sampled
//! token and the draft in one two-position step, sampling at both. The
//! draft is accepted when the first sample equals it; then both samples are
//! emitted, else only the first, and the DeltaNet state rolls back to the
//! kept first row at the start of the next batch. Every emitted token is the
//! model's own sample with the draw it would get without speculation, and
//! the decode kernels give each row the same result in a step of two as in
//! a step of one, so the output is the same, bitwise. Rounds run one batch
//! at a time: the next round's positions depend on the verdict. Prefill
//! runs the MTP block over each chunk too, so drafts see the whole prompt.
//! The rollback also uncounts the discarded second sample in the sampler's
//! output counts, which the penalties read. Steps that would cross a
//! checkpoint position or the last sample max_tokens allows decode one token.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cmd = @import("../gpu/cmd.zig");
const plan = @import("../plan/plan.zig");
const kernels = @import("../kernels/kernels.zig");
const sampler = @import("../sampler/sampler.zig");
const state = @import("../state/state.zig");
const Tokenizer = @import("../tokenizer/tokenizer.zig").Tokenizer;
const Stream = @import("../tokenizer/stream.zig").Stream;

/// Target GPU time per submission (spec §8).
pub const budget_ns = 150 * std.time.ns_per_ms;
/// No submission may take longer (spec §8).
pub const hard_cap_ns = 250 * std.time.ns_per_ms;
/// Largest prefill chunk while the per-token cost is not yet known. Below it
/// the first chunk is as many tokens as one-token steps fit in 90% of the
/// budget: a prefill token costs no more than a whole decode step, so even a
/// large model's first chunk stays within the budget (and far from GPU
/// watchdogs).
pub const first_chunk = 64;
/// Prefill chunks end at multiples of this (or at the prompt's last two
/// positions). Small, so that a large model's steps can stay short. Every
/// chunk but the prompt's last token runs the prefill kernels, however few
/// tokens it has (the chunk before the last token can be a short tail), so
/// the kernels each prompt position goes through depend on the prompt
/// alone, not on step timings: the prefill kernels give the same result
/// however a prompt is split.
pub const chunk_align = 8;

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
    /// Each emitted token's log-probability under the model's raw logits,
    /// and the `logprobs` most likely tokens' (at most max_top_logprobs), as
    /// vLLM reports them by default. Null: none.
    logprobs: ?u32 = null,
    /// Speculate with the MTP block, where the program has it; null: where
    /// it pays for the model (`speculates`).
    speculate: ?bool = null,
};

/// The most top log-probabilities a request may ask for.
pub const max_top_logprobs = 20;

/// A token's log-probability, with the most likely alternatives.
pub const Logprob = struct {
    logprob: f32,
    top: []const Top,

    pub const Top = struct { id: u32, logprob: f32 };
};

pub const Finish = enum {
    /// A stop token or stop string.
    stop,
    /// max_tokens, or the context is full.
    length,
    /// `cancel`.
    abort,
};

pub const Usage = struct {
    prompt_tokens: u32 = 0,
    /// Prompt tokens served from the resident sequence instead of prefill.
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    prefill_steps: u32 = 0,
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,
    /// Speculative rounds, and how many of their drafts were accepted.
    drafted: u32 = 0,
    accepted: u32 = 0,
};

pub const Event = union(enum) {
    /// Execute this batch, then call `complete`; or, executing
    /// asynchronously, call `next` again. The words are valid until the
    /// next call.
    submit: []const u32,
    /// A sampled token and the text it released (valid until the next call).
    token: struct { id: u32, text: []const u8, logprob: ?Logprob = null },
    /// Complete the oldest outstanding batch before anything else can happen.
    wait,
    done: struct { finish: Finish, usage: Usage },
};

/// Batches outstanding at once.
pub const max_in_flight = 2;

pub const Error = Allocator.Error || Tokenizer.DecodeError || sampler.Params.Error || plan.Program.StepError || error{ Busy, EmptyPrompt, PromptTooLong, TooManyStops, TooManyLogprobs, NotSubmitted };

pub const Engine = struct {
    gpa: Allocator,
    program: *plan.Program,
    tok: *const Tokenizer,
    seq: state.Sequence,
    rec: cmd.Recorder = .{},
    phase: Phase = .idle,
    /// Measured cost of a one-token step (the fixed part), 0 until seen.
    step_ns: u64 = 0,
    /// Cost per additional token on the latest prefill step, its tokens
    /// counted in whole tiles, 0 until seen. It grows with position
    /// (attention), so the latest is the best guess.
    token_ns: u64 = 0,
    /// Tokens in the step `token_ns` was measured on.
    rate_tokens: u32 = 0,
    /// Calibration batches run.
    calibrations: u32 = 0,
    submits: u32 = 0,

    // Per request.
    req: Request = undefined,
    draws: sampler.Draws = undefined,
    stream: Stream = .{ .stops = &.{} },
    text: std.ArrayList(u8) = .empty,
    usage: Usage = .{},
    /// Prompt position of the next prefill chunk.
    prefill_pos: u32 = 0,
    /// Readbacks of sampling steps in flight, (token, is_stop), one per
    /// queue slot.
    sampled: [max_in_flight][2]u32 = @splat(.{ 0, 0 }),
    /// Outstanding batches, oldest first.
    queue: [max_in_flight]Pending = undefined,
    queued: u32 = 0,
    /// The readback slot the next sampling step uses.
    next_slot: u32 = 0,
    /// Sampling steps recorded for this request.
    samples_recorded: u32 = 0,
    /// The latest sampled token the host knows, which the next decode step
    /// processes.
    last_token: u32 = 0,
    /// A resident position whose token is the sample of a step still in
    /// flight, filled when it completes.
    placeholder: ?u32 = null,
    /// The token to emit (phase .emit).
    emit_token: u32 = 0,
    /// Logits rows for logprobs, one per queue slot (vocab each).
    lp_rows: []f32 = &.{},
    /// Where each slot's sampling step reads its logits back to.
    lp_src: [max_in_flight][]const f32 = @splat(&.{}),
    /// The token to emit's logprobs (phase .emit), if asked for.
    emit_logprob: ?Logprob = null,
    lp_top: [max_top_logprobs]Logprob.Top = undefined,
    finish: ?Finish = null,
    cancelled: bool = false,
    /// Speculating in this request.
    spec: bool = false,
    /// The last step's positions the MTP block has yet to run over (their
    /// final hidden states are rows 0.. of the program's `h`).
    mtp_pos: u32 = 0,
    mtp_rows: u32 = 0,
    /// The last draft was rejected: roll the DeltaNet state back, and
    /// uncount the second sample, before anything else runs.
    rollback: bool = false,
    /// The draws of the verify step in flight.
    verify_draws: [2]f32 = .{ 0, 0 },
    /// The second draw of a rejected verify step, for the next sample.
    spare_draw: ?f32 = null,
    /// The verify step's draft, read back.
    draft: u32 = 0,
    /// An accepted verify step's second sample, emitted after the first.
    second: ?Second = null,
    take_second: bool = false,
    /// `rec` holds a batch already handed to the host: the next recording
    /// starts a fresh one. (`start` leaves commands for the first batch.)
    handed: bool = false,

    const Phase = enum { idle, prefill, decode, emit };
    const Pending = struct { tokens: u32, prefill: bool, slot: ?u32, verify: bool = false };
    const Second = struct { token: u32, stop: bool, logits: []const f32 };

    /// The engine keeps pointers to itself in recorded batches: it must not
    /// move while a batch is pending.
    pub fn init(gpa: Allocator, program: *plan.Program, tok: *const Tokenizer) Allocator.Error!Engine {
        return .{ .gpa = gpa, .program = program, .tok = tok, .seq = try .init(gpa, program.checkpoints()) };
    }

    pub fn deinit(e: *Engine) void {
        e.seq.deinit(e.gpa);
        e.rec.deinit(e.gpa);
        e.stream.deinit(e.gpa);
        e.text.deinit(e.gpa);
        e.gpa.free(e.lp_rows);
        e.* = undefined;
    }

    /// The resident sequence: every token whose state the GPU holds.
    pub fn resident(e: *const Engine) []const u32 {
        return e.seq.tokens.items;
    }

    /// The next calibration batch, or null when calibration is done: a
    /// single token processed on a zeroed state, which is zeroed again after,
    /// dropping the resident sequence. The host runs each before the first
    /// request and passes its time to `calibrated`: the first warms up the
    /// pipelines and is discarded, the second measures the fixed cost of a
    /// step, which the prefill cost model needs from the start. The batch is
    /// valid until `calibrated`.
    pub fn calibration(e: *Engine) Error!?[]const u32 {
        if (e.queued != 0 or e.phase != .idle) return error.Busy;
        if (e.calibrations == 2) return null;
        e.forget();
        e.rec.clear();
        try e.program.recordReset(e.gpa, &e.rec);
        try e.program.recordStep(e.gpa, &e.rec, .{ .pos0 = 0, .tokens = &calibration_token });
        try e.program.recordReset(e.gpa, &e.rec);
        e.submits += 1;
        try e.rec.submit(e.gpa, e.submits);
        return e.rec.batch();
    }

    pub fn calibrated(e: *Engine, elapsed_ns: u64) void {
        e.rec.clear();
        if (e.calibrations == 1) e.step_ns = elapsed_ns;
        e.calibrations += 1;
    }

    /// Whether requests speculate by default (plan.speculationPays).
    pub fn speculates(e: *const Engine) bool {
        return e.program.mtp() and e.program.speculationPays();
    }

    const calibration_token = [_]u32{0};

    /// `req` and the memory it points to must stay valid until `.done`.
    pub fn start(e: *Engine, req: Request) Error!void {
        if (e.phase != .idle) return error.Busy;
        const vocab = e.program.vocab();
        if (req.prompt.len == 0) return error.EmptyPrompt;
        if (req.prompt.len > e.program.context()) return error.PromptTooLong;
        for (req.prompt) |t| if (t >= vocab) return error.InvalidToken;
        if (req.stop_tokens.len > kernels.max_stops) return error.TooManyStops;
        try req.sampling.validate(vocab);
        if (req.logprobs) |n| {
            if (n > max_top_logprobs) return error.TooManyLogprobs;
            if (e.lp_rows.len < max_in_flight * vocab) {
                e.gpa.free(e.lp_rows);
                e.lp_rows = &.{};
                e.lp_rows = try e.gpa.alloc(f32, max_in_flight * vocab);
            }
        }

        e.rec.clear();
        e.handed = false;
        if (e.rollback) try e.program.recordRollback(e.gpa, &e.rec);
        e.rollback = false;
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
        e.queued = 0;
        e.samples_recorded = 0;
        e.placeholder = null;
        e.finish = null;
        e.cancelled = false;
        e.spec = e.program.mtp() and (req.speculate orelse e.speculates());
        e.mtp_rows = 0;
        e.spare_draw = null;
        e.second = null;
        e.take_second = false;
        e.phase = .prefill;
    }

    /// Ends the running request after the batch in flight (see above).
    pub fn cancel(e: *Engine) void {
        if (e.phase != .idle) e.cancelled = true;
    }

    pub fn next(e: *Engine) Error!Event {
        if (e.take_second) {
            e.take_second = false;
            const s = e.second.?;
            e.second = null;
            try e.takeSample(s.token, s.stop, s.logits);
        }
        switch (e.phase) {
            .idle => return error.NotSubmitted,
            .emit => {
                e.phase = .decode;
                e.take_second = e.second != null;
                return .{ .token = .{ .id = e.emit_token, .text = e.text.items, .logprob = e.emit_logprob } };
            },
            .prefill, .decode => {},
        }
        if (e.cancelled and e.finish == null) e.finish = .abort;
        if (e.finish) |f| {
            if (e.queued > 0) return .wait;
            e.phase = .idle;
            return .{ .done = .{ .finish = f, .usage = e.usage } };
        }
        if (e.phase == .prefill) {
            // Prefill steps run one at a time: each chunk's size follows the
            // previous one's measured cost.
            if (e.queued > 0) return .wait;
            try e.recordPrefill();
        } else if (e.spec) {
            if (e.queued > 0 or !e.canDecode()) return .wait;
            try e.recordRound();
        } else {
            const in_flight: u32 = if (e.program.chainsSamples()) max_in_flight else 1;
            if (e.queued == in_flight or !e.canDecode()) return .wait;
            try e.recordDecode();
        }
        e.handed = true;
        return .{ .submit = e.rec.batch() };
    }

    /// Starts a new batch unless `rec` holds commands for it already.
    fn freshBatch(e: *Engine) void {
        if (e.handed) e.rec.clear();
        e.handed = false;
    }

    /// Whether another decode step may be recorded: its sample stays within
    /// max_tokens and the token it processes within the context.
    fn canDecode(e: *const Engine) bool {
        return e.samples_recorded < e.req.max_tokens and e.seq.len() < e.program.context();
    }

    /// Reports that the oldest outstanding batch finished, `elapsed_ns` of
    /// GPU time after it started.
    pub fn complete(e: *Engine, elapsed_ns: u64) Error!void {
        if (e.queued == 0 or e.phase == .emit) return error.NotSubmitted;
        const p = e.queue[0];
        std.mem.copyForwards(Pending, e.queue[0 .. e.queued - 1], e.queue[1..e.queued]);
        e.queued -= 1;
        if (p.prefill) {
            e.usage.prefill_ns += elapsed_ns;
            e.usage.prefill_steps += 1;
        } else {
            e.usage.decode_ns += elapsed_ns;
        }
        if (p.tokens == 1) {
            e.step_ns = ema(e.step_ns, elapsed_ns);
        } else if (e.step_ns != 0 and elapsed_ns > e.step_ns and p.tokens > kernels.decode_max_rows) {
            // A step faster than the fixed estimate is noise, not a rate.
            // Steps of a few tokens run the decode kernels, whose cost says
            // nothing about a prefill tile's.
            e.token_ns = (elapsed_ns - e.step_ns) / std.mem.alignForward(u64, p.tokens, e.program.prefillTile());
            e.rate_tokens = p.tokens;
        }
        if (p.verify) return e.completeVerify();
        const slot = p.slot orelse return;

        const token = e.sampled[slot][0];
        if (e.placeholder) |i| {
            // A later step already processes this token.
            e.seq.tokens.items[i] = token;
            e.placeholder = null;
        }
        try e.takeSample(token, e.sampled[slot][1] != 0, e.lp_src[slot]);
    }

    /// Settles a verify step: keeps the draft or drops it (rolling the
    /// state back at the next batch), then emits the first sample, and the
    /// second after it when the draft was right.
    fn completeVerify(e: *Engine) Error!void {
        const pos = e.seq.len() - 2;
        const first = e.sampled[0];
        const accepted = first[0] == e.draft;
        e.usage.drafted += 1;
        if (accepted) {
            e.usage.accepted += 1;
            e.seq.tokens.items[pos + 1] = e.draft;
        } else {
            e.seq.tokens.shrinkRetainingCapacity(pos + 1);
            e.rollback = true;
            e.spare_draw = e.verify_draws[1];
            e.samples_recorded -= 1;
        }
        e.mtp_pos = pos;
        e.mtp_rows = if (accepted) 2 else 1;
        try e.takeSample(first[0], first[1] != 0, e.lp_src[0]);
        if (accepted and e.finish == null) e.second = .{ .token = e.sampled[1][0], .stop = e.sampled[1][1] != 0, .logits = e.lp_src[1] };
    }

    /// Makes a sample the latest token and, unless the request has
    /// finished, the next one to emit, deciding whether it finishes it.
    fn takeSample(e: *Engine, token: u32, stop: bool, logits: []const f32) Error!void {
        e.last_token = token;
        e.emit_logprob = if (e.req.logprobs) |n| e.logprob(logits, token, n) else null;
        // Recorded before the request finished: the step processed its
        // token, but its sample is not part of the reply.
        if (e.finish != null) return;
        e.usage.completion_tokens += 1;
        e.text.clearRetainingCapacity();
        if (stop) {
            try e.stream.flush(e.gpa, &e.text);
            e.finish = .stop;
        } else if (try e.stream.push(e.gpa, e.tok, token, &e.text)) {
            e.finish = .stop;
        } else if (e.usage.completion_tokens == e.req.max_tokens or (e.queued == 0 and e.seq.len() == e.program.context())) {
            try e.stream.flush(e.gpa, &e.text);
            e.finish = .length;
        }
        e.emit_token = token;
        e.phase = .emit;
    }

    /// Tokens for the next prefill step: as many as the budget allows, up to
    /// the next checkpoint position and the end of the prompt. The target is
    /// 90% of the budget, as the cost per token keeps growing with position.
    /// Where a one-token step already takes more than half of it (a slow
    /// GPU), smaller chunks would only add steps that each pay that fixed
    /// cost, so the target becomes twice the fixed cost instead. Chunks are
    /// whole tiles; where not even one fits the target, a partial tile would
    /// cost as much, so the chunk is one tile if that stays within 90% of
    /// the hard cap. A chunk grows at most 4× past the step its rate was
    /// measured on, since the rate rises with the chunk size too.
    fn chunkSize(e: *const Engine) u32 {
        const pos = e.prefill_pos;
        const left: u32 = @intCast(e.req.prompt.len - pos);
        // Without calibration, measure the fixed cost first.
        var m: u32 = if (e.step_ns == 0) 1 else @intCast(std.math.clamp(budget_ns / 10 * 9 / e.step_ns, 1, first_chunk));
        if (e.token_ns != 0) {
            const tile: u64 = e.program.prefillTile();
            const target: u64 = @max(budget_ns / 10 * 9, 2 * e.step_ns);
            const ramp = 4 * @as(u64, @max(e.rate_tokens, first_chunk));
            var fit = @min((target -| e.step_ns) / e.token_ns, ramp);
            if (fit >= tile) {
                fit = fit / tile * tile;
            } else if (e.step_ns + e.token_ns * tile <= hard_cap_ns / 10 * 9) {
                fit = tile;
            }
            m = @intCast(@max(1, fit));
        }
        const max = e.program.maxTokens();
        m = @min(m, max, left);
        const boundary = (pos / state.checkpoint_interval + 1) * state.checkpoint_interval;
        m = @min(m, boundary - pos);
        const len: u32 = @intCast(e.req.prompt.len);
        const before_last = len - 1;
        var end = pos + m;
        if (pos < before_last) end = @min(end, before_last);
        if (end != before_last and end != len) {
            // Steps can be shorter than the alignment (Qwen4Exp's 4).
            const step_align = @min(chunk_align, max);
            const aligned = end / step_align * step_align;
            end = if (aligned > pos) aligned else @min((pos / step_align + 1) * step_align, before_last);
        }
        return end - pos;
    }

    fn recordPrefill(e: *Engine) Error!void {
        e.freshBatch();
        const pos0 = e.prefill_pos;
        const m = e.chunkSize();
        const tokens = e.req.prompt[pos0..][0..m];
        const last = pos0 + m == e.req.prompt.len;
        const slot = if (last) e.takeSlot() else null;
        try e.program.recordStep(e.gpa, &e.rec, .{
            .pos0 = pos0,
            .tokens = tokens,
            .logit_rows = @intFromBool(last),
            .draw = if (last) e.takeDraw() else null,
            .sampled = if (slot) |s| &e.sampled[s] else null,
            .logits = if (slot) |s| e.logitsRow(s, e.samples_recorded - 1) else null,
            .path = if (last) null else .prefill,
            .hidden = e.spec,
        });
        if (e.spec) {
            // Each position's MTP row takes the token after it; the prompt's
            // last token waits for its sample (the first round runs it).
            std.debug.assert(!last or m == 1);
            if (last) {
                e.mtp_pos = pos0;
                e.mtp_rows = 1;
            } else {
                try e.program.recordMtp(e.gpa, &e.rec, .{ .pos0 = pos0, .tokens = e.req.prompt[pos0 + 1 ..][0..m] });
            }
        }
        try e.seq.append(e.gpa, tokens);
        e.prefill_pos += m;
        if (e.seq.len() + 1 == e.req.prompt.len or e.seq.len() % state.checkpoint_interval == 0) try e.save();
        try e.submitBatch(.{ .tokens = m, .prefill = true, .slot = slot });
        if (last) e.phase = .decode;
    }

    /// Processes the latest sampled token, or a placeholder for the sample
    /// of the step still in flight, and samples the next.
    fn recordDecode(e: *Engine) Error!void {
        e.freshBatch();
        const pos0 = e.seq.len();
        const slot = e.takeSlot();
        // Unless the program chains samples, nothing is in flight: the host
        // knows the token.
        const chains = e.program.chainsSamples();
        try e.program.recordStep(e.gpa, &e.rec, .{
            .pos0 = pos0,
            .tokens = if (chains) null else (&e.last_token)[0..1],
            .logit_rows = 1,
            .draw = e.takeDraw(),
            .sampled = &e.sampled[slot],
            .logits = e.logitsRow(slot, e.samples_recorded - 1),
        });
        const unknown = for (e.queue[0..e.queued]) |q| {
            if (q.slot != null) break true;
        } else false;
        if (unknown) e.placeholder = pos0;
        try e.seq.append(e.gpa, &.{if (unknown) 0 else e.last_token});
        if (e.seq.len() % state.checkpoint_interval == 0) try e.save();
        try e.submitBatch(.{ .tokens = 1, .prefill = false, .slot = slot });
    }

    /// A speculative round: the MTP block over the last step's positions,
    /// drafting the token after next, then the sampled token and the draft
    /// verified in one step; or, where a round may not verify (see the top),
    /// the sampled token alone, through the same kernels.
    fn recordRound(e: *Engine) Error!void {
        e.freshBatch();
        if (e.rollback) try e.program.recordRollback(e.gpa, &e.rec);
        e.rollback = false;
        const pos = e.seq.len();
        const token = e.last_token;
        const interval = state.checkpoint_interval;
        const verify = e.mtp_rows > 0 and e.samples_recorded + 2 <= e.req.max_tokens and
            pos + 2 <= e.program.context() and (pos + 1) % interval != 0 and (pos + 2) % interval != 0;
        if (e.mtp_rows > 0) {
            // Each row takes the token after its position: resident, or the
            // latest sample for the last.
            var tokens: [2]u32 = undefined;
            for (tokens[0..e.mtp_rows], 0..) |*t, i| {
                const at = e.mtp_pos + i + 1;
                t.* = if (at < pos) e.seq.tokens.items[at] else token;
            }
            try e.program.recordMtp(e.gpa, &e.rec, .{
                .pos0 = e.mtp_pos,
                .tokens = tokens[0..e.mtp_rows],
                .draft = verify,
            });
            e.mtp_rows = 0;
        }
        if (verify) {
            e.verify_draws = .{ e.takeDraw(), e.takeDraw() };
            e.samples_recorded += 2;
            const k = e.samples_recorded - 2;
            try e.program.recordVerify(e.gpa, &e.rec, .{
                .pos0 = pos,
                .token = token,
                .draws = e.verify_draws,
                .sampled = &e.sampled,
                .draft = &e.draft,
                .logits = .{ e.logitsRow(0, k), e.logitsRow(1, k + 1) },
            });
            try e.seq.append(e.gpa, &.{ token, 0 });
            try e.submitBatch(.{ .tokens = 2, .prefill = false, .slot = null, .verify = true });
            return;
        }
        const slot = e.takeSlot();
        try e.program.recordStep(e.gpa, &e.rec, .{
            .pos0 = pos,
            .tokens = (&token)[0..1],
            .logit_rows = 1,
            .draw = e.takeDraw(),
            .sampled = &e.sampled[slot],
            .logits = e.logitsRow(slot, e.samples_recorded - 1),
            .hidden = true,
        });
        try e.seq.append(e.gpa, &.{token});
        e.mtp_pos = pos;
        e.mtp_rows = 1;
        if (e.seq.len() % interval == 0) try e.save();
        try e.submitBatch(.{ .tokens = 1, .prefill = false, .slot = slot });
    }

    /// The next sample's draw: a rejected verify step's unused one first.
    fn takeDraw(e: *Engine) f32 {
        if (e.spare_draw) |u| {
            e.spare_draw = null;
            return u;
        }
        return e.draws.next();
    }

    /// The readback slot for a new sampling step.
    fn takeSlot(e: *Engine) u32 {
        const slot = e.next_slot;
        e.next_slot = (slot + 1) % max_in_flight;
        e.samples_recorded += 1;
        return slot;
    }

    /// The row sample `k` (of the request) reads its logits back to, in
    /// `slot`: the request's buffer, for checks, or the slot's row for
    /// logprobs.
    fn logitsRow(e: *Engine, slot: u32, k: u32) ?[]f32 {
        const vocab = e.program.vocab();
        const row: []f32 = if (e.req.logits) |all|
            all[k * vocab ..][0..vocab]
        else if (e.req.logprobs != null)
            e.lp_rows[slot * vocab ..][0..vocab]
        else
            return null;
        e.lp_src[slot] = row;
        return row;
    }

    /// The log-probability of `token` under `logits`, and the `n` most likely
    /// tokens', most likely first (ties: lower id first).
    fn logprob(e: *Engine, logits: []const f32, token: u32, n: u32) Logprob {
        var max: f32 = -std.math.inf(f32);
        for (logits) |l| max = @max(max, l);
        var sum: f64 = 0;
        for (logits) |l| sum += @exp(@as(f64, l) - max);
        const lse: f64 = @as(f64, max) + @log(sum);
        const top = e.lp_top[0..n];
        var have: usize = 0;
        for (logits, 0..) |l, id| {
            if (have == n and (n == 0 or l <= top[n - 1].logprob)) continue;
            var i = if (have < n) have else n - 1;
            while (i > 0 and top[i - 1].logprob < l) : (i -= 1) top[i] = top[i - 1];
            top[i] = .{ .id = @intCast(id), .logprob = l };
            if (have < n) have += 1;
        }
        for (top[0..have]) |*t| t.logprob = @floatCast(@as(f64, t.logprob) - lse);
        return .{ .logprob = @floatCast(@as(f64, logits[token]) - lse), .top = top[0..have] };
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
        e.queue[e.queued] = p;
        e.queued += 1;
    }
};

test "logprobs: the sampled token's and the top ones, most likely first" {
    var e: Engine = undefined;
    const logits = [_]f32{ 1.0, 3.0, -2.0, 3.0, 0.5, 2.0 };
    var sum: f64 = 0;
    for (logits) |l| sum += @exp(@as(f64, l));
    const lse = @log(sum);
    const lp = e.logprob(&logits, 4, 3);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(0.5 - lse)), lp.logprob, 1e-6);
    // Ties keep the lower id first.
    try std.testing.expectEqual(@as(usize, 3), lp.top.len);
    try std.testing.expectEqual(@as(u32, 1), lp.top[0].id);
    try std.testing.expectEqual(@as(u32, 3), lp.top[1].id);
    try std.testing.expectEqual(@as(u32, 5), lp.top[2].id);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(2.0 - lse)), lp.top[2].logprob, 1e-6);
    try std.testing.expectEqual(@as(usize, 0), e.logprob(&logits, 0, 0).top.len);
    try std.testing.expectEqual(@as(usize, 6), e.logprob(&logits, 0, 20).top.len);
}
