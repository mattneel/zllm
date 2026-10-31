//! Load-time precision choices and CPU-only quality gates. Each tensor class
//! walks its own fastest-first ladder; every observation uses the same exact
//! f32 reference logits. Stored weights are never silently re-quantized.

const std = @import("std");
const common = @import("plan/common.zig");

pub const Mode = enum { auto, exact };
pub const Prefill = enum { fp16_single, fp16_compensated, scalar_f32 };
pub const WeightStorage = enum { stored };

pub const kv_rungs = [_]common.KvFormat{ .fp8, .f16, .f32 };
pub const prefill_rungs = [_]Prefill{ .fp16_single, .fp16_compensated, .scalar_f32 };
pub const weight_rungs = [_]WeightStorage{.stored};

pub const Selection = struct {
    kv: common.KvFormat = .f32,
    prefill: Prefill = .scalar_f32,
    weights: WeightStorage = .stored,
};

pub const Policy = struct {
    mode: Mode = .auto,
    kv: ?common.KvFormat = null,

    /// A forced KV format only opts out of the KV ladder. Exact overrides
    /// every approximation, including an explicitly forced KV format.
    pub fn resolve(policy: Policy, selection: Selection) Selection {
        if (policy.mode == .exact) return .{};
        var resolved = selection;
        if (policy.kv) |kv| resolved.kv = kv;
        return resolved;
    }
};

/// Bump whenever model thresholds, metric semantics or ladder ordering change.
/// The plan cache hashes this version together with the requested policy.
pub const gate_policy_version: u32 = 1;

pub const Gate = struct {
    min_top1_agreement: f64 = 0.99,
    max_mean_kl: f64 = 0.001,
    max_rel_l2: f64 = 0.1,

    /// A named checkpoint rule takes precedence over its architecture default.
    /// These quality limits are independent of the strict implementation gates;
    /// docs/src/performance.md records measured accepted and rejected rungs.
    pub fn forModel(architecture: []const u8, model_name: []const u8) Gate {
        var architecture_default: ?Gate = null;
        for (model_policies) |rule| {
            if (!std.mem.eql(u8, architecture, rule.architecture)) continue;
            if (rule.model_name) |name| {
                if (std.mem.eql(u8, model_name, name)) return rule.gate;
            } else architecture_default = rule.gate;
        }
        return architecture_default orelse .{};
    }

    pub fn accepts(gate: Gate, metrics: Metrics) bool {
        if (!metrics.valid() or metrics.positions == 0) return false;
        if (!std.math.isFinite(gate.min_top1_agreement) or gate.min_top1_agreement < 0 or gate.min_top1_agreement > 1 or
            !std.math.isFinite(gate.max_mean_kl) or gate.max_mean_kl < 0 or
            !std.math.isFinite(gate.max_rel_l2) or gate.max_rel_l2 < 0) return false;
        return metrics.top1Agreement() >= gate.min_top1_agreement and
            metrics.mean_kl <= gate.max_mean_kl and metrics.max_rel_l2 <= gate.max_rel_l2;
    }
};

pub const ModelPolicy = struct {
    architecture: []const u8,
    model_name: ?[]const u8 = null,
    gate: Gate,
};

// Defaults apply per checkpoint unless a named override precedes/follows them.
// The 0.001-nat KL limit separates the measured Gyro FP16-single/FP8 failures
// from compensated arithmetic; the 99% top-1 gate also rejects Qwen's FP8 rung.
pub const model_policies = [_]ModelPolicy{
    .{ .architecture = "qwen35", .gate = .{ .min_top1_agreement = 0.99, .max_mean_kl = 0.001, .max_rel_l2 = 0.1 } },
    .{ .architecture = "qwen4exp", .gate = .{ .min_top1_agreement = 0.99, .max_mean_kl = 0.001, .max_rel_l2 = 0.1 } },
    .{ .architecture = "qwen3", .gate = .{ .min_top1_agreement = 0.99, .max_mean_kl = 0.001, .max_rel_l2 = 0.1 } },
    .{ .architecture = "llama", .gate = .{ .min_top1_agreement = 0.99, .max_mean_kl = 0.001, .max_rel_l2 = 0.1 } },
    .{ .architecture = "granite", .gate = .{ .min_top1_agreement = 0.99, .max_mean_kl = 0.001, .max_rel_l2 = 0.1 } },
};

/// Public, serializable aggregate; add commits a position only after every
/// result is finite. All arithmetic and reductions use deterministic f64 CPU
/// operations, with compensated sums within each vocabulary reduction.
pub const Metrics = struct {
    positions: u64 = 0,
    top1_matches: u64 = 0,
    mean_kl: f64 = 0,
    max_rel_l2: f64 = 0,

    pub const Error = error{ InvalidShape, NonFinite, TooManyPositions };

    pub fn valid(metrics: Metrics) bool {
        if (metrics.top1_matches > metrics.positions or
            !std.math.isFinite(metrics.mean_kl) or metrics.mean_kl < 0 or
            !std.math.isFinite(metrics.max_rel_l2) or metrics.max_rel_l2 < 0) return false;
        return metrics.positions != 0 or (metrics.mean_kl == 0 and metrics.max_rel_l2 == 0);
    }

    pub fn top1Agreement(metrics: Metrics) f64 {
        if (metrics.positions == 0) return 0;
        return @as(f64, @floatFromInt(metrics.top1_matches)) / @as(f64, @floatFromInt(metrics.positions));
    }

    /// KL is P_exact || P_candidate. Log-softmax subtracts the maximum before
    /// both exponentiation and logarithms, avoiding overflow and cancellation
    /// from large common logit offsets. A zero reference norm has relative L2
    /// zero only for an identical candidate; otherwise that position rejects.
    pub fn add(metrics: *Metrics, exact: []const f32, candidate: []const f32) Error!void {
        if (exact.len == 0 or exact.len != candidate.len) return error.InvalidShape;
        if (!metrics.valid()) return error.NonFinite;
        if (metrics.positions == std.math.maxInt(u64)) return error.TooManyPositions;
        var exact_top: usize = 0;
        var candidate_top: usize = 0;
        var exact_max: f64 = -std.math.inf(f64);
        var candidate_max: f64 = -std.math.inf(f64);
        var squared_error: Sum = .{};
        var squared_reference: Sum = .{};
        for (exact, candidate, 0..) |a, b, i| {
            if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.NonFinite;
            const x: f64 = a;
            const y: f64 = b;
            if (x > exact_max) {
                exact_max = x;
                exact_top = i;
            }
            if (y > candidate_max) {
                candidate_max = y;
                candidate_top = i;
            }
            squared_reference.add(x * x);
            const delta = x - y;
            squared_error.add(delta * delta);
        }
        var exact_exp: Sum = .{};
        var candidate_exp: Sum = .{};
        for (exact, candidate) |a, b| {
            exact_exp.add(@exp(@as(f64, a) - exact_max));
            candidate_exp.add(@exp(@as(f64, b) - candidate_max));
        }
        const exact_log_sum = @log(exact_exp.value());
        const candidate_log_sum = @log(candidate_exp.value());
        var divergence: Sum = .{};
        for (exact, candidate) |a, b| {
            const log_p = (@as(f64, a) - exact_max) - exact_log_sum;
            const log_q = (@as(f64, b) - candidate_max) - candidate_log_sum;
            divergence.add(@exp(log_p) * (log_p - log_q));
        }
        // Roundoff in the signed KL reduction can produce a tiny negative
        // result even though the mathematical divergence is nonnegative.
        const raw_kl = divergence.value();
        if (!std.math.isFinite(raw_kl)) return error.NonFinite;
        const kl = @max(0, raw_kl);
        const reference_norm = squared_reference.value();
        const error_norm = squared_error.value();
        const relative_l2 = if (reference_norm == 0)
            (if (error_norm == 0) @as(f64, 0) else std.math.inf(f64))
        else
            @sqrt(error_norm / reference_norm);
        const positions = metrics.positions + 1;
        const mean_kl = metrics.mean_kl + (kl - metrics.mean_kl) / @as(f64, @floatFromInt(positions));
        if (!std.math.isFinite(kl) or !std.math.isFinite(relative_l2) or !std.math.isFinite(mean_kl)) return error.NonFinite;
        metrics.* = .{
            .positions = positions,
            .top1_matches = metrics.top1_matches + @intFromBool(exact_top == candidate_top),
            .mean_kl = mean_kl,
            .max_rel_l2 = @max(metrics.max_rel_l2, relative_l2),
        };
    }
};

// Neumaier summation also handles signed KL terms with very different sizes.
const Sum = struct {
    sum: f64 = 0,
    correction: f64 = 0,

    fn add(s: *Sum, value_: f64) void {
        const next = s.sum + value_;
        s.correction += if (@abs(s.sum) >= @abs(value_)) (s.sum - next) + value_ else (value_ - next) + s.sum;
        s.sum = next;
    }

    fn value(s: Sum) f64 {
        return s.sum + s.correction;
    }
};

pub const Observation = struct {
    selection: Selection,
    metrics: Metrics,
};

/// Observations must be supplied in fastest-first order for the tensor class
/// being tuned. A rejected or unsupported rung cannot mask a later valid one.
pub fn fastestPassing(observations: []const Observation, gate: Gate) ?Selection {
    for (observations) |observation| {
        if (gate.accepts(observation.metrics)) return observation.selection;
    }
    return null;
}

test "exact policy overrides forced KV; auto only overrides KV" {
    const approximate: Selection = .{ .kv = .fp8, .prefill = .fp16_single };
    try std.testing.expectEqualDeep(Selection{}, (Policy{ .mode = .exact, .kv = .f16 }).resolve(approximate));
    try std.testing.expectEqualDeep(approximate, (Policy{}).resolve(approximate));
    try std.testing.expectEqualDeep(Selection{ .kv = .f16, .prefill = .fp16_single }, (Policy{ .kv = .f16 }).resolve(approximate));
    try std.testing.expectEqualSlices(common.KvFormat, &.{ .fp8, .f16, .f32 }, &kv_rungs);
    try std.testing.expectEqualSlices(Prefill, &.{ .fp16_single, .fp16_compensated, .scalar_f32 }, &prefill_rungs);
    try std.testing.expectEqual(@as(usize, 1), weight_rungs.len);
}

test "identical finite logits pass, including large offsets and exact zeros" {
    var metrics: Metrics = .{};
    try metrics.add(&.{ 10000, 9999, -10000 }, &.{ 10000, 9999, -10000 });
    try metrics.add(&.{ 0, 0, 0 }, &.{ 0, 0, 0 });
    try metrics.add(&.{ std.math.floatMax(f32), -std.math.floatMax(f32) }, &.{ std.math.floatMax(f32), -std.math.floatMax(f32) });
    try std.testing.expectEqual(@as(u64, 3), metrics.positions);
    try std.testing.expectEqual(@as(f64, 1), metrics.top1Agreement());
    try std.testing.expectEqual(@as(f64, 0), metrics.mean_kl);
    try std.testing.expectEqual(@as(f64, 0), metrics.max_rel_l2);
    try std.testing.expect((Gate{}).accepts(metrics));
    try std.testing.expect(!(Gate{}).accepts(.{}));
}

test "mutated top1 fails agreement independently of divergence thresholds" {
    var metrics: Metrics = .{};
    try metrics.add(&.{ 1, 0.999 }, &.{ 0.999, 1 });
    try std.testing.expectEqual(@as(u64, 0), metrics.top1_matches);
    try std.testing.expect(metrics.mean_kl < 0.001);
    try std.testing.expect(metrics.max_rel_l2 < 0.1);
    try std.testing.expect(!(Gate{}).accepts(metrics));
    // A tie consistently picks the first vocabulary index.
    try metrics.add(&.{ 1, 1 }, &.{ 1, 1 });
    try std.testing.expectEqual(@as(u64, 1), metrics.top1_matches);
}

test "KL uses exact distribution, stable offsets and mean over positions" {
    var metrics: Metrics = .{};
    try metrics.add(&.{ 1, 0 }, &.{ 1, -1 });
    const p = @exp(@as(f64, 1)) / (1 + @exp(@as(f64, 1)));
    const q = @exp(@as(f64, 2)) / (1 + @exp(@as(f64, 2)));
    const want = p * @log(p / q) + (1 - p) * @log((1 - p) / (1 - q));
    try std.testing.expectApproxEqAbs(want, metrics.mean_kl, 1e-14);
    try std.testing.expect(!(Gate{ .max_rel_l2 = 2 }).accepts(metrics));
    var shifted: Metrics = .{};
    try shifted.add(&.{ 10001, 10000 }, &.{ 10001, 9999 });
    try std.testing.expectApproxEqAbs(want, shifted.mean_kl, 1e-14);
    try metrics.add(&.{ 1, 0 }, &.{ 1, 0 });
    try std.testing.expectApproxEqAbs(want / 2, metrics.mean_kl, 1e-14);
}

test "max relative logits L2 rejects a distribution-preserving mutation" {
    var metrics: Metrics = .{};
    try metrics.add(&.{ 3, 4 }, &.{ 4, 5 });
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 2)) / 5, metrics.max_rel_l2, 1e-14);
    try std.testing.expectApproxEqAbs(@as(f64, 0), metrics.mean_kl, 1e-14);
    try std.testing.expectEqual(@as(f64, 1), metrics.top1Agreement());
    try std.testing.expect(!(Gate{}).accepts(metrics));
    try metrics.add(&.{ 3, 4 }, &.{ 3, 4 });
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 2)) / 5, metrics.max_rel_l2, 1e-14);
}

test "nonfinite and malformed positions reject transactionally" {
    var metrics: Metrics = .{};
    try metrics.add(&.{ 1, 2 }, &.{ 1, 2 });
    const before = metrics;
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        try std.testing.expectError(error.NonFinite, metrics.add(&.{ bad, 2 }, &.{ 1, 2 }));
        try std.testing.expectError(error.NonFinite, metrics.add(&.{ 1, 2 }, &.{ 1, bad }));
        try std.testing.expectEqualDeep(before, metrics);
    }
    try std.testing.expectError(error.InvalidShape, metrics.add(&.{}, &.{}));
    try std.testing.expectError(error.InvalidShape, metrics.add(&.{1}, &.{ 1, 2 }));
    try std.testing.expectError(error.NonFinite, metrics.add(&.{0}, &.{1}));
    try std.testing.expectEqualDeep(before, metrics);
    try std.testing.expect(!(Gate{}).accepts(.{ .positions = 1, .top1_matches = 1, .mean_kl = std.math.nan(f64) }));
    try std.testing.expect(!(Gate{}).accepts(.{ .positions = 1, .top1_matches = 1, .max_rel_l2 = std.math.inf(f64) }));
    try std.testing.expect(!(Gate{}).accepts(.{ .positions = 1, .top1_matches = 2 }));
}

test "fastest passing rung wins, not the most accurate rung" {
    const observations = [_]Observation{
        .{ .selection = .{ .kv = .fp8 }, .metrics = .{ .positions = 100, .top1_matches = 98 } },
        .{ .selection = .{ .kv = .f16 }, .metrics = .{ .positions = 100, .top1_matches = 99, .mean_kl = 0.001, .max_rel_l2 = 0.1 } },
        .{ .selection = .{}, .metrics = .{ .positions = 100, .top1_matches = 100 } },
    };
    const gate = Gate.forModel("qwen35", "Qwen3.5-0.8B");
    try std.testing.expectEqual(common.KvFormat.f16, fastestPassing(&observations, gate).?.kv);
    try std.testing.expect(fastestPassing(observations[0..1], gate) == null);
    try std.testing.expect(fastestPassing(&.{}, gate) == null);
    try std.testing.expectEqualDeep(gate, Gate.forModel("unknown", "unknown"));
    for (model_policies) |rule| {
        try std.testing.expectEqualDeep(rule.gate, Gate.forModel(rule.architecture, "unmeasured checkpoint"));
    }
}
