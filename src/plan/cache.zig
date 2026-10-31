//! The plan cache (spec §7): what load-time tuning found, so a later load
//! with the same GPU, model and kernels skips the sweep. The key is a
//! SHA-256 over everything that could change the outcome: the adapter
//! (vendor, architecture, device, its description, which carries the driver
//! version on most backends, and its backend), the runtime (the browser, or
//! Dawn), the device limits the plan uses, the model's GGUF header and
//! tensor index, the zllm version, and every kernel's source (the template
//! set). The cache records tuned variants, selected precision and quality
//! observations; pipelines are rebuilt from them and the keyed templates.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const kernels = @import("../kernels/kernels.zig");
const common = @import("common.zig");
const Tuning = common.Tuning;
const precision = @import("../precision.zig");
const version = @import("../version.zig").zllm;

pub const Key = [std.crypto.hash.sha2.Sha256.digest_length]u8;

pub const Entry = struct {
    tuning: Tuning,
    selection: precision.Selection,
    observations: []const precision.Observation = &.{},
    /// A completed tuning sweep may legitimately produce no entries. False
    /// means this cache contains precision observations only (--no-tune).
    tuned: bool = false,
};

pub const KeyInputs = struct {
    /// Vendor, architecture, device, description, backend.
    adapter: []const []const u8,
    /// The browser's user agent, or the native host's name.
    runtime: []const u8,
    limits: []const u64,
    /// The GGUF file up to its tensor data: header, metadata, tensor index.
    model_header: []const u8,
    /// Whose kernels the tuning measured: the key hashes that backend's own
    /// kernel sources, so a change to any of them invalidates the cache.
    backend: kernels.Backend = .webgpu,
    policy: precision.Policy = .{},
};

/// The cache file's name for `k`: its hex and ".json".
pub const file_name_len = 2 * @sizeOf(Key) + ".json".len;

pub fn fileName(k: Key, buf: *[file_name_len]u8) []const u8 {
    return std.fmt.bufPrint(buf, "{x}.json", .{&k}) catch unreachable;
}

pub fn key(in: KeyInputs) Key {
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    const Feed = struct {
        fn bytes(hh: *std.crypto.hash.sha2.Sha256, b: []const u8) void {
            // Length-prefixed, so fields cannot run into each other.
            hh.update(std.mem.asBytes(&@as(u64, b.len)));
            hh.update(b);
        }
    };
    Feed.bytes(&h, "zllm plan cache 2");
    Feed.bytes(&h, std.mem.asBytes(&precision.gate_policy_version));
    Feed.bytes(&h, @tagName(in.policy.mode));
    Feed.bytes(&h, if (in.policy.kv) |kv| @tagName(kv) else "automatic");
    Feed.bytes(&h, version);
    for (in.adapter) |s| Feed.bytes(&h, s);
    Feed.bytes(&h, in.runtime);
    Feed.bytes(&h, std.mem.sliceAsBytes(in.limits));
    Feed.bytes(&h, in.model_header);
    Feed.bytes(&h, @tagName(in.backend));
    for (std.enums.values(kernels.Kernel)) |k| Feed.bytes(&h, kernels.backendSource(in.backend, k));
    var out: Key = undefined;
    h.final(&out);
    return out;
}

const Json = struct {
    key: []const u8,
    tuned: bool,
    tuning: []const struct {
        type: []const u8,
        n: u32,
        k: u32,
        wg: u32,
        lanes: u32,
        rows: u32,
    },
    // No defaults here: an old or partially written precision record is not
    // a cache hit, even though the runtime-facing types have safe defaults.
    selection: JsonSelection,
    observations: []const JsonObservation,
};

const JsonSelection = struct {
    kv: common.KvFormat,
    prefill: precision.Prefill,
    weights: precision.WeightStorage,

    fn selection(j: JsonSelection) precision.Selection {
        return .{ .kv = j.kv, .prefill = j.prefill, .weights = j.weights };
    }
};

const JsonMetrics = struct {
    positions: u64,
    top1_matches: u64,
    mean_kl: f64,
    max_rel_l2: f64,

    fn metrics(j: JsonMetrics) precision.Metrics {
        return .{ .positions = j.positions, .top1_matches = j.top1_matches, .mean_kl = j.mean_kl, .max_rel_l2 = j.max_rel_l2 };
    }
};

const JsonObservation = struct {
    selection: JsonSelection,
    metrics: JsonMetrics,
};

/// The cache file's complete tuning and precision entry under `k`.
pub fn serialize(gpa: Allocator, k: Key, entry: Entry, out: *std.ArrayList(u8)) Allocator.Error!void {
    var aw: std.Io.Writer.Allocating = .fromArrayList(gpa, out);
    defer out.* = aw.toArrayList();
    const w = &aw.writer;
    w.print("{{\"key\":\"{x}\",\"tuning\":[", .{&k}) catch return error.OutOfMemory;
    for (entry.tuning.entries, 0..) |e, i| {
        w.print("{s}\n{{\"type\":\"{t}\",\"n\":{d},\"k\":{d},\"wg\":{d},\"lanes\":{d},\"rows\":{d}}}", .{
            if (i == 0) "" else ",", e.type, e.n, e.k, e.variant.wg, e.variant.lanes, e.variant.rows,
        }) catch return error.OutOfMemory;
    }
    w.print("],\"tuned\":{},\"selection\":{f},\"observations\":{f}}}\n", .{
        entry.tuned,
        std.json.fmt(entry.selection, .{}),
        std.json.fmt(entry.observations, .{}),
    }) catch return error.OutOfMemory;
}

/// The entry stored in `bytes` if it was made under `k`; null for another
/// key or anything malformed (unknown precision, invalid metrics, or tuning
/// outside the actual kernel candidates), which a fresh sweep then replaces.
pub fn parse(arena: Allocator, bytes: []const u8, k: Key) ?Entry {
    const json = std.json.parseFromSliceLeaky(Json, arena, bytes, .{}) catch return null;
    var want: [2 * k.len]u8 = undefined;
    _ = std.fmt.bufPrint(&want, "{x}", .{&k}) catch unreachable;
    if (!std.mem.eql(u8, json.key, &want)) return null;
    const entries = arena.alloc(Tuning.Entry, json.tuning.len) catch return null;
    for (json.tuning, entries) |j, *e| {
        const t = std.meta.stringToEnum(gguf.TensorType, j.type) orelse return null;
        if (kernels.Kernel.weight(.matvec, t) == null or kernels.isHalf(t)) return null;
        const v: kernels.MatvecVariant = .{ .wg = j.wg, .lanes = j.lanes, .rows = j.rows };
        const known = for (kernels.MatvecVariant.candidates) |cand| {
            if (std.meta.eql(cand, v)) break true;
        } else false;
        if (!known) return null;
        e.* = .{ .type = t, .n = j.n, .k = j.k, .variant = v };
    }
    const observations = arena.alloc(precision.Observation, json.observations.len) catch return null;
    for (json.observations, observations) |j, *observation| {
        const metrics = j.metrics.metrics();
        if (!metrics.valid()) return null;
        observation.* = .{ .selection = j.selection.selection(), .metrics = metrics };
    }
    return .{ .tuning = .{ .entries = entries }, .selection = json.selection.selection(), .observations = observations, .tuned = json.tuned };
}

test "round trip, and a cache made under another key is ignored" {
    const gpa = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const inputs: KeyInputs = .{ .adapter = &.{ "nvidia", "blackwell", "RTX", "driver 1", "d3d12" }, .runtime = "dawn", .limits = &.{ 1 << 30, 256 }, .model_header = "GGUF..." };
    const k = key(inputs);
    const entries = [_]Tuning.Entry{
        .{ .type = .q3_k, .n = 34816, .k = 5120, .variant = .{ .lanes = 8 } },
        .{ .type = .q8_0, .n = 5120, .k = 6144, .variant = .{ .wg = 128, .rows = 2 } },
    };
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    const observations = [_]precision.Observation{
        .{ .selection = .{ .kv = .fp8 }, .metrics = .{ .positions = 100, .top1_matches = 98, .mean_kl = 0.0005, .max_rel_l2 = 0.05 } },
        .{ .selection = .{ .kv = .f16 }, .metrics = .{ .positions = 100, .top1_matches = 100, .mean_kl = 0.000001234567890123, .max_rel_l2 = 0.0009876543210987654 } },
        .{ .selection = .{ .kv = .f16, .prefill = .fp16_single }, .metrics = .{ .positions = 100, .top1_matches = 99, .mean_kl = 0.0001, .max_rel_l2 = 0.01 } },
    };
    const entry: Entry = .{
        .tuning = .{ .entries = &entries },
        .selection = .{ .kv = .f16, .prefill = .fp16_single },
        .observations = &observations,
        .tuned = true,
    };
    try serialize(gpa, k, entry, &bytes);
    const back = parse(arena, bytes.items, k).?;
    try std.testing.expectEqualSlices(Tuning.Entry, &entries, back.tuning.entries);
    try std.testing.expectEqualDeep(entry.selection, back.selection);
    try std.testing.expectEqualDeep(entry.observations, back.observations);
    try std.testing.expectEqual(entry.tuned, back.tuned);

    var other = inputs;
    other.adapter = &.{ "nvidia", "blackwell", "RTX", "driver 2", "d3d12" };
    try std.testing.expect(!std.mem.eql(u8, &k, &key(other)));
    // Agility's tuning is keyed by its own kernels, not WebGPU's.
    var agility = inputs;
    agility.backend = .agility;
    try std.testing.expect(!std.mem.eql(u8, &k, &key(agility)));
    try std.testing.expect(parse(arena, bytes.items, key(other)) == null);
}

test "malformed caches are ignored" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const k = key(.{ .adapter = &.{}, .runtime = "dawn", .limits = &.{}, .model_header = "" });
    var hex: [64]u8 = undefined;
    _ = try std.fmt.bufPrint(&hex, "{x}", .{&k});
    const precision_fields = "\"tuned\":true,\"selection\":{\"kv\":\"f32\",\"prefill\":\"scalar_f32\",\"weights\":\"stored\"},\"observations\":[]";
    const bad_variant = try std.fmt.allocPrint(arena, "{{\"key\":\"{s}\",\"tuning\":[{{\"type\":\"q3_k\",\"n\":1,\"k\":256,\"wg\":64,\"lanes\":8,\"rows\":1}}],{s}}}", .{ &hex, precision_fields });
    try std.testing.expect(parse(arena, bad_variant, k) == null);
    const bad_type = try std.fmt.allocPrint(arena, "{{\"key\":\"{s}\",\"tuning\":[{{\"type\":\"bf16\",\"n\":1,\"k\":256,\"wg\":256,\"lanes\":8,\"rows\":1}}],{s}}}", .{ &hex, precision_fields });
    try std.testing.expect(parse(arena, bad_type, k) == null);
    try std.testing.expect(parse(arena, "not json", k) == null);
    const empty = try std.fmt.allocPrint(arena, "{{\"key\":\"{s}\",\"tuning\":[],{s}}}", .{ &hex, precision_fields });
    try std.testing.expectEqual(@as(usize, 0), parse(arena, empty, k).?.tuning.entries.len);
    const old = try std.fmt.allocPrint(arena, "{{\"key\":\"{s}\",\"tuning\":[]}}", .{&hex});
    try std.testing.expect(parse(arena, old, k) == null);
}

test "precision policy choices invalidate cached selections" {
    const inputs: KeyInputs = .{ .adapter = &.{}, .runtime = "dawn", .limits = &.{}, .model_header = "" };
    const automatic = key(inputs);
    var forced = inputs;
    forced.policy.kv = .f32;
    const forced_f32 = key(forced);
    try std.testing.expect(!std.mem.eql(u8, &automatic, &forced_f32));
    forced.policy.kv = .f16;
    const forced_f16 = key(forced);
    try std.testing.expect(!std.mem.eql(u8, &forced_f32, &forced_f16));
    forced.policy.kv = .fp8;
    const forced_fp8 = key(forced);
    try std.testing.expect(!std.mem.eql(u8, &forced_f16, &forced_fp8));
    try std.testing.expect(!std.mem.eql(u8, &forced_f32, &forced_fp8));
    var exact = inputs;
    exact.policy.mode = .exact;
    const exact_key = key(exact);
    try std.testing.expect(!std.mem.eql(u8, &automatic, &exact_key));
    try std.testing.expect(!std.mem.eql(u8, &forced_f32, &exact_key));
    try std.testing.expect(!std.mem.eql(u8, &forced_f16, &exact_key));
    try std.testing.expect(!std.mem.eql(u8, &forced_fp8, &exact_key));
    exact.policy.kv = .fp8;
    try std.testing.expect(!std.mem.eql(u8, &exact_key, &key(exact)));
}

test "unknown and missing precision enum fields reject rather than default" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const k = key(.{ .adapter = &.{}, .runtime = "dawn", .limits = &.{}, .model_header = "" });
    for ([_][]const u8{
        "{\"kv\":\"unknown\",\"prefill\":\"scalar_f32\",\"weights\":\"stored\"}",
        "{\"kv\":\"f32\",\"prefill\":\"unknown\",\"weights\":\"stored\"}",
        "{\"kv\":\"f32\",\"prefill\":\"scalar_f32\",\"weights\":\"unknown\"}",
        "{\"prefill\":\"scalar_f32\",\"weights\":\"stored\"}",
        "{\"kv\":\"f32\",\"weights\":\"stored\"}",
        "{\"kv\":\"f32\",\"prefill\":\"scalar_f32\"}",
        "{}",
    }) |selection| {
        const bytes = try std.fmt.allocPrint(arena, "{{\"key\":\"{x}\",\"tuned\":true,\"tuning\":[],\"selection\":{s},\"observations\":[]}}", .{ &k, selection });
        try std.testing.expect(parse(arena, bytes, k) == null);
        const observation = try std.fmt.allocPrint(arena, "{{\"key\":\"{x}\",\"tuned\":true,\"tuning\":[],\"selection\":{{\"kv\":\"f32\",\"prefill\":\"scalar_f32\",\"weights\":\"stored\"}},\"observations\":[{{\"selection\":{s},\"metrics\":{{\"positions\":1,\"top1_matches\":1,\"mean_kl\":0,\"max_rel_l2\":0}}}}]}}", .{ &k, selection });
        try std.testing.expect(parse(arena, observation, k) == null);
    }
    const missing_observations = try std.fmt.allocPrint(arena, "{{\"key\":\"{x}\",\"tuned\":true,\"tuning\":[],\"selection\":{{\"kv\":\"f32\",\"prefill\":\"scalar_f32\",\"weights\":\"stored\"}}}}", .{&k});
    try std.testing.expect(parse(arena, missing_observations, k) == null);
    const missing_selection = try std.fmt.allocPrint(arena, "{{\"key\":\"{x}\",\"tuned\":true,\"tuning\":[],\"observations\":[]}}", .{&k});
    try std.testing.expect(parse(arena, missing_selection, k) == null);
}

test "invalid and incomplete observation metrics reject" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const k = key(.{ .adapter = &.{}, .runtime = "dawn", .limits = &.{}, .model_header = "" });
    for ([_][]const u8{
        "{\"positions\":1,\"top1_matches\":2,\"mean_kl\":0,\"max_rel_l2\":0}",
        "{\"positions\":1,\"top1_matches\":1,\"mean_kl\":-0.1,\"max_rel_l2\":0}",
        "{\"positions\":1,\"top1_matches\":1,\"mean_kl\":0,\"max_rel_l2\":-0.1}",
        "{\"positions\":1,\"top1_matches\":1,\"mean_kl\":1e999,\"max_rel_l2\":0}",
        "{\"positions\":1,\"top1_matches\":1,\"mean_kl\":0,\"max_rel_l2\":1e999}",
        "{\"positions\":1,\"top1_matches\":1,\"mean_kl\":0}",
        "{\"positions\":1,\"top1_matches\":1,\"max_rel_l2\":0}",
        "{\"positions\":1,\"mean_kl\":0,\"max_rel_l2\":0}",
        "{\"top1_matches\":1,\"mean_kl\":0,\"max_rel_l2\":0}",
        "{}",
    }) |metrics| {
        const bytes = try std.fmt.allocPrint(arena, "{{\"key\":\"{x}\",\"tuned\":true,\"tuning\":[],\"selection\":{{\"kv\":\"f32\",\"prefill\":\"scalar_f32\",\"weights\":\"stored\"}},\"observations\":[{{\"selection\":{{\"kv\":\"f16\",\"prefill\":\"fp16_compensated\",\"weights\":\"stored\"}},\"metrics\":{s}}}]}}", .{ &k, metrics });
        try std.testing.expect(parse(arena, bytes, k) == null);
    }
}

test "quality-only and completed empty tuning round trip distinctly" {
    const gpa = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const k = key(.{ .adapter = &.{}, .runtime = "dawn", .limits = &.{}, .model_header = "" });
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    const observation: precision.Observation = .{
        .selection = .{ .kv = .f16 },
        .metrics = .{ .positions = 100, .top1_matches = 100, .mean_kl = 0.00001, .max_rel_l2 = 0.0001 },
    };
    for ([_]bool{ false, true }) |tuned| {
        bytes.clearRetainingCapacity();
        try serialize(gpa, k, .{ .tuning = .{}, .selection = observation.selection, .observations = &.{observation}, .tuned = tuned }, &bytes);
        const back = parse(arena, bytes.items, k).?;
        try std.testing.expectEqual(tuned, back.tuned);
        try std.testing.expectEqual(@as(usize, 0), back.tuning.entries.len);
        try std.testing.expectEqualDeep(observation.selection, back.selection);
        try std.testing.expectEqualDeep(observation, back.observations[0]);
    }
}

test "tuning completion metadata must be an explicit boolean" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const k = key(.{ .adapter = &.{}, .runtime = "dawn", .limits = &.{}, .model_header = "" });
    for ([_][]const u8{ "", "\"tuned\":null,", "\"tuned\":1,", "\"tuned\":\"true\"," }) |field| {
        const bytes = try std.fmt.allocPrint(arena, "{{\"key\":\"{x}\",{s}\"tuning\":[],\"selection\":{{\"kv\":\"f32\",\"prefill\":\"scalar_f32\",\"weights\":\"stored\"}},\"observations\":[]}}", .{ &k, field });
        try std.testing.expect(parse(arena, bytes, k) == null);
    }
}
