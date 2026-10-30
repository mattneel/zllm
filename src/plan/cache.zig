//! The plan cache (spec §7): what load-time tuning found, so a later load
//! with the same GPU, model and kernels skips the sweep. The key is a
//! SHA-256 over everything that could change the outcome: the adapter
//! (vendor, architecture, device, its description, which carries the driver
//! version on most backends, and its backend), the runtime (the browser, or
//! Dawn), the device limits the plan uses, the model's GGUF header and
//! tensor index, the zllm version, and every kernel's source (the template
//! set). The cache holds the tuned variants only: the WGSL is rendered from
//! them and the templates, which the key covers.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const kernels = @import("../kernels/kernels.zig");
const Tuning = @import("common.zig").Tuning;
const version = @import("../version.zig").zllm;

pub const Key = [std.crypto.hash.sha2.Sha256.digest_length]u8;

pub const KeyInputs = struct {
    /// Vendor, architecture, device, description, backend.
    adapter: []const []const u8,
    /// The browser's user agent, or the native host's name.
    runtime: []const u8,
    limits: []const u64,
    /// The GGUF file up to its tensor data: header, metadata, tensor index.
    model_header: []const u8,
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
    Feed.bytes(&h, "zllm plan cache 1");
    Feed.bytes(&h, version);
    for (in.adapter) |s| Feed.bytes(&h, s);
    Feed.bytes(&h, in.runtime);
    Feed.bytes(&h, std.mem.sliceAsBytes(in.limits));
    Feed.bytes(&h, in.model_header);
    for (std.enums.values(kernels.Kernel)) |k| Feed.bytes(&h, k.source());
    var out: Key = undefined;
    h.final(&out);
    return out;
}

const Json = struct {
    key: []const u8,
    tuning: []const struct {
        type: []const u8,
        n: u32,
        k: u32,
        wg: u32,
        lanes: u32,
        rows: u32,
    },
};

/// The cache file's contents for `tuning` under `k`.
pub fn serialize(gpa: Allocator, k: Key, tuning: Tuning, out: *std.ArrayList(u8)) Allocator.Error!void {
    var aw: std.Io.Writer.Allocating = .fromArrayList(gpa, out);
    defer out.* = aw.toArrayList();
    const w = &aw.writer;
    w.print("{{\"key\":\"{x}\",\"tuning\":[", .{&k}) catch return error.OutOfMemory;
    for (tuning.entries, 0..) |e, i| {
        w.print("{s}\n{{\"type\":\"{t}\",\"n\":{d},\"k\":{d},\"wg\":{d},\"lanes\":{d},\"rows\":{d}}}", .{
            if (i == 0) "" else ",", e.type, e.n, e.k, e.variant.wg, e.variant.lanes, e.variant.rows,
        }) catch return error.OutOfMemory;
    }
    w.writeAll("]}\n") catch return error.OutOfMemory;
}

/// The tuning stored in `bytes` if it was made under `k`; null for another
/// key, or anything malformed (a variant not among the candidates, an
/// unknown type), which a fresh sweep then replaces.
pub fn parse(arena: Allocator, bytes: []const u8, k: Key) ?Tuning {
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
    return .{ .entries = entries };
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
    try serialize(gpa, k, .{ .entries = &entries }, &bytes);
    const back = parse(arena, bytes.items, k).?;
    try std.testing.expectEqualSlices(Tuning.Entry, &entries, back.entries);

    var other = inputs;
    other.adapter = &.{ "nvidia", "blackwell", "RTX", "driver 2", "d3d12" };
    try std.testing.expect(!std.mem.eql(u8, &k, &key(other)));
    try std.testing.expect(parse(arena, bytes.items, key(other)) == null);
}

test "malformed caches are ignored" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const k = key(.{ .adapter = &.{}, .runtime = "dawn", .limits = &.{}, .model_header = "" });
    var hex: [64]u8 = undefined;
    _ = try std.fmt.bufPrint(&hex, "{x}", .{&k});
    const bad_variant = try std.fmt.allocPrint(arena, "{{\"key\":\"{s}\",\"tuning\":[{{\"type\":\"q3_k\",\"n\":1,\"k\":256,\"wg\":64,\"lanes\":8,\"rows\":1}}]}}", .{&hex});
    try std.testing.expect(parse(arena, bad_variant, k) == null);
    const bad_type = try std.fmt.allocPrint(arena, "{{\"key\":\"{s}\",\"tuning\":[{{\"type\":\"bf16\",\"n\":1,\"k\":256,\"wg\":256,\"lanes\":8,\"rows\":1}}]}}", .{&hex});
    try std.testing.expect(parse(arena, bad_type, k) == null);
    try std.testing.expect(parse(arena, "not json", k) == null);
    const empty = try std.fmt.allocPrint(arena, "{{\"key\":\"{s}\",\"tuning\":[]}}", .{&hex});
    try std.testing.expectEqual(@as(usize, 0), parse(arena, empty, k).?.entries.len);
}
