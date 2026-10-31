//! Warm, encode-only throughput on independent JSONL corpus records.
const std = @import("std");
const zllm = @import("zllm");
const Io = std.Io;

pub fn run(gpa: std.mem.Allocator, io: Io, tok: *const zllm.tokenizer.Tokenizer, corpus: []const u8, out: *Io.Writer) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const alloc = arena.allocator();
    var texts: std.ArrayList([]const u8) = .empty;
    var corpus_bytes: u64 = 0;
    var lines = std.mem.splitScalar(u8, corpus, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const parsed = try std.json.parseFromSlice(struct { text: []const u8 }, alloc, line, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        try texts.append(alloc, parsed.value.text);
        corpus_bytes += parsed.value.text.len;
    }
    if (texts.items.len == 0 or corpus_bytes == 0) return error.EmptyCorpus;

    var ids: std.ArrayList(zllm.tokenizer.Id) = .empty;
    defer ids.deinit(gpa);
    // Warm the tokenizer and output capacity once before starting the clock.
    const corpus_tokens = try encodePass(gpa, tok, texts.items, &ids);
    var repetitions: u64 = 0;
    var tokens: u64 = 0;
    var elapsed_ns: u64 = 0;
    const start = Io.Timestamp.now(io, .awake);
    while (elapsed_ns < std.time.ns_per_s / 2) {
        tokens += try encodePass(gpa, tok, texts.items, &ids);
        repetitions += 1;
        elapsed_ns = @intCast(start.durationTo(Io.Timestamp.now(io, .awake)).toNanoseconds());
    }
    const bytes = corpus_bytes * repetitions;
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    var json: std.json.Stringify = .{ .writer = out };
    try json.write(.{
        .engine = "zllm",
        .corpus_records = texts.items.len,
        .corpus_bytes = corpus_bytes,
        .corpus_tokens = corpus_tokens,
        .repetitions = repetitions,
        .elapsed_ns = elapsed_ns,
        .bytes = bytes,
        .tokens = tokens,
        .mb_per_second = @as(f64, @floatFromInt(bytes)) / seconds / 1_000_000,
        .tokens_per_second = @as(f64, @floatFromInt(tokens)) / seconds,
    });
    try out.writeByte('\n');
}

fn encodePass(gpa: std.mem.Allocator, tok: *const zllm.tokenizer.Tokenizer, texts: []const []const u8, ids: *std.ArrayList(zllm.tokenizer.Id)) !u64 {
    var count: u64 = 0;
    for (texts) |text| {
        ids.clearRetainingCapacity();
        try tok.encode(gpa, text, ids);
        count += ids.items.len;
    }
    return count;
}
