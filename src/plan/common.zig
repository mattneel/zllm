//! The parts of a step program that do not depend on the model: weight
//! upload, buffer allocation, the multiplies' launches, and the sampler's.

const std = @import("std");
const Allocator = std.mem.Allocator;
const gguf = @import("../gguf/gguf.zig");
const cmd = @import("../gpu/cmd.zig");
const kernels = @import("../kernels/kernels.zig");
const Arg = kernels.Arg;
const P = kernels.params;

pub const BuildError = kernels.Set.Error || error{ BadConfig, TooLarge };

pub const UploadError = Allocator.Error || error{ MissingTensor, BadShape, UnsupportedTensorType, TensorTooLarge };

pub const Limits = struct {
    /// maxStorageBufferBindingSize
    max_binding: u64,
    /// minStorageBufferOffsetAlignment
    offset_align: u64,
};

/// A weight matrix of `n` rows × `k` (row j = output j's weights), split by
/// rows into chunks.
pub const Matrix = struct {
    n: u32,
    k: u32,
    chunks: []const Chunk,

    /// `type`: BF16, F16 or a quantized type (kernels.Kernel.weight).
    pub const Chunk = struct { arg: Arg, row0: u32, rows: u32, type: gguf.TensorType };
};

pub const KvFormat = enum(u32) { f32 = 0, f16 = 1 };

/// The decode-multiply variant chosen for each weight shape.
pub const Tuning = struct {
    entries: []const Entry = &.{},

    pub const Entry = struct { type: gguf.TensorType, n: u32, k: u32, variant: kernels.MatvecVariant };

    pub fn matvec(t: Tuning, ty: gguf.TensorType, n: u32, k: u32) kernels.MatvecVariant {
        for (t.entries) |e| if (e.type == ty and e.n == n and e.k == k) return e.variant;
        return .{};
    }
};

/// Completion token of every read a Program records.
pub const read_token = 0xffff_fffe;

pub const Upload = struct {
    arena: Allocator,
    gpa: Allocator,
    rec: *cmd.Recorder,
    /// For `repad`.
    set: *const kernels.Set,
    file: gguf.File,
    limits: Limits,

    pub fn info(u: *Upload, comptime fmt: []const u8, layer: ?usize, shape: []const u64, t: gguf.TensorType) !*const gguf.TensorInfo {
        var buf: [96]u8 = undefined;
        const name = if (comptime std.mem.indexOf(u8, fmt, "{d}") != null)
            std.fmt.bufPrint(&buf, fmt, .{layer.?}) catch unreachable
        else
            fmt;
        const ti = u.file.tensor(name) orelse return error.MissingTensor;
        if (!std.mem.eql(u64, ti.shape(), shape)) return error.BadShape;
        if (ti.type != t) return error.UnsupportedTensorType;
        return ti;
    }

    pub fn has(u: *const Upload, comptime fmt: []const u8, layer: usize) bool {
        var buf: [96]u8 = undefined;
        return u.file.tensor(std.fmt.bufPrint(&buf, fmt, .{layer}) catch unreachable) != null;
    }

    /// A matrix weight: BF16, F16, or a type the quantized kernels decode.
    pub fn weightInfo(u: *Upload, comptime fmt: []const u8, layer: ?usize, shape: []const u64) !*const gguf.TensorInfo {
        var buf: [96]u8 = undefined;
        const name = if (comptime std.mem.indexOf(u8, fmt, "{d}") != null)
            std.fmt.bufPrint(&buf, fmt, .{layer.?}) catch unreachable
        else
            fmt;
        const ti = u.file.tensor(name) orelse return error.MissingTensor;
        if (!std.mem.eql(u64, ti.shape(), shape)) return error.BadShape;
        if (kernels.Kernel.weight(.matvec, ti.type) == null) return error.UnsupportedTensorType;
        return ti;
    }

    /// Where `ti`'s bytes start in the file.
    pub fn fileOffset(u: *const Upload, ti: *const gguf.TensorInfo) u64 {
        return u.file.data_offset + ti.offset;
    }

    /// An f32 tensor of any shape.
    pub fn tensor(u: *Upload, comptime fmt: []const u8, layer: ?usize, shape: []const u64) !Arg {
        const ti = try u.info(fmt, layer, shape, .f32);
        const size = ti.byteSize();
        if (size > u.limits.max_binding) return error.TensorTooLarge;
        const buf = try u.rec.createBuffer(u.gpa, size, .{ .storage = true, .copy_dst = true });
        try u.rec.writeFile(u.gpa, buf, 0, u.fileOffset(ti), size);
        return .{ .buffer = buf, .size = size };
    }

    pub fn vector(u: *Upload, comptime fmt: []const u8, layer: ?usize, n: u64) !Arg {
        return u.tensor(fmt, layer, &.{n});
    }

    /// A tensor of type `t` as stored, in one buffer (rounded up to whole
    /// words): weights only their own kernels read, as the Gyro experts.
    pub fn raw(u: *Upload, comptime fmt: []const u8, layer: ?usize, shape: []const u64, t: gguf.TensorType) !Arg {
        const ti = try u.info(fmt, layer, shape, t);
        const size = ti.byteSize();
        if (size > u.limits.max_binding) return error.TensorTooLarge;
        const buf = try u.rec.createBuffer(u.gpa, std.mem.alignForward(u64, size, 4), .{ .storage = true, .copy_dst = true });
        try u.rec.writeFile(u.gpa, buf, 0, u.fileOffset(ti), size);
        return .{ .buffer = buf, .size = std.mem.alignForward(u64, size, 4) };
    }

    /// A GGML {k, n} weight (half-precision or quantized), split into chunks of whole
    /// rows.
    pub fn matrix(u: *Upload, comptime fmt: []const u8, layer: ?usize, k: u64, n: u64) !Matrix {
        return u.stacked(layer, k, &.{.{ fmt, n }});
    }

    /// GGML {k, n_i} weights sharing an input, stacked by rows into one
    /// matrix (part i's rows follow part i − 1's) and split into chunks of
    /// whole rows. Consecutive parts of one weight type share chunks, each
    /// part's bytes written at their place; a change of type starts new
    /// chunks, since a chunk is decoded by one kernel.
    pub fn stacked(u: *Upload, layer: ?usize, k: u64, parts: anytype) !Matrix {
        const Span = struct { offset: u64, rows: u64, type: gguf.TensorType };
        var spans: [parts.len]Span = undefined;
        var n: u64 = 0;
        inline for (parts, &spans) |part, *span| {
            const ti = try u.weightInfo(part[0], layer, &.{ k, part[1] });
            span.* = .{ .offset = u.fileOffset(ti), .rows = part[1], .type = ti.type };
            n += part[1];
        }
        var chunks: std.ArrayList(Matrix.Chunk) = .empty;
        var first: usize = 0;
        var run_row0: u64 = 0;
        while (first < spans.len) {
            // A run of parts of one type.
            var end = first + 1;
            while (end < spans.len and spans[end].type == spans[first].type) end += 1;
            var run_rows: u64 = 0;
            for (spans[first..end]) |s| run_rows += s.rows;
            try u.chunkRun(&chunks, k, spans[first].type, spans[first..end], run_row0, run_rows);
            run_row0 += run_rows;
            first = end;
        }
        return .{ .n = @intCast(n), .k = @intCast(k), .chunks = chunks.items };
    }

    /// Chunks of a run of parts of type `t` covering matrix rows
    /// [run_row0, run_row0 + run_rows). Quantized blocks whose size is not a
    /// multiple of 16 (Q3_K, Q6_K) are laid out padded
    /// (kernels.blockStride): the rows go to a staging buffer first, then
    /// `repad` places each block, and the staging buffer is dropped after
    /// its own submit, so at most one chunk's worth is ever extra.
    fn chunkRun(u: *Upload, chunks: *std.ArrayList(Matrix.Chunk), k: u64, t: gguf.TensorType, spans: anytype, run_row0: u64, run_rows: u64) !void {
        const l = t.layout().?;
        const blocks_per_row = k / l.block_elems;
        const file_row = blocks_per_row * l.block_bytes;
        const stride: u64 = kernels.blockStride(t);
        const gpu_row = blocks_per_row * stride;
        const repad = stride != l.block_bytes;
        const rows_per_chunk = @min(run_rows, u.limits.max_binding / gpu_row);
        if (rows_per_chunk == 0) return error.TensorTooLarge;
        var row0: u64 = 0;
        while (row0 < run_rows) : (row0 += rows_per_chunk) {
            const rows = @min(rows_per_chunk, run_rows - row0);
            // Storage bindings are whole words; the quantized kernels read
            // 16-byte vectors.
            const size = std.mem.alignForward(u64, rows * gpu_row, 16);
            const buf = try u.rec.createBuffer(u.gpa, size, .{ .storage = true, .copy_dst = true });
            const raw_size = std.mem.alignForward(u64, rows * file_row, 4);
            const staging = if (repad) try u.rec.createBuffer(u.gpa, raw_size, .{ .storage = true, .copy_dst = true }) else buf;
            // Copy the parts' rows that fall in [row0, row0 + rows) of the run.
            var part_row0: u64 = 0;
            for (spans) |span| {
                const lo = @max(row0, part_row0);
                const hi = @min(row0 + rows, part_row0 + span.rows);
                if (lo < hi) {
                    try u.rec.writeFile(u.gpa, staging, (lo - row0) * file_row, span.offset + (lo - part_row0) * file_row, (hi - lo) * file_row);
                }
                part_row0 += span.rows;
            }
            if (repad) {
                const blocks = rows * blocks_per_row;
                const launcher: kernels.Launcher = .{ .gpa = u.gpa, .arena = u.arena, .rec = u.rec, .set = u.set };
                try launcher.layout(t, .{ .buffer = staging, .size = raw_size }, .{ .buffer = buf, .size = size }, blocks);
                try u.rec.submit(u.gpa, 0);
                try u.rec.destroy(u.gpa, .buffer, @backingInt(staging));
            }
            try chunks.append(u.arena, .{ .arg = .{ .buffer = buf, .size = size }, .row0 = @intCast(run_row0 + row0), .rows = @intCast(rows), .type = t });
        }
    }
};

/// Records buffer creation, checking each fits one binding.
pub const Alloc = struct {
    gpa: Allocator,
    rec: *cmd.Recorder,
    limits: Limits,

    pub fn buffer(a: *Alloc, size: u64, usage: cmd.Usage) BuildError!Arg {
        if (usage.storage and size > a.limits.max_binding) return error.TooLarge;
        return .{ .buffer = try a.rec.createBuffer(a.gpa, size, usage), .size = size };
    }

    /// `count` f32 or u32 values.
    pub fn storage(a: *Alloc, count: u64) BuildError!Arg {
        return a.buffer(count * 4, .{ .storage = true, .copy_src = true, .copy_dst = true });
    }
};

/// The sampler's bindings besides the logits row and the step uniform.
pub const SamplerArgs = struct {
    bias: Arg,
    prompt_mask: Arg,
    counts: Arg,
    work: Arg,
    state: Arg,
    next: Arg,
    params: Arg,

    /// One launch per sampler phase and round, over `logits` with the draw
    /// in `step`.
    pub fn launches(s: SamplerArgs, b: *kernels.Builder, vocab: u32, logits: Arg, step: Arg) Allocator.Error!std.EnumArray(kernels.SamplePhase, [4]kernels.Launch) {
        var out: std.EnumArray(kernels.SamplePhase, [4]kernels.Launch) = undefined;
        for (std.enums.values(kernels.SamplePhase)) |phase| {
            const groups: u32 = if (phase.wide()) kernels.sampleGroups(vocab) else 1;
            const rounds: u32 = if (phase == .hist or phase == .digit) 4 else 1;
            for (out.getPtr(phase), 0..) |*l, r| {
                // Phases with one round reuse round 0's launch.
                if (r >= rounds) {
                    l.* = out.get(phase)[0];
                    continue;
                }
                l.* = try b.add(.sample, P.SamplePhase{ .phase = @backingInt(phase), .round = @intCast(r) }, &.{
                    logits, s.bias, s.prompt_mask, s.counts, s.work, s.state, s.next, step, s.params,
                }, .{ .fixed = .{ groups, 1, 1 } });
            }
        }
        return out;
    }
};

/// Builds a weight's multiply, one launch per chunk, either way.
pub const Mm = struct {
    b: *kernels.Builder,
    arena: Allocator,
    cap: u32,
    eps: f32,
    /// Partial sums of the prefill multiplies split over k
    /// (partialFloats).
    partials: Arg,
    /// The quantized prefill multiplies' input rounded to 8-bit blocks
    /// (quantize_q8), per block size: [0] Q8_K (256), [1] Q8_0 (32).
    act: [2]?Act,
    tuning: Tuning,

    pub const Act = struct { qx: Arg, dx: Arg };

    pub fn actIndex(block: u32) usize {
        return @intFromBool(block == 32);
    }

    /// Scratch for x rounded to `block`-value blocks, for steps of `cap`
    /// tokens, if any quantized weight takes that block.
    pub fn actScratch(a: *Alloc, m: anytype, cap: u64, block: u32) BuildError!?Act {
        const Widest = struct {
            block: u32,
            k: u64 = 0,
            fn add(s: *@This(), w: *const Matrix) void {
                for (w.chunks) |ch| {
                    if (!kernels.isHalf(ch.type) and kernels.actBlock(ch.type) == s.block) s.k = @max(s.k, w.k);
                }
            }
        };
        var widest: Widest = .{ .block = block };
        m.eachMatrix(&widest, Widest.add);
        if (widest.k == 0) return null;
        return .{ .qx = try a.storage(cap * widest.k / 4), .dx = try a.storage(cap * widest.k / block) };
    }

    /// Floats of `partials` for steps of `cap` tokens: the largest split
    /// chunk's slices × rows × cap, at least one.
    pub fn partialFloats(m: anytype, cap: u64) u64 {
        const Most = struct {
            cap: u64,
            floats: u64 = 1,
            fn add(s: *@This(), w: *const Matrix) void {
                for (w.chunks) |ch| {
                    if (kernels.isHalf(ch.type)) continue;
                    const slices = kernels.matmulSlices(ch.rows, w.k);
                    if (slices.count > 1) s.floats = @max(s.floats, @as(u64, slices.count) * ch.rows * s.cap);
                }
            }
        };
        var most: Most = .{ .cap = cap };
        m.eachMatrix(&most, Most.add);
        return most.floats;
    }

    /// What the decode multiply fuses around it.
    pub const Fuse = struct {
        /// RMSNorm the input with these weights first.
        norm: ?Arg = null,
        /// out += result.
        accumulate: bool = false,
        /// The weight is gate | up, each this many rows: out = silu(gate) · up.
        silu_pair: ?u32 = null,
    };

    /// A prefill multiply's input: x as is, or computed from x by the
    /// kernel that would otherwise run first (into `out`).
    pub const Input = union(enum) {
        plain: Arg,
        /// RMSNorm(x) · w.
        rmsnorm: struct { x: Arg, w: Arg, out: Arg },
        /// silu(gate) · up over gu's rows of gate | up, `width` each.
        silu: struct { gu: Arg, width: u32, out: Arg },
    };

    /// out[t][j] (+)= in[t] · w[j] for the rows of a prefill step: tiled.
    /// Quantized chunks take the input rounded to 8-bit blocks
    /// (quantize_q8, once per block size), which also computes it from
    /// `in`, so the f32 input is only produced for BF16 chunks. Chunks with
    /// too few column tiles split over k into `partials`, added up by
    /// matmul_reduce.
    pub fn tiled(mm: Mm, out: Arg, in: Input, w: *const Matrix, accumulate: bool) Allocator.Error![]const kernels.Launch {
        std.debug.assert(w.k % kernels.matmul_tile_k == 0);
        var list: std.ArrayList(kernels.Launch) = .empty;
        var half = false;
        for (w.chunks) |ch| half = half or kernels.isHalf(ch.type);
        const x: Arg = switch (in) {
            .plain => |a| a,
            .rmsnorm => |r| r.out,
            .silu => |s| s.out,
        };
        if (half) switch (in) {
            .plain => {},
            .rmsnorm => |r| try list.append(mm.arena, try mm.b.add(.rmsnorm, P.RmsNorm{ .rows = mm.cap, .dim = w.k, .eps = mm.eps }, &.{ r.x, r.w, r.out }, .rows_x)),
            .silu => |s| try list.append(mm.arena, try mm.b.add(.silu_mul, P.SiluMul{ .count = mm.cap * s.width, .width = s.width }, &.{ s.gu, s.out }, .{ .elems = s.width })),
        };
        var rounded: [2]bool = @splat(false);
        for (w.chunks) |ch| {
            if (kernels.isHalf(ch.type)) continue;
            // quantize_q8 takes 256 values per workgroup.
            std.debug.assert(w.k % 256 == 0);
            const block = kernels.actBlock(ch.type);
            const i = actIndex(block);
            if (rounded[i]) continue;
            rounded[i] = true;
            const act = mm.act[i].?;
            const q: P.QuantizeQ8, const src: Arg, const nw: Arg = switch (in) {
                .plain => |a| .{ .{ .k = w.k, .block = block }, a, a },
                .rmsnorm => |r| .{ .{ .k = w.k, .block = block, .mode = 1, .eps = mm.eps }, r.x, r.w },
                .silu => |s| .{ .{ .k = w.k, .block = block, .mode = 2 }, s.gu, s.gu },
            };
            try list.append(mm.arena, try mm.b.add(.quantize_q8, q, &.{ src, act.qx, act.dx, nw }, .{ .rows_y = w.k / 256 }));
        }
        for (w.chunks) |ch| {
            const tiles = kernels.matmulTiles(ch.type);
            const x_tiles = std.math.divCeil(u32, ch.rows, tiles[0]) catch unreachable;
            if (kernels.isHalf(ch.type)) {
                try list.append(mm.arena, try mm.b.add(weightKernel(.matmul, ch.type), P.MatmulBf16{
                    .rows = mm.cap,
                    .n = ch.rows,
                    .k = w.k,
                    .col0 = ch.row0,
                    .out_stride = w.n,
                    .accumulate = @intFromBool(accumulate),
                }, &.{ x, ch.arg, out }, .{ .tiles_y = .{ .x = x_tiles, .tile = tiles[1] } }));
                continue;
            }
            const s = kernels.matmulSlices(ch.rows, w.k);
            const kernel = weightKernel(.matmul, ch.type);
            const act = mm.act[actIndex(kernels.actBlock(ch.type))].?;
            if (s.count == 1) {
                try list.append(mm.arena, try mm.b.add(kernel, P.MatmulQuant{
                    .rows = mm.cap,
                    .n = ch.rows,
                    .k = w.k,
                    .col0 = ch.row0,
                    .out_stride = w.n,
                    .accumulate = @intFromBool(accumulate),
                    .k_slice = w.k,
                }, &.{ act.qx, ch.arg, out, act.dx }, .{ .tiles_y = .{ .x = x_tiles, .tile = tiles[1] } }));
                continue;
            }
            const stride = mm.cap * ch.rows;
            std.debug.assert(@as(u64, s.count) * stride * 4 <= mm.partials.size);
            try list.append(mm.arena, try mm.b.add(kernel, P.MatmulQuant{
                .rows = mm.cap,
                .n = ch.rows,
                .k = w.k,
                .col0 = 0,
                .out_stride = ch.rows,
                .k_slice = s.k_slice,
                .slice_stride = stride,
            }, &.{ act.qx, ch.arg, mm.partials, act.dx }, .{ .tiles_y = .{ .x = x_tiles, .tile = tiles[1], .z = s.count } }));
            try list.append(mm.arena, try mm.b.add(.matmul_reduce, P.MatmulReduce{
                .rows = mm.cap,
                .n = ch.rows,
                .slices = s.count,
                .col0 = ch.row0,
                .out_stride = w.n,
                .accumulate = @intFromBool(accumulate),
                .slice_stride = stride,
            }, &.{ mm.partials, out }, .{ .elems = ch.rows }));
        }
        return list.items;
    }

    /// The same for a decode step, matrix-vector, with `fuse` folded in.
    pub fn vector(mm: Mm, out: Arg, x: Arg, w: *const Matrix, fuse: Fuse) BuildError![]const kernels.Launch {
        if (fuse.silu_pair) |half| {
            // Both halves of a row pair must be in one binding.
            if (w.chunks.len != 1) return error.TooLarge;
            std.debug.assert(w.n == 2 * half);
        }
        const flags: P.MatvecBf16.Flags = .{ .norm = fuse.norm != null, .accumulate = fuse.accumulate, .silu_pair = fuse.silu_pair != null };
        const list = try mm.arena.alloc(kernels.Launch, w.chunks.len);
        for (w.chunks, list) |ch, *l| {
            // The half-precision kernel normalizes only staged inputs.
            if (kernels.isHalf(ch.type) and fuse.norm != null and w.k > kernels.matvec_max_staged) return error.BadConfig;
            const n = fuse.silu_pair orelse ch.rows;
            const kernel = weightKernel(.matvec, ch.type);
            const params: P.MatvecBf16 = .{
                .n = n,
                .k = w.k,
                .col0 = if (fuse.silu_pair != null) 0 else ch.row0,
                .out_stride = if (fuse.silu_pair) |half| half else w.n,
                .eps = mm.eps,
                .flags = flags,
                .half = fuse.silu_pair orelse 0,
            };
            const args = [_]Arg{
                x,                       ch.arg, out,
                // Unread without the norm; the weight is a read-only stand-in.
                fuse.norm orelse ch.arg,
            };
            if (kernels.isHalf(ch.type)) {
                l.* = try mm.b.add(kernel, params, &args, .{ .rows_y = std.math.divCeil(u32, n, kernels.matvec_rows) catch unreachable });
            } else {
                const v = mm.tuning.matvec(ch.type, w.n, w.k);
                l.* = try mm.b.addMatvec(kernel, v, params, &args, .{ .rows_y = std.math.divCeil(u32, n, v.groupRows()) catch unreachable });
            }
        }
        return list;
    }
};

/// The kernel for operation `op` on a weight chunk of type `t` (BF16, F16, or a
/// quantized type Upload.weightInfo accepted).
pub fn weightKernel(comptime op: kernels.QuantOp, t: gguf.TensorType) kernels.Kernel {
    return kernels.Kernel.weight(op, t).?;
}

pub fn concat(arena: Allocator, parts: []const []const kernels.Launch) Allocator.Error![]const kernels.Launch {
    return std.mem.concat(arena, kernels.Launch, parts);
}
