//! WGSL compute kernels and how to launch them through the command stream.
//!
//! Each kernel is a `<name>.wgsl` file with one entry point, `main`, and its
//! parameters in a uniform `Params` struct at binding 0. The structs below
//! mirror those, field for field, padded to 16 bytes. Weights stay bf16 on
//! the GPU (two per u32); activations and accumulation are f32.
//!
//! Kernels whose work depends on the step (its token count `m`, its first
//! position `pos0`, the sampler's draw `u`) read them from a `Step` uniform,
//! their last binding, so that one prebuilt bind group serves every step.
//! Kernels that treat rows independently take capacity-sized parameters
//! instead and are dispatched over the step's `m` rows only.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cmd = @import("../gpu/cmd.zig");
const ops = @import("../ref/ops.zig");

pub const Kernel = enum {
    rmsnorm,
    matmul_bf16,
    matvec_bf16,
    silu_mul,
    embed_bf16,
    rope_append,
    attention,
    attn_split,
    attn_combine,
    delta_prep,
    delta_rule,
    gated_norm,
    mark_tokens,
    sample,

    pub fn source(k: Kernel) []const u8 {
        return switch (k) {
            inline else => |tag| @embedFile(@tagName(tag) ++ ".wgsl"),
        };
    }
};

/// Matrix multiply tile sizes (matmul_bf16.wgsl: TN outputs × TT tokens).
pub const matmul_tile_n = 64;
pub const matmul_tile_t = 16;
pub const matmul_tile_k = 32;
/// Steps of at most this many tokens (decoding) use the decode kernels:
/// `matvec_bf16` and split attention.
pub const decode_max_rows = 4;
/// Rows per `matvec_bf16` workgroup.
pub const matvec_rows = 8;
/// Widest input `matvec_bf16` stages in workgroup memory, which its RMSNorm
/// prologue needs.
pub const matvec_max_staged = 3584;
/// Widest head `attention` and `delta_rule` handle.
pub const max_head_dim = 256;
/// State columns per `delta_rule` workgroup.
pub const delta_cols = 32;
/// Cache positions per `attn_split` workgroup, and the largest GQA group.
pub const attn_block = 64;
pub const attn_max_group = 8;
/// Widest DeltaNet head `delta_prep` normalizes in one workgroup.
pub const max_delta_head = 256;
/// Sampler limits: logit-bias entries and stop tokens per request.
pub const max_bias = 1024;
pub const max_stops = 16;
/// Tokens per workgroup in the sampler's vocabulary-wide phases.
pub const sample_span = 1024;
/// Most candidates the sampler sorts in workgroup memory (top-k ≤ this).
pub const sample_cap = 1024;

pub const params = struct {
    /// Per-step values, shared by every launch of a step.
    pub const Step = extern struct { m: u32, pos0: u32, u: f32, _pad: u32 = 0 };
    pub const RmsNorm = extern struct { rows: u32, dim: u32, eps: f32, _pad: u32 = 0 };
    pub const MatmulBf16 = extern struct {
        rows: u32,
        n: u32,
        k: u32,
        col0: u32,
        out_stride: u32,
        /// out += result (a residual add).
        accumulate: u32 = 0,
        _pad: [2]u32 = @splat(0),
    };
    pub const MatvecBf16 = extern struct {
        n: u32,
        k: u32,
        col0: u32,
        out_stride: u32,
        eps: f32 = 0,
        flags: Flags = .{},
        /// The up rows' first row, with `silu_pair`.
        half: u32 = 0,
        _pad: u32 = 0,

        pub const Flags = packed struct(u32) {
            /// RMSNorm the input first, with `eps` and the weights bound at 4.
            norm: bool = false,
            /// out += result.
            accumulate: bool = false,
            /// out[j] = silu(row j) · (row half + j).
            silu_pair: bool = false,
            _: u29 = 0,
        };
    };
    pub const SiluMul = extern struct { count: u32, width: u32, _pad: [2]u32 = @splat(0) };
    pub const EmbedBf16 = extern struct { dim: u32, row0: u32, chunk_rows: u32, _pad: u32 = 0 };
    pub const RopeAppend = extern struct {
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        n_rot: u32,
        row: u32,
        k_off: u32,
        v_off: u32,
        f16: u32,
        eps: f32,
        _pad: [3]u32 = @splat(0),
    };
    pub const Attention = extern struct {
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        f16: u32,
        scale: f32,
        q_row: u32,
        _pad: [2]u32 = @splat(0),
    };
    /// `attn_split` and `attn_combine`.
    pub const AttnDecode = extern struct {
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        f16: u32,
        scale: f32,
        max_blocks: u32,
        q_row: u32,
        _pad: u32 = 0,
    };
    pub const DeltaPrep = extern struct {
        channels: u32,
        taps: u32,
        key_channels: u32,
        head: u32,
        row: u32,
        v_heads: u32,
        beta_off: u32,
        alpha_off: u32,
        eps: f32,
        _pad: [3]u32 = @splat(0),
    };
    pub const DeltaRule = extern struct {
        k_heads: u32,
        v_heads: u32,
        dim: u32,
        conv_stride: u32,
        taps: u32,
        row: u32,
        _pad: [2]u32 = @splat(0),
    };
    /// Which phase of `sample` a launch runs.
    pub const SamplePhase = extern struct { phase: u32, round: u32 = 0, _pad: [2]u32 = @splat(0) };
    pub const GatedNorm = extern struct {
        v_heads: u32,
        dim: u32,
        eps: f32,
        z_row: u32,
        z_off: u32,
        _pad: [3]u32 = @splat(0),
    };
    /// Per-request sampler parameters (`sample` and `mark_tokens`).
    pub const Sample = extern struct {
        vocab: u32,
        n_bias: u32,
        n_stop: u32,
        greedy: u32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        repetition: f32,
        presence: f32,
        frequency: f32,
        n_prompt: u32,
        stops: [max_stops]u32,
    };
};

/// The phases of `sample.wgsl`, in its numbering.
pub const SamplePhase = enum(u32) {
    prep,
    reduce,
    hist,
    digit,
    gather,
    final,
    mass,
    pick,
    bisect_init,
    bisect_mass,
    bisect_step,

    /// Whether the phase runs one workgroup per `sample_span` tokens.
    pub fn wide(ph: SamplePhase) bool {
        return switch (ph) {
            .prep, .hist, .gather, .mass, .bisect_mass => true,
            else => false,
        };
    }
};

pub const SampleStep = struct { phase: SamplePhase, round: u32 = 0 };
/// Bisection rounds for top-p without a small top-k: one per key bit.
pub const sample_bisect_rounds = 32;
pub const max_sample_steps = 2 + 2 * 4 + 4 + 2 * sample_bisect_rounds;

/// The phases one draw runs for a request's parameters.
pub fn sampleSteps(s: params.Sample, out: *[max_sample_steps]SampleStep) []SampleStep {
    var n: usize = 0;
    const add = struct {
        fn f(o: *[max_sample_steps]SampleStep, i: *usize, ph: SamplePhase, round: u32) void {
            o[i.*] = .{ .phase = ph, .round = round };
            i.* += 1;
        }
    }.f;
    add(out, &n, .prep, 0);
    add(out, &n, .reduce, 0);
    if (s.greedy != 0) return out[0..n];
    const top_k = s.top_k > 0;
    if (top_k) for (0..4) |r| {
        add(out, &n, .hist, @intCast(r));
        add(out, &n, .digit, @intCast(r));
    };
    if (s.top_p < 1 and top_k and s.top_k <= sample_cap) {
        add(out, &n, .gather, 0);
        add(out, &n, .final, 0);
        return out[0..n];
    }
    if (s.top_p < 1) {
        add(out, &n, .mass, 0);
        add(out, &n, .bisect_init, 0);
        for (0..sample_bisect_rounds) |_| {
            add(out, &n, .bisect_mass, 0);
            add(out, &n, .bisect_step, 0);
        }
    }
    add(out, &n, .mass, 0);
    add(out, &n, .pick, 0);
    return out[0..n];
}

/// Workgroups of a vocabulary-wide sampler phase.
pub fn sampleGroups(vocab: u32) u32 {
    return std.math.divCeil(u32, vocab, sample_span) catch unreachable;
}

/// Words of the sampler's state buffer; the drawn token and its stop flag
/// are the first two.
pub fn sampleStateWords(vocab: u32) u32 {
    return 16 + 256 + 2 * sampleGroups(vocab) + 2 * sample_cap;
}

/// RoPE (cos, sin) for positions [0, positions) and the n_rot/2 rotation
/// pairs, as `rope_append` reads it. Same formula as the CPU reference.
pub fn ropeTable(gpa: Allocator, positions: usize, n_rot: usize, base: f32) Allocator.Error![]f32 {
    const half = n_rot / 2;
    const table = try gpa.alloc(f32, positions * half * 2);
    for (0..positions) |pos| for (0..half) |i| {
        const cs = ops.ropeCosSin(@intCast(pos), i, n_rot, base);
        table[(pos * half + i) * 2 ..][0..2].* = cs;
    };
    return table;
}

/// One pipeline per kernel.
pub const Set = struct {
    pipelines: std.EnumArray(Kernel, cmd.Pipeline),

    /// Records shader and pipeline creation for every kernel.
    pub fn create(gpa: Allocator, rec: *cmd.Recorder) Allocator.Error!Set {
        var set: Set = .{ .pipelines = undefined };
        for (std.enums.values(Kernel)) |k| {
            const shader = try rec.createShader(gpa, k.source());
            set.pipelines.set(k, try rec.createPipeline(gpa, shader));
        }
        return set;
    }
};

/// A buffer range bound at binding 1, 2, ... in kernel order.
pub const Arg = struct {
    buffer: cmd.Buffer,
    offset: u64 = 0,
    size: u64,
};

/// Group 0 for a kernel: `uniform` at binding 0, then `args` at 1, 2, ...
fn bindGroup(gpa: Allocator, rec: *cmd.Recorder, pipeline: cmd.Pipeline, uniform: Arg, args: []const Arg) Allocator.Error!cmd.BindGroup {
    var entries: [16]cmd.Binding = undefined;
    entries[0] = .{ .binding = 0, .buffer = uniform.buffer, .offset = uniform.offset, .size = uniform.size };
    for (args, 1..) |a, i| entries[i] = .{ .binding = @intCast(i), .buffer = a.buffer, .offset = a.offset, .size = a.size };
    return rec.createBindGroup(gpa, pipeline, 0, entries[0 .. args.len + 1]);
}

/// Records kernel launches. Parameter blocks are copied into `arena`, which
/// must outlive execution of the recorded batch.
pub const Launcher = struct {
    gpa: Allocator,
    arena: Allocator,
    rec: *cmd.Recorder,
    set: *const Set,

    pub fn launch(l: Launcher, kernel: Kernel, p: anytype, args: []const Arg, groups: [3]u32) Allocator.Error!void {
        const P = @TypeOf(p);
        comptime std.debug.assert(@sizeOf(P) % 16 == 0);
        const bytes = try l.arena.dupe(u8, std.mem.asBytes(&p));
        const ub = try l.rec.createBuffer(l.gpa, @sizeOf(P), .{ .uniform = true, .copy_dst = true });
        try l.rec.writeBuffer(l.gpa, ub, 0, bytes);
        try l.launchWith(kernel, .{ .buffer = ub, .size = @sizeOf(P) }, args, groups);
    }

    /// A launch whose binding 0 is the caller's `uniform`.
    pub fn launchWith(l: Launcher, kernel: Kernel, uniform: Arg, args: []const Arg, groups: [3]u32) Allocator.Error!void {
        const pipeline = l.set.pipelines.get(kernel);
        const bg = try bindGroup(l.gpa, l.rec, pipeline, uniform, args);
        try l.rec.dispatch(l.gpa, pipeline, groups, &.{bg});
    }

    /// Workgroup grid for `count` threads at 256 per group, folded into y
    /// past the 65535-per-dimension limit.
    pub fn elementwise(count: usize) [3]u32 {
        const groups = std.math.divCeil(usize, count, 256) catch unreachable;
        if (groups <= 65535) return .{ @intCast(groups), 1, 1 };
        return .{ 65535, @intCast(std.math.divCeil(usize, groups, 65535) catch unreachable), 1 };
    }
};

/// What a launch covers in a step: its rows (tokens), and the cache
/// positions the step's last token attends to.
pub const Extent = struct { rows: u32, keys: u32 = 0 };

/// How a prebuilt launch's workgroup grid follows the step's extent.
pub const Grid = union(enum) {
    fixed: [3]u32,
    /// (rows, 1, 1): one workgroup per row.
    rows_x,
    /// (x, rows, 1)
    rows_y: u32,
    /// (x, ⌈rows / tile⌉, 1)
    tiles_y: struct { x: u32, tile: u32 },
    /// One thread per element of rows × width (`Launcher.elementwise`).
    elems: u32,
    /// (x, ⌈keys / block⌉, rows): blocks of cache positions per token.
    key_blocks: struct { x: u32, block: u32 },

    pub fn resolve(g: Grid, e: Extent) [3]u32 {
        const rows = e.rows;
        return switch (g) {
            .fixed => |f| f,
            .rows_x => .{ rows, 1, 1 },
            .rows_y => |x| .{ x, rows, 1 },
            .tiles_y => |t| .{ t.x, std.math.divCeil(u32, rows, t.tile) catch unreachable, 1 },
            .elems => |w| Launcher.elementwise(@as(usize, rows) * w),
            .key_blocks => |k| .{ k.x, std.math.divCeil(u32, e.keys, k.block) catch unreachable, rows },
        };
    }
};

/// A launch whose bind group (parameters and buffers) is built once; each
/// step only dispatches it.
pub const Launch = struct {
    kernel: Kernel,
    pipeline: cmd.Pipeline,
    bind_group: cmd.BindGroup,
    grid: Grid,

    pub fn record(l: Launch, gpa: Allocator, rec: *cmd.Recorder, e: Extent) Allocator.Error!void {
        try rec.dispatch(gpa, l.pipeline, l.grid.resolve(e), &.{l.bind_group});
    }
};

/// Builds `Launch`es. Parameter blocks are packed into 64 KiB uniform
/// buffers at 256-byte slots (the largest minUniformBufferOffsetAlignment
/// WebGPU allows), uploaded by `finish`.
pub const Builder = struct {
    gpa: Allocator,
    /// Owns the parameter pages; must outlive execution of the recorded batch.
    arena: Allocator,
    rec: *cmd.Recorder,
    set: *const Set,
    pages: std.ArrayList(Page) = .empty,

    const page_size = 64 * 1024;
    const slot = 256;
    const Page = struct { buffer: cmd.Buffer, data: []u8, used: usize };

    pub fn deinit(b: *Builder) void {
        b.pages.deinit(b.gpa);
    }

    /// A launch with its own parameters at binding 0.
    pub fn add(b: *Builder, kernel: Kernel, p: anytype, args: []const Arg, grid: Grid) Allocator.Error!Launch {
        const P = @TypeOf(p);
        comptime std.debug.assert(@sizeOf(P) % 16 == 0 and @sizeOf(P) <= slot);
        if (b.pages.items.len == 0 or b.pages.items[b.pages.items.len - 1].used == page_size) {
            const data = try b.arena.alloc(u8, page_size);
            @memset(data, 0);
            const buffer = try b.rec.createBuffer(b.gpa, page_size, .{ .uniform = true, .copy_dst = true });
            try b.pages.append(b.gpa, .{ .buffer = buffer, .data = data, .used = 0 });
        }
        const page = &b.pages.items[b.pages.items.len - 1];
        @memcpy(page.data[page.used..][0..@sizeOf(P)], std.mem.asBytes(&p));
        const uniform: Arg = .{ .buffer = page.buffer, .offset = page.used, .size = @sizeOf(P) };
        page.used += slot;
        return b.addWith(kernel, uniform, args, grid);
    }

    /// A launch whose binding 0 is `uniform`, a buffer the caller rewrites.
    pub fn addWith(b: *Builder, kernel: Kernel, uniform: Arg, args: []const Arg, grid: Grid) Allocator.Error!Launch {
        const pipeline = b.set.pipelines.get(kernel);
        return .{ .kernel = kernel, .pipeline = pipeline, .bind_group = try bindGroup(b.gpa, b.rec, pipeline, uniform, args), .grid = grid };
    }

    /// Records the upload of every parameter page.
    pub fn finish(b: *Builder) Allocator.Error!void {
        for (b.pages.items) |page| try b.rec.writeBuffer(b.gpa, page.buffer, 0, page.data[0..page.used]);
    }
};
