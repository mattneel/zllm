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
const gguf = @import("../gguf/gguf.zig");
const tmpl = @import("../tmpl/tmpl.zig");
pub const Backend = @import("../backend/backend.zig").Backend;
const agility_kernels = @import("../agility/kernels.zig");
const precision = @import("../precision.zig");

pub const Kernel = enum {
    rmsnorm,
    // Half-precision weights: matvec_half.wgsl, matmul_half.wgsl and
    // embed_half.wgsl, each composed with half/<type>.wgsl.
    matmul_bf16,
    matmul_f16,
    matmul_reduce,
    matvec_bf16,
    matvec_f16,
    silu_mul,
    embed_bf16,
    embed_f16,
    rope_append,
    // attn_split.wgsl with MAX_GROUP set: groups of up to attn_max_group,
    // and the wide variant's up to attn_max_group_wide (more workgroup
    // memory, which costs the narrow groups' prefill ~2%); the sparse one
    // is the wide one over the sparse attention's selection.
    attn_split,
    attn_split_wide,
    attn_split_sparse,
    attn_combine,
    // Qwen3.8-Flash-Next's sparse attention indexer.
    qsa_keys,
    qsa_pool,
    qsa_score,
    qsa_select,
    delta_prep,
    delta_rule,
    gated_norm,
    mark_tokens,
    sample,
    draft,
    gyro_matvec,
    gyro_matmul,
    moe_group,
    moe_logits,
    moe_route,
    rotate128,
    moe_combine,
    hc_act,
    hc_mix,
    hc_combine,
    ple_mix,
    ple_conv,
    repad,
    repack_q3_k,
    repack_q8_0,
    quantize_q8,
    // Quantized weights: matvec_quant.wgsl.hbs (a template, MatvecVariant),
    // matmul_quant.wgsl and embed_quant.wgsl, each composed with its type's
    // decoder.
    matvec_q8_0,
    matvec_q3_k,
    matvec_q4_k,
    matvec_q5_k,
    matvec_q6_k,
    matmul_q8_0,
    matmul_q8_0_float,
    matmul_q3_k,
    matmul_q4_k,
    matmul_q5_k,
    matmul_q6_k,
    embed_q8_0,
    embed_q3_k,
    embed_q4_k,
    embed_q5_k,
    embed_q6_k,

    /// The kernel's WGSL, or for a template (`isTemplate`) its source.
    pub fn source(k: Kernel) []const u8 {
        return switch (k) {
            inline else => |tag| comptime composed(@tagName(tag)),
        };
    }

    /// The kernel's bindings as its WGSL declares them: the contract every
    /// backend's implementation keeps, and what hosts may rely on (Agility
    /// orders dispatches by it).
    pub fn contract(k: Kernel) Contract {
        return switch (k) {
            inline else => |tag| comptime parseContract(composed(@tagName(tag))),
        };
    }

    /// Whether the kernel's source is a template over a MatvecVariant: the
    /// quantized decode multiplies.
    pub fn isTemplate(k: Kernel) bool {
        return switch (k) {
            .matvec_q8_0, .matvec_q3_k, .matvec_q4_k, .matvec_q5_k, .matvec_q6_k => true,
            else => false,
        };
    }

    /// The kernel for an operation on weights of type `t`, if supported.
    pub fn quant(comptime op: QuantOp, t: gguf.TensorType) ?Kernel {
        return switch (t) {
            inline .q8_0, .q3_k, .q4_k, .q5_k, .q6_k => |q| @field(Kernel, @tagName(op) ++ "_" ++ @tagName(q)),
            else => null,
        };
    }

    /// The kernel for operation `op` on weights of type `t`, if supported:
    /// half-precision (BF16, F16) or quantized.
    pub fn weight(comptime op: QuantOp, t: gguf.TensorType) ?Kernel {
        return switch (t) {
            inline .bf16, .f16 => |h| @field(Kernel, @tagName(op) ++ "_" ++ @tagName(h)),
            else => quant(op, t),
        };
    }
};

pub const QuantOp = enum { matvec, matmul, embed };

/// Bytes per block of type `t` as the quantized kernels read it: the GGUF
/// block size, padded to a multiple of 16 where it is not one already
/// (`repad` lays the blocks out), so blocks take 16-byte loads. Q3_K is
/// repacked instead, into eight self-contained 16-byte units
/// (repack_q3_k.wgsl). Q8_0 rows are split instead (rowBytes).
pub fn blockStride(t: gguf.TensorType) u32 {
    return switch (t) {
        .q3_k => 128,
        .q6_k => 224,
        else => t.layout().?.block_bytes,
    };
}

/// Bytes per row of `k` weights of quantized type `t` as the kernels read
/// it (each type's row_stride): blocks at blockStride, or for Q8_0 split
/// rows (repack_q8_0.wgsl), the k values as i8 then the k/32 scales as f16,
/// padded to 16 bytes (exact when k is a multiple of 256).
pub fn rowBytes(t: gguf.TensorType, k: u64) u64 {
    if (t == .q8_0) return std.mem.alignForward(u64, k + k / 16, 16);
    return k / t.layout().?.block_elems * blockStride(t);
}

/// Whether type `t`'s rows need laying out (Launcher.layout) on the GPU,
/// rather than reading as they are in the file.
pub fn needsLayout(t: gguf.TensorType) bool {
    return t == .q8_0 or blockStride(t) != t.layout().?.block_bytes;
}

/// Whether weights of type `t` take the half-precision kernels, which read
/// them as they are and multiply in f32.
pub fn isHalf(t: gguf.TensorType) bool {
    return t == .bf16 or t == .f16;
}

/// A kernel's WGSL: its own file, or for `<op>_<type>` the type's helpers
/// and the operation's body: half/<type>.wgsl and <op>_half.wgsl for BF16
/// and F16; the shared quant helpers, the type's decoder and <op>_quant.wgsl
/// for the quantized types.
/// A kernel's bindings: how many (0 .. count − 1, binding 0 the params
/// block), and which it writes (bit i: binding i is `var<storage,
/// read_write>`); every other binding is only read.
pub const Contract = struct {
    count: u32,
    writes: u32,

    pub fn reads(c: Contract, binding: u32) bool {
        return binding < c.count and c.writes & (@as(u32, 1) << @intCast(binding)) == 0;
    }
};

/// Scans WGSL for `@group(0) @binding(N) var<...>` declarations.
fn parseContract(comptime wgsl: []const u8) Contract {
    @setEvalBranchQuota(1_000_000);
    var c: Contract = .{ .count = 0, .writes = 0 };
    const tag = "@binding(";
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, wgsl, at, tag)) |i| {
        const start = i + tag.len;
        const close = std.mem.indexOfScalarPos(u8, wgsl, start, ')').?;
        const binding = std.fmt.parseInt(u32, wgsl[start..close], 10) catch unreachable;
        const decl_end = std.mem.indexOfScalarPos(u8, wgsl, close, '>').?;
        const decl = wgsl[close..decl_end];
        if (std.mem.indexOf(u8, decl, "read_write") != null) c.writes |= 1 << binding;
        c.count = @max(c.count, binding + 1);
        at = decl_end;
    }
    return c;
}

test "kernel contracts come from the WGSL's binding declarations" {
    // rmsnorm: params, x, w read; y written.
    try std.testing.expectEqual(Contract{ .count = 4, .writes = 0b1000 }, Kernel.rmsnorm.contract());
    // rope_append writes the fused QKV row in place and both caches.
    try std.testing.expectEqual(Contract{ .count = 8, .writes = 0b0110_0010 }, Kernel.rope_append.contract());
    try std.testing.expectEqual(Contract{ .count = 5, .writes = 0b01100 }, Kernel.moe_group.contract());
    try std.testing.expectEqual(Contract{ .count = 8, .writes = 0b0001_0000 }, Kernel.gyro_matmul.contract());
    try std.testing.expectEqual(Contract{ .count = 4, .writes = 0b1000 }, Kernel.matmul_q8_0_float.contract());
    try std.testing.expect(Kernel.matvec_q8_0.contract().reads(1));
    try std.testing.expect(!Kernel.matvec_q8_0.contract().reads(3));
    for (std.enums.values(Kernel)) |kernel| try std.testing.expect(kernel.contract().writes != 0);
}

fn composed(comptime name: []const u8) []const u8 {
    @setEvalBranchQuota(10_000);
    if (std.mem.eql(u8, name, "matmul_q8_0_float")) return @embedFile("matmul_q8_0_float.wgsl");
    for (std.enums.values(QuantOp)) |op| {
        for ([_][]const u8{ "bf16", "f16" }) |h| {
            if (std.mem.eql(u8, name, @tagName(op) ++ "_" ++ h)) {
                return @embedFile("half/" ++ h ++ ".wgsl") ++ @embedFile(@tagName(op) ++ "_half.wgsl");
            }
        }
    }
    for (std.enums.values(QuantOp)) |op| {
        const prefix = @tagName(op) ++ "_q";
        if (std.mem.startsWith(u8, name, prefix)) {
            const t = name[@tagName(op).len + 1 ..];
            // The decode multiply also takes the type's block-major decoder.
            const dot = if (op == .matvec) @embedFile("quant/" ++ t ++ "_dot.wgsl") else "";
            const body = if (op == .matvec) "matvec_quant.wgsl.hbs" else @tagName(op) ++ "_quant.wgsl";
            return @embedFile("quant/common.wgsl") ++ @embedFile("quant/" ++ t ++ ".wgsl") ++ dot ++ @embedFile(body);
        }
    }
    const group = std.fmt.comptimePrint;
    // The cache's readers and writer take the FP8 format (fp8.wgsl).
    if (std.mem.eql(u8, name, "attn_split")) return group("const MAX_GROUP: u32 = {d}u;\n", .{attn_max_group}) ++ @embedFile("fp8.wgsl") ++ @embedFile("attn_dense.wgsl") ++ @embedFile("attn_split.wgsl");
    if (std.mem.eql(u8, name, "attn_split_wide")) return group("const MAX_GROUP: u32 = {d}u;\n", .{attn_max_group_wide}) ++ @embedFile("fp8.wgsl") ++ @embedFile("attn_dense.wgsl") ++ @embedFile("attn_split.wgsl");
    if (std.mem.eql(u8, name, "attn_split_sparse")) return group("const MAX_GROUP: u32 = {d}u;\n", .{attn_max_group_wide}) ++ @embedFile("fp8.wgsl") ++ @embedFile("attn_sparse.wgsl") ++ @embedFile("attn_split.wgsl");
    // The kernels that store f16 round it themselves.
    if (std.mem.eql(u8, name, "rope_append")) return @embedFile("f16_round.wgsl") ++ @embedFile("fp8.wgsl") ++ @embedFile("rope_append.wgsl");
    if (std.mem.eql(u8, name, "qsa_keys")) return @embedFile("f16_round.wgsl") ++ @embedFile("qsa_keys.wgsl");
    return @embedFile(name ++ ".wgsl");
}

/// Rows per gyro_matvec workgroup.
pub const gyro_rows = 32;

/// Matrix multiply tile sizes (matmul_half.wgsl: TN outputs × TT tokens).
pub const matmul_tile_n = 64;
pub const matmul_tile_t = 16;
pub const matmul_tile_k = 32;
/// Quantized prefill multiply tiles (matmul_quant.wgsl: TN outputs × TM
/// tokens).
pub const matmul_quant_tile_n = 128;
pub const matmul_quant_tile_t = 32;

/// Split of a quantized prefill multiply over k: a matrix of n rows has
/// only ⌈n / tile⌉ column tiles per token tile, too few workgroups to keep
/// the GPU busy, so each takes `count` slices of `k_slice` positions (a
/// multiple of the 32-weight unit), aiming at matmul_target_groups, and
/// matmul_reduce adds them up. Slices are at least matmul_min_slice deep.
pub const matmul_target_groups = 256;
pub const matmul_min_slice = 512;
pub const Slices = struct { count: u32, k_slice: u32 };

pub fn matmulSlices(n: u32, k: u32) Slices {
    const tiles = std.math.divCeil(u32, n, matmul_quant_tile_n) catch unreachable;
    const want = std.math.divCeil(u32, matmul_target_groups, tiles) catch unreachable;
    const count = std.math.clamp(want, 1, @max(1, k / matmul_min_slice));
    const k_slice = std.mem.alignForward(u32, std.math.divCeil(u32, k, count) catch unreachable, 32);
    return .{ .count = std.math.divCeil(u32, k, k_slice) catch unreachable, .k_slice = k_slice };
}

/// Values per activation block the quantized prefill multiply rounds x to
/// for weights of type `t`, as llama.cpp's CPU backend does (vec_dot_type):
/// Q8_0 blocks of 32 for Q8_0, Q8_K blocks of 256 for the K-quants
/// (quantize_q8.wgsl).
pub fn actBlock(t: gguf.TensorType) u32 {
    return if (t == .q8_0) 32 else 256;
}

/// Tile sizes (outputs, tokens) of the prefill multiply for weights of type `t`.
pub fn matmulTiles(t: gguf.TensorType) [2]u32 {
    return if (isHalf(t)) .{ matmul_tile_n, matmul_tile_t } else .{ matmul_quant_tile_n, matmul_quant_tile_t };
}

/// Steps of at most this many tokens (decoding) use the decode kernels:
/// `matvec_bf16` and split attention.
pub const decode_max_rows = 4;
/// Rows per `matvec_bf16` workgroup.
pub const matvec_rows = 8;
/// A quantized decode multiply's tunable parameters (matvec_quant.wgsl.hbs):
/// `wg` threads, `lanes` of them per row, `rows` rows per lane group. Any of
/// them gives each row the same result for steps of one and two tokens; the
/// lane count fixes the order of each row's sums.
pub const MatvecVariant = struct {
    wg: u32 = 256,
    lanes: u32 = 32,
    rows: u32 = 1,
    /// A fused prologue, not a tuned parameter: nonzero, each input is
    /// staged as silu(x / act_div) (hc_act folded into the multiply after
    /// it).
    act_div: u32 = 0,

    /// The variant's id in the command stream (create_kernel): 0 for the
    /// default, else its fields packed (wg / 32, lanes, rows, act_div).
    pub fn id(v: MatvecVariant) u32 {
        if (std.meta.eql(v, MatvecVariant{})) return 0;
        return (v.wg / 32) | (v.lanes << 8) | (v.rows << 16) | (@as(u32, @min(v.act_div, 255)) << 24);
    }

    /// Rows one workgroup computes.
    pub fn groupRows(v: MatvecVariant) u32 {
        return v.wg / v.lanes * v.rows;
    }

    /// Whether the device runs it: its threads per workgroup, and its
    /// workgroup memory (x staged as 2,048 floats; two partial sums per
    /// thread and row).
    pub fn fits(v: MatvecVariant, d: DeviceLimits) bool {
        return v.wg <= d.invocations and 2048 * 4 + 2 * v.wg * v.rows * 4 <= d.workgroup_storage;
    }

    /// Whether `n` rows fit its grid: one workgroup per groupRows rows,
    /// at most 65,535 along x.
    pub fn fitsRows(v: MatvecVariant, n: u32) bool {
        return (std.math.divCeil(u32, n, v.groupRows()) catch unreachable) <= 65535;
    }

    /// Whether it may run weights of type `t`: its lanes must divide the
    /// tasks of both tiles (TILE_PAIR and TILE_ONE floats), so a token's sums
    /// are the same whether it runs alone or in a pair. Every type has 32
    /// tasks in a pair tile or more; Q8_0 has 64 (two per 32-weight block),
    /// so it also takes 64 lanes a row, which gives long rows more
    /// workgroups.
    pub fn suits(v: MatvecVariant, t: gguf.TensorType) bool {
        return v.lanes <= 32 or (t == .q8_0 and v.lanes == 64);
    }

    /// The sweep: lanes 8, 16, 32 (or for Q8_0 64) per row, 1 or 2 rows
    /// per lane group, 128 or 256 threads.
    pub const candidates = blk: {
        var list: [16]MatvecVariant = undefined;
        var i: usize = 0;
        for ([_]u32{ 256, 128 }) |wg| for ([_]u32{ 32, 16, 8, 64 }) |lanes| for ([_]u32{ 1, 2 }) |rows| {
            list[i] = .{ .wg = wg, .lanes = lanes, .rows = rows };
            i += 1;
        };
        break :blk list;
    };

    /// The WGSL of decode multiply `k` with these parameters, in `arena`.
    /// Renders template `source` (a templated kernel's, on any backend)
    /// for this variant.
    pub fn render(v: MatvecVariant, arena: Allocator, source: []const u8) (tmpl.Error)![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        try tmpl.render(arena, source, &.{
            .{ .name = "wg", .value = .{ .int = v.wg } },
            .{ .name = "lanes", .value = .{ .int = v.lanes } },
            .{ .name = "rows", .value = .{ .int = v.rows } },
            .{ .name = "act_div", .value = .{ .int = v.act_div } },
        }, &.{}, &out);
        return out.items;
    }
};

/// Device limits kernel variants are checked against (WebGPU's defaults).
pub const DeviceLimits = struct {
    invocations: u32 = 256,
    workgroup_storage: u32 = 16384,
};
/// Widest input `matvec_bf16` stages in workgroup memory, which its RMSNorm
/// prologue needs.
pub const matvec_max_staged = 3584;
/// Widest head attention and `delta_rule` handle.
pub const max_head_dim = 256;
/// State columns per `delta_rule` workgroup.
pub const delta_cols = 32;
/// Cache positions per `attn_split` workgroup, and the largest GQA group.
pub const attn_block = 64;
/// Tokens per `attn_split` launch of a prefill step: a step's attention runs
/// in chunks of this many rows, so the partials hold one chunk.
pub const attn_rows = 32;

comptime {
    // Decode steps use the first chunk.
    std.debug.assert(decode_max_rows <= attn_rows);
}
/// Most query heads per KV head attn_split holds, and attn_split_wide (its
/// workgroup memory: 16,384 bytes, WebGPU's guaranteed limit).
pub const attn_max_group = 8;
pub const attn_max_group_wide = 12;

/// The attention kernel for GQA groups of `group` query heads.
pub fn attnSplit(group: u32) Kernel {
    std.debug.assert(group <= attn_max_group_wide);
    return if (group <= attn_max_group) .attn_split else .attn_split_wide;
}
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
    /// `keep`: 0, or 1 + the row after which delta_rule saves the DeltaNet
    /// state and conv history to its kept binding (for a rollback to it).
    pub const Step = extern struct { m: u32, pos0: u32, u: f32, keep: u32 = 0 };
    /// `w_rows` > 1: row r takes weight row r mod w_rows.
    pub const RmsNorm = extern struct { rows: u32, dim: u32, eps: f32, w_rows: u32 = 0 };
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
    /// The quantized prefill multiply: as MatmulBf16 over positions
    /// [z·k_slice, (z + 1)·k_slice) for workgroup z, writing at
    /// z·slice_stride (matmulSlices).
    pub const MatmulQuant = extern struct {
        rows: u32,
        n: u32,
        k: u32,
        col0: u32,
        out_stride: u32,
        accumulate: u32 = 0,
        k_slice: u32,
        slice_stride: u32 = 0,
    };
    pub const MatmulReduce = extern struct {
        rows: u32,
        n: u32,
        slices: u32,
        col0: u32,
        out_stride: u32,
        accumulate: u32 = 0,
        slice_stride: u32,
        _pad: u32 = 0,
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
        /// Vectors of 4 between tokens' inputs; 0: k / 4. (The quantized
        /// kernel only: an input read out of a wider row.)
        x_stride: u32 = 0,

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
    pub const QuantizeQ8 = extern struct {
        k: u32,
        block: u32,
        /// What is rounded: 0 x, 1 RMSNorm(x) · w, 2 silu(gate) · up.
        mode: u32 = 0,
        eps: f32 = 0,
    };
    pub const RepackQ80 = extern struct { rows: u32, k: u32, _pad: [2]u32 = @splat(0) };
    pub const RepackQ3K = extern struct { blocks: u32, _pad: [3]u32 = @splat(0) };
    pub const Repad = extern struct { blocks: u32, src_bytes: u32, dst_bytes: u32, _pad: u32 = 0 };
    pub const SiluMul = extern struct { count: u32, width: u32, _pad: [2]u32 = @splat(0) };
    /// `stride`: bytes from one quantized row to the next; 0, the type's
    /// row size (the half-precision kernels ignore it).
    pub const EmbedBf16 = extern struct { dim: u32, row0: u32, chunk_rows: u32, stride: u32 = 0 };
    pub const RopeAppend = extern struct {
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        n_rot: u32,
        row: u32,
        k_off: u32,
        v_off: u32,
        /// The cache format (plan.common.KvFormat).
        kv: u32,
        eps: f32,
        /// Query heads interleaved with their output gates (2·head_dim
        /// apart); else head_dim apart.
        gated: u32 = 1,
        /// Per-head RMSNorm of Q and K by q_norm and k_norm; else neither
        /// is normalized (both bindings are still bound).
        qk_norm: u32 = 1,
        /// RoPE pairs (2i, 2i + 1) (Llama's "normal" RoPE); else
        /// (i, i + n_rot/2) (NeoX).
        normal: u32 = 0,
        /// FP8 cache: the word where the scales start (kvScales).
        scales: u32 = 0,
        _pad: [3]u32 = @splat(0),
    };
    /// The sparse attention indexer's (qsa_*.wgsl, attn_sparse.wgsl):
    /// `heads` query heads of `dim`, blocks of `ratio` cells, `width` cells
    /// attended past which selection starts, `stride` blocks per row of the
    /// scores and the selection, `row` floats per row of the projection.
    pub const Qsa = extern struct {
        heads: u32,
        dim: u32,
        ratio: u32,
        width: u32,
        stride: u32,
        n_rot: u32,
        eps: f32,
        row: u32,
    };
    /// `attn_split` and `attn_combine`, for tokens row0 ...
    pub const Attention = extern struct {
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        /// The cache format (plan.common.KvFormat).
        kv: u32,
        scale: f32,
        max_blocks: u32,
        q_row: u32,
        row0: u32 = 0,
        /// Query heads interleaved with their output gates, which
        /// attn_combine applies; else head_dim apart, ungated.
        gated: u32 = 1,
        /// FP8 cache: the word where the scales start (kvScales).
        scales: u32 = 0,
        _pad: [2]u32 = @splat(0),
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
        /// Where the kept conv history starts in the kept binding, in floats
        /// (the kept state starts at 0).
        kept_hist: u32 = 0,
        _pad: u32 = 0,
    };
    pub const Draft = extern struct { vocab: u32, row: u32, slot: u32, _pad: u32 = 0 };
    pub const MoeRoute = extern struct { n_expert: u32, n_used: u32, cols: u32, x_stride: u32, scale: f32 = 0, _pad: [3]u32 = @splat(0) };
    pub const Rotate128 = extern struct { width: u32, x_stride: u32, y_stride: u32, silu_pair: u32 = 0 };
    pub const MoeCombine = extern struct { n_used: u32, dim: u32, _pad: [2]u32 = @splat(0) };
    /// `count` = rows · rank; the down rows are `stride` wide.
    pub const HcAct = extern struct { count: u32, hc: u32, rank: u32, stride: u32 };
    pub const Hc = extern struct { dim: u32, hc: u32, _pad: [2]u32 = @splat(0) };
    /// The inject weights at row t's `offset + c`, rows `stride` apart.
    pub const HcCombine = extern struct { dim: u32, hc: u32, stride: u32, offset: u32 };
    pub const PleMix = extern struct { dim: u32, hc: u32, eps: f32, _pad: u32 = 0 };
    pub const PleConv = extern struct { channels: u32, taps: u32, dilation: u32, _pad: u32 = 0 };
    pub const GyroMatvec = extern struct {
        rows: u32,
        cols: u32,
        n_used: u32,
        x_per_slot: u32,
        x_stride: u32,
        out_stride: u32,
        out_z: u32,
        _pad: u32 = 0,
    };
    /// Stable expert lists, with active rows supplied by the final Step binding.
    pub const MoeGroup = extern struct { n_used: u32, n_expert: u32, max_slots: u32, _pad: u32 = 0 };
    pub const GyroMatmul = extern struct {
        rows: u32,
        cols: u32,
        n_used: u32,
        x_per_slot: u32,
        x_stride: u32,
        out_stride: u32,
        out_z: u32,
        max_slots: u32,
    };
    /// Which phase of `sample` a launch runs.
    pub const SamplePhase = extern struct { phase: u32, round: u32 = 0, _pad: [2]u32 = @splat(0) };
    pub const GatedNorm = extern struct {
        v_heads: u32,
        dim: u32,
        eps: f32,
        z_row: u32,
        z_off: u32,
        /// 1: gate by sigmoid(z) (Qwen4Exp), not silu(z).
        sigmoid: u32 = 0,
        _pad: [2]u32 = @splat(0),
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
    /// Not part of a draw: undoes the last draw's output count.
    unbump,

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
    return ropeTableScaled(gpa, positions, n_rot, base, null);
}

/// As `ropeTable`, pair i's angle divided by factors[i] (Llama 3's
/// `rope_freqs`).
pub fn ropeTableScaled(gpa: Allocator, positions: usize, n_rot: usize, base: f32, factors: ?[]const f32) Allocator.Error![]f32 {
    const half = n_rot / 2;
    const table = try gpa.alloc(f32, positions * half * 2);
    for (0..positions) |pos| for (0..half) |i| {
        const cs = ops.ropeCosSinScaled(@intCast(pos), i, n_rot, base, if (factors) |f| f[i] else 1);
        table[(pos * half + i) * 2 ..][0..2].* = cs;
    };
    return table;
}

/// One pipeline per kernel (templates with their default parameters),
/// plus the other decode-multiply variants the plan asks for.
pub const Set = struct {
    pipelines: std.EnumArray(Kernel, cmd.Pipeline),
    /// The backend the pipelines are built for.
    backend: Backend,
    /// Kernels the backend has no implementation of yet: their pipelines
    /// exist (with an empty artifact) but dispatching one fails the batch
    /// with error.UnsupportedKernel. Empty on WebGPU.
    missing: std.EnumSet(Kernel) = .empty,
    variants: [max_variants]Variant = undefined,
    n_variants: usize = 0,
    /// Original scalar pipelines are never replaced or recompiled.
    scalar_prefill: [4]cmd.Pipeline = undefined,
    /// Lazily recorded real LinAlg pipelines, single then compensated.
    prefill_variants: [2][4]?cmd.Pipeline = .{ @splat(null), @splat(null) },

    const Variant = struct { kernel: Kernel, v: MatvecVariant, pipeline: cmd.Pipeline };
    /// Every candidate of every quantized type, with and without a fused
    /// prologue.
    const max_variants = 2 * 5 * MatvecVariant.candidates.len;

    pub const Error = Allocator.Error || tmpl.Error;

    /// Records pipeline creation for every kernel on `backend`. Artifacts
    /// rendered from templates live in `arena`, which must outlive the
    /// batch's execution.
    pub fn create(gpa: Allocator, arena: Allocator, rec: *cmd.Recorder, backend: Backend) Error!Set {
        var set: Set = .{ .pipelines = undefined, .backend = backend };
        for (std.enums.values(Kernel)) |k| {
            const code = artifact(backend, arena, k, .{}) catch |err| switch (err) {
                error.UnsupportedKernel => blk: {
                    set.missing.insert(k);
                    break :blk "";
                },
                else => |e| return e,
            };
            set.pipelines.set(k, try rec.createKernel(gpa, @backingInt(k), 0, k.contract().writes, code));
        }
        for (agility_kernels.prefill_kernels, 0..) |k, i| set.scalar_prefill[i] = set.pipelines.get(k);
        return set;
    }

    /// Fails, logging each missing kernel, unless the backend implements
    /// every kernel: models load only on a complete backend.
    pub fn requireAll(s: *const Set) error{UnsupportedKernel}!void {
        if (s.missing.count() == 0) return;
        var it = s.missing.iterator();
        while (it.next()) |k| std.log.err("{t}: kernel {t} has no implementation", .{ s.backend, k });
        return error.UnsupportedKernel;
    }

    /// Switches the four prefill identities without changing their contracts
    /// or grids. Sources live for the recorder batch, as with create. Native
    /// support must be checked with Device.supportsPrefill before selection;
    /// the Agility host independently rejects unsupported LinAlg pipelines.
    /// Existing plans retain their captured pipelines: select before planning.
    pub fn selectPrefill(s: *Set, gpa: Allocator, _: Allocator, rec: *cmd.Recorder, rung: precision.Prefill) (Error || error{UnsupportedPrefill})!void {
        if (rung == .scalar_f32) {
            for (agility_kernels.prefill_kernels, 0..) |k, i| s.pipelines.set(k, s.scalar_prefill[i]);
            return;
        }
        if (s.backend != .agility) return error.UnsupportedPrefill;
        const slot: usize = switch (rung) {
            .fp16_single => 0,
            .fp16_compensated => 1,
            .scalar_f32 => unreachable,
        };
        for (agility_kernels.prefill_kernels, 0..) |k, i| {
            if (s.prefill_variants[slot][i] == null) {
                const code = agility_kernels.prefillSource(k, rung) orelse return error.UnsupportedPrefill;
                s.prefill_variants[slot][i] = try rec.createKernel(gpa, @backingInt(k), agility_kernels.prefillVariant(rung), k.contract().writes, code);
            }
        }
        // Only publish the selection after all four pipelines are recorded.
        for (agility_kernels.prefill_kernels, 0..) |k, i| s.pipelines.set(k, s.prefill_variants[slot][i].?);
    }

    /// The pipeline of decode multiply `k` with variant `v`, recording its
    /// creation the first time (its source in `arena`, as for `create`).
    pub fn matvec(s: *Set, gpa: Allocator, arena: Allocator, rec: *cmd.Recorder, k: Kernel, v: MatvecVariant) Error!cmd.Pipeline {
        if (std.meta.eql(v, MatvecVariant{})) return s.pipelines.get(k);
        for (s.variants[0..s.n_variants]) |x| if (x.kernel == k and std.meta.eql(x.v, v)) return x.pipeline;
        std.debug.assert(s.n_variants < max_variants);
        const code = artifact(s.backend, arena, k, v) catch |err| switch (err) {
            error.UnsupportedKernel => "",
            else => |e| return e,
        };
        const pipeline = try rec.createKernel(gpa, @backingInt(k), v.id(), k.contract().writes, code);
        s.variants[s.n_variants] = .{ .kernel = k, .v = v, .pipeline = pipeline };
        s.n_variants += 1;
        return pipeline;
    }
};

test "prefill selection preserves scalar originals and caches each native rung" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var rec: cmd.Recorder = .{};
    defer rec.deinit(gpa);
    var set = try Set.create(gpa, arena.allocator(), &rec, .agility);
    const scalar = set.pipelines;
    var originals: cmd.Iterator = .{ .words = rec.batch() };
    while (try originals.next()) |command| {
        try std.testing.expectEqual(cmd.Op.create_kernel, command.op);
        try std.testing.expectEqual(@as(u32, 0), command.args[2]);
    }
    const base = rec.created(.pipeline);
    for ([_]precision.Prefill{ .fp16_single, .fp16_compensated }) |rung| {
        rec.clear();
        try set.selectPrefill(gpa, arena.allocator(), &rec, rung);
        var it: cmd.Iterator = .{ .words = rec.batch() };
        for (agility_kernels.prefill_kernels) |k| {
            const command = (try it.next()).?;
            try std.testing.expectEqual(@as(u32, @backingInt(k)), command.args[1]);
            try std.testing.expectEqual(agility_kernels.prefillVariant(rung), command.args[2]);
            try std.testing.expectEqual(k.contract().writes, command.args[3]);
            const code = command.ptrAt([*]const u8, 4)[0..command.args[6]];
            try std.testing.expectEqualSlices(u8, agility_kernels.prefillSource(k, rung).?, code);
            try std.testing.expect(set.pipelines.get(k) != scalar.get(k));
        }
        try std.testing.expect((try it.next()) == null);
        const selected = set.pipelines;
        try set.selectPrefill(gpa, arena.allocator(), &rec, .scalar_f32);
        try std.testing.expectEqualDeep(scalar, set.pipelines);
        rec.clear();
        try set.selectPrefill(gpa, arena.allocator(), &rec, rung);
        try std.testing.expectEqualDeep(selected, set.pipelines);
        try std.testing.expectEqual(@as(usize, 0), rec.batch().len);
        for (std.enums.values(Kernel)) |k| {
            if (agility_kernels.linAlgSource(k) == null) try std.testing.expectEqual(scalar.get(k), set.pipelines.get(k));
        }
    }
    try std.testing.expectEqual(base + 8, rec.created(.pipeline));
}

test "WebGPU rejects approximate prefill without recording or relabeling" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var rec: cmd.Recorder = .{};
    defer rec.deinit(gpa);
    var set = try Set.create(gpa, arena.allocator(), &rec, .webgpu);
    const scalar = set.pipelines;
    rec.clear();
    for ([_]precision.Prefill{ .fp16_single, .fp16_compensated }) |rung| {
        try std.testing.expectError(error.UnsupportedPrefill, set.selectPrefill(gpa, arena.allocator(), &rec, rung));
        try std.testing.expectEqualDeep(scalar, set.pipelines);
        try std.testing.expectEqual(@as(usize, 0), rec.batch().len);
    }
    try set.selectPrefill(gpa, arena.allocator(), &rec, .scalar_f32);
    try std.testing.expectEqualDeep(scalar, set.pipelines);
}

/// Kernel `k`'s implementation on `backend`, in variant `v` (templated
/// kernels only; the default otherwise), rendered into `arena` where it is
/// a template.
fn artifact(backend: Backend, arena: Allocator, k: Kernel, v: MatvecVariant) (Set.Error || error{UnsupportedKernel})![]const u8 {
    return switch (backend) {
        .webgpu => if (k.isTemplate()) v.render(arena, k.source()) else k.source(),
        .agility => {
            const source = agility_kernels.source(k) orelse return error.UnsupportedKernel;
            return if (k.isTemplate()) v.render(arena, source) else source;
        },
    };
}

/// Kernel `k`'s source on `backend` (its template, for a templated
/// kernel), or empty where the backend lacks it: what the plan cache's key
/// hashes.
pub fn backendSource(backend: Backend, k: Kernel) []const u8 {
    return switch (backend) {
        .webgpu => k.source(),
        .agility => agility_kernels.cacheSource(k),
    };
}

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
        try l.launchPipeline(l.set.pipelines.get(kernel), p, args, groups);
    }

    /// `launch` with a given pipeline (a kernel variant, Set.matvec).
    pub fn launchPipeline(l: Launcher, pipeline: cmd.Pipeline, p: anytype, args: []const Arg, groups: [3]u32) Allocator.Error!void {
        const P = @TypeOf(p);
        comptime std.debug.assert(@sizeOf(P) % 16 == 0);
        const bytes = try l.arena.dupe(u8, std.mem.asBytes(&p));
        const ub = try l.rec.createBuffer(l.gpa, @sizeOf(P), .{ .uniform = true, .copy_dst = true });
        try l.rec.writeBuffer(l.gpa, ub, 0, bytes);
        const bg = try bindGroup(l.gpa, l.rec, pipeline, .{ .buffer = ub, .size = @sizeOf(P) }, args);
        try l.rec.dispatch(l.gpa, pipeline, groups, &.{bg});
    }

    /// A launch whose binding 0 is the caller's `uniform`.
    pub fn launchWith(l: Launcher, kernel: Kernel, uniform: Arg, args: []const Arg, groups: [3]u32) Allocator.Error!void {
        const pipeline = l.set.pipelines.get(kernel);
        const bg = try bindGroup(l.gpa, l.rec, pipeline, uniform, args);
        try l.rec.dispatch(l.gpa, pipeline, groups, &.{bg});
    }

    /// Lays out `rows` raw GGUF rows of `k` weights of type `t` from `src`
    /// in `dst` as the quantized kernels read them (rowBytes): Q3_K
    /// repacked, Q8_0 split, other types padded.
    pub fn layout(l: Launcher, t: gguf.TensorType, src: Arg, dst: Arg, rows: u64, k: u64) Allocator.Error!void {
        const bytes = t.layout().?.block_bytes;
        const stride = blockStride(t);
        const blocks = rows * k / t.layout().?.block_elems;
        if (t == .q8_0) {
            try l.launch(.repack_q8_0, params.RepackQ80{ .rows = @intCast(rows), .k = @intCast(k) }, &.{ src, dst }, elementwise(@intCast(rows * rowBytes(t, k) / 4)));
        } else if (t == .q3_k) {
            try l.launch(.repack_q3_k, params.RepackQ3K{ .blocks = @intCast(blocks) }, &.{ src, dst }, elementwise(@intCast(blocks * 8)));
        } else {
            try l.launch(.repad, params.Repad{ .blocks = @intCast(blocks), .src_bytes = bytes, .dst_bytes = stride }, &.{ src, dst }, elementwise(@intCast(blocks * stride / 4)));
        }
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
    /// (x, ⌈rows / tile⌉, z)
    tiles_y: struct { x: u32, tile: u32, z: u32 = 1 },
    /// One thread per element of rows × width (`Launcher.elementwise`).
    elems: u32,
    /// (x, ⌈keys / block⌉, rows past row0, at most max_rows): blocks of
    /// cache positions per token.
    key_blocks: struct { x: u32, block: u32, row0: u32 = 0, max_rows: u32 = std.math.maxInt(u32) },
    /// (x, rows past row0, at most max_rows, 1)
    rows_y_from: struct { x: u32, row0: u32, max_rows: u32 },

    pub fn resolve(g: Grid, e: Extent) [3]u32 {
        const rows = e.rows;
        return switch (g) {
            .fixed => |f| f,
            .rows_x => .{ rows, 1, 1 },
            .rows_y => |x| .{ x, rows, 1 },
            .tiles_y => |t| .{ t.x, std.math.divCeil(u32, rows, t.tile) catch unreachable, t.z },
            .elems => |w| Launcher.elementwise(@as(usize, rows) * w),
            .key_blocks => |k| .{ k.x, std.math.divCeil(u32, e.keys, k.block) catch unreachable, @min(k.max_rows, rows -| k.row0) },
            .rows_y_from => |r| .{ r.x, @min(r.max_rows, rows -| r.row0), 1 },
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

    /// Records the launch for extent `e`; nothing if its grid is empty (a
    /// chunk past the step's rows).
    pub fn record(l: Launch, gpa: Allocator, rec: *cmd.Recorder, e: Extent) Allocator.Error!void {
        const g = l.grid.resolve(e);
        if (g[0] == 0 or g[1] == 0 or g[2] == 0) return;
        try rec.dispatch(gpa, l.pipeline, g, &.{l.bind_group});
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
    set: *Set,
    pages: std.ArrayList(Page) = .empty,

    const page_size = 64 * 1024;
    const slot = 256;
    const Page = struct { buffer: cmd.Buffer, data: []u8, used: usize };

    pub fn deinit(b: *Builder) void {
        b.pages.deinit(b.gpa);
    }

    /// A launch with its own parameters at binding 0.
    pub fn add(b: *Builder, kernel: Kernel, p: anytype, args: []const Arg, grid: Grid) Allocator.Error!Launch {
        return b.addWith(kernel, try b.param(p), args, grid);
    }

    /// A slot holding parameters `p`.
    fn param(b: *Builder, p: anytype) Allocator.Error!Arg {
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
        return uniform;
    }

    /// A launch whose binding 0 is `uniform`, a buffer the caller rewrites.
    pub fn addWith(b: *Builder, kernel: Kernel, uniform: Arg, args: []const Arg, grid: Grid) Allocator.Error!Launch {
        return b.addPipeline(kernel, b.set.pipelines.get(kernel), uniform, args, grid);
    }

    /// `add` with variant `v` of decode multiply `kernel`.
    pub fn addMatvec(b: *Builder, kernel: Kernel, v: MatvecVariant, p: anytype, args: []const Arg, grid: Grid) Set.Error!Launch {
        const pipeline = try b.set.matvec(b.gpa, b.arena, b.rec, kernel, v);
        return b.addPipeline(kernel, pipeline, try b.param(p), args, grid);
    }

    fn addPipeline(b: *Builder, kernel: Kernel, pipeline: cmd.Pipeline, uniform: Arg, args: []const Arg, grid: Grid) Allocator.Error!Launch {
        return .{ .kernel = kernel, .pipeline = pipeline, .bind_group = try bindGroup(b.gpa, b.rec, pipeline, uniform, args), .grid = grid };
    }

    /// Records the upload of every parameter page.
    pub fn finish(b: *Builder) Allocator.Error!void {
        for (b.pages.items) |page| try b.rec.writeBuffer(b.gpa, page.buffer, 0, page.data[0..page.used]);
    }
};
