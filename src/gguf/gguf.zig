//! GGUF container reader.
//!
//! `parse` reads the header, metadata and tensor index from a borrowed byte
//! slice without copying: strings and arrays in the result point into that
//! slice, so it must outlive the `File`. Natively the slice is an mmap of the
//! whole file; in the browser it is the header prefix staged from a range read.
//!
//! Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const magic = "GGUF";
pub const default_alignment: u32 = 32;

/// GGML tensor storage types, numbered as in `ggml.h`.
pub const TensorType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    tq1_0 = 34,
    tq2_0 = 35,
    mxfp4 = 39,
    _,

    pub const Layout = struct {
        /// Elements per block.
        block_elems: u32,
        /// Bytes per block.
        block_bytes: u32,
    };

    /// Block geometry, or null for types this reader does not know.
    pub fn layout(t: TensorType) ?Layout {
        return switch (t) {
            .f32, .i32 => .{ .block_elems = 1, .block_bytes = 4 },
            .f16, .bf16, .i16 => .{ .block_elems = 1, .block_bytes = 2 },
            .i8 => .{ .block_elems = 1, .block_bytes = 1 },
            .f64, .i64 => .{ .block_elems = 1, .block_bytes = 8 },
            .q4_0 => .{ .block_elems = 32, .block_bytes = 18 },
            .q4_1 => .{ .block_elems = 32, .block_bytes = 20 },
            .q5_0 => .{ .block_elems = 32, .block_bytes = 22 },
            .q5_1 => .{ .block_elems = 32, .block_bytes = 24 },
            .q8_0 => .{ .block_elems = 32, .block_bytes = 34 },
            .q8_1 => .{ .block_elems = 32, .block_bytes = 36 },
            .iq4_nl => .{ .block_elems = 32, .block_bytes = 18 },
            .mxfp4 => .{ .block_elems = 32, .block_bytes = 17 },
            .q2_k => .{ .block_elems = 256, .block_bytes = 84 },
            .q3_k => .{ .block_elems = 256, .block_bytes = 110 },
            .q4_k => .{ .block_elems = 256, .block_bytes = 144 },
            .q5_k => .{ .block_elems = 256, .block_bytes = 176 },
            .q6_k => .{ .block_elems = 256, .block_bytes = 210 },
            .q8_k => .{ .block_elems = 256, .block_bytes = 292 },
            .iq2_xxs => .{ .block_elems = 256, .block_bytes = 66 },
            .iq2_xs => .{ .block_elems = 256, .block_bytes = 74 },
            .iq2_s => .{ .block_elems = 256, .block_bytes = 82 },
            .iq3_xxs => .{ .block_elems = 256, .block_bytes = 98 },
            .iq3_s => .{ .block_elems = 256, .block_bytes = 110 },
            .iq1_s => .{ .block_elems = 256, .block_bytes = 50 },
            .iq1_m => .{ .block_elems = 256, .block_bytes = 56 },
            .iq4_xs => .{ .block_elems = 256, .block_bytes = 136 },
            .tq1_0 => .{ .block_elems = 256, .block_bytes = 54 },
            .tq2_0 => .{ .block_elems = 256, .block_bytes = 66 },
            _ => null,
        };
    }
};

/// Metadata value type tags, numbered as in the GGUF spec.
pub const ValueType = enum(u32) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    u32 = 4,
    i32 = 5,
    f32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    u64 = 10,
    i64 = 11,
    f64 = 12,
    _,

    /// Encoded size for fixed-size scalars; null for string and array.
    fn scalarSize(t: ValueType) ?usize {
        return switch (t) {
            .u8, .i8, .bool => 1,
            .u16, .i16 => 2,
            .u32, .i32, .f32 => 4,
            .u64, .i64, .f64 => 8,
            .string, .array, _ => null,
        };
    }
};

/// A metadata array. Elements stay encoded in the source bytes; walk them
/// with `iterator` or `get`. Strings require a walk, scalars index directly.
pub const Array = struct {
    elem_type: ValueType,
    len: u64,
    /// Encoded elements, exactly spanning `len` items.
    bytes: []const u8,

    /// Element `i` of a scalar array as the unsigned integer of the element's
    /// width; `@bitCast` it for signed and float element types.
    pub fn get(a: Array, comptime T: type, i: usize) T {
        const size = @sizeOf(T);
        std.debug.assert(a.elem_type.scalarSize() == size);
        std.debug.assert(i < a.len);
        return std.mem.readInt(T, a.bytes[i * size ..][0..size], .little);
    }

    pub fn iterator(a: Array) Iterator {
        return .{ .array = a, .pos = 0, .index = 0 };
    }

    pub const Iterator = struct {
        array: Array,
        pos: usize,
        index: u64,

        /// Next string element; only valid for string arrays.
        pub fn nextString(it: *Iterator) ?[]const u8 {
            std.debug.assert(it.array.elem_type == .string);
            if (it.index == it.array.len) return null;
            const b = it.array.bytes;
            const n: usize = @intCast(std.mem.readInt(u64, b[it.pos..][0..8], .little));
            const s = b[it.pos + 8 ..][0..n];
            it.pos += 8 + n;
            it.index += 1;
            return s;
        }
    };
};

pub const Value = union(ValueType) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    f32: f32,
    bool: bool,
    string: []const u8,
    array: Array,
    u64: u64,
    i64: i64,
    f64: f64,
};

pub const KeyValue = struct {
    key: []const u8,
    value: Value,
};

pub const max_dims = 4;

pub const TensorInfo = struct {
    name: []const u8,
    /// GGML order: `dims[0]` is the contiguous (row) dimension.
    dims: [max_dims]u64,
    n_dims: u32,
    type: TensorType,
    /// Offset from the start of the tensor data section.
    offset: u64,

    pub fn shape(t: *const TensorInfo) []const u64 {
        return t.dims[0..t.n_dims];
    }

    pub fn elemCount(t: TensorInfo) u64 {
        var n: u64 = 1;
        for (t.dims[0..t.n_dims]) |d| n *= d;
        return n;
    }

    /// Encoded byte size. Callers must have validated the type via `parse`.
    pub fn byteSize(t: TensorInfo) u64 {
        const l = t.type.layout().?;
        return t.elemCount() / l.block_elems * l.block_bytes;
    }
};

pub const File = struct {
    version: u32,
    alignment: u32,
    metadata: []KeyValue,
    tensors: []TensorInfo,
    /// Absolute file offset of the tensor data section.
    data_offset: u64,

    pub fn deinit(f: *File, gpa: Allocator) void {
        gpa.free(f.metadata);
        gpa.free(f.tensors);
        f.* = undefined;
    }

    pub fn get(f: File, key: []const u8) ?Value {
        for (f.metadata) |kv| {
            if (std.mem.eql(u8, kv.key, key)) return kv.value;
        }
        return null;
    }

    pub fn tensor(f: File, name: []const u8) ?*const TensorInfo {
        for (f.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    /// Encoded bytes of `t` inside `file_bytes`, the whole file.
    pub fn tensorBytes(f: File, file_bytes: []const u8, t: *const TensorInfo) []const u8 {
        const start: usize = @intCast(f.data_offset + t.offset);
        return file_bytes[start..][0..@intCast(t.byteSize())];
    }
};

pub const ParseError = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    BadValueType,
    BadTensorType,
    BadDims,
    BadAlignment,
    BadTensorOffset,
    TensorOutOfBounds,
    Overflow,
} || Allocator.Error;

/// Parse a GGUF header, metadata and tensor index.
///
/// `bytes` must contain at least the full header and index. When it is the
/// whole file (`whole_file`), every tensor is also bounds-checked against it.
pub fn parse(gpa: Allocator, bytes: []const u8, whole_file: bool) ParseError!File {
    var r: Reader = .{ .bytes = bytes };

    if (!std.mem.eql(u8, try r.take(4), magic)) return error.BadMagic;
    const version = try r.int(u32);
    if (version != 3) return error.UnsupportedVersion;
    const n_tensors = try r.int(u64);
    const n_kv = try r.int(u64);

    // Each KV needs at least 8 (key len) + 4 (type) bytes; each tensor at
    // least 8 + 4 + 4 + 8. Reject counts the input cannot possibly hold
    // before allocating for them.
    if (n_kv > bytes.len / 12 or n_tensors > bytes.len / 24) return error.Truncated;

    const metadata = try gpa.alloc(KeyValue, @intCast(n_kv));
    errdefer gpa.free(metadata);
    var alignment = default_alignment;
    for (metadata) |*kv| {
        kv.key = try r.string();
        kv.value = try r.value();
        if (std.mem.eql(u8, kv.key, "general.alignment")) {
            alignment = switch (kv.value) {
                .u32 => |a| a,
                else => return error.BadAlignment,
            };
            if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.BadAlignment;
        }
    }

    const tensors = try gpa.alloc(TensorInfo, @intCast(n_tensors));
    errdefer gpa.free(tensors);
    for (tensors) |*t| {
        t.name = try r.string();
        t.n_dims = try r.int(u32);
        if (t.n_dims == 0 or t.n_dims > max_dims) return error.BadDims;
        t.dims = @splat(1);
        for (t.dims[0..t.n_dims]) |*d| d.* = try r.int(u64);
        t.type = std.enums.fromInt(TensorType, try r.int(u32)) orelse return error.BadTensorType;
        const l = t.type.layout() orelse return error.BadTensorType;
        t.offset = try r.int(u64);
        if (t.offset % alignment != 0) return error.BadTensorOffset;

        var elems: u64 = 1;
        for (t.dims[0..t.n_dims]) |d| elems = std.math.mul(u64, elems, d) catch return error.Overflow;
        if (t.dims[0] % l.block_elems != 0) return error.BadDims;
        _ = std.math.mul(u64, elems / l.block_elems, l.block_bytes) catch return error.Overflow;
    }

    const data_offset = std.mem.alignForward(u64, r.pos, alignment);
    if (whole_file) {
        for (tensors) |t| {
            const end = std.math.add(u64, data_offset + t.offset, t.byteSize()) catch return error.Overflow;
            if (end > bytes.len) return error.TensorOutOfBounds;
        }
    }

    return .{
        .version = version,
        .alignment = alignment,
        .metadata = metadata,
        .tensors = tensors,
        .data_offset = data_offset,
    };
}

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(r: *Reader, n: u64) error{Truncated}![]const u8 {
        if (n > r.bytes.len - r.pos) return error.Truncated;
        const s = r.bytes[r.pos..][0..@intCast(n)];
        r.pos += @intCast(n);
        return s;
    }

    fn int(r: *Reader, comptime T: type) error{Truncated}!T {
        return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    fn string(r: *Reader) error{Truncated}![]const u8 {
        return r.take(try r.int(u64));
    }

    fn valueType(r: *Reader) ParseError!ValueType {
        return std.enums.fromInt(ValueType, try r.int(u32)) orelse error.BadValueType;
    }

    fn value(r: *Reader) ParseError!Value {
        return switch (try r.valueType()) {
            .u8 => .{ .u8 = try r.int(u8) },
            .i8 => .{ .i8 = try r.int(i8) },
            .u16 => .{ .u16 = try r.int(u16) },
            .i16 => .{ .i16 = try r.int(i16) },
            .u32 => .{ .u32 = try r.int(u32) },
            .i32 => .{ .i32 = try r.int(i32) },
            .f32 => .{ .f32 = @bitCast(try r.int(u32)) },
            .bool => .{ .bool = switch (try r.int(u8)) {
                0 => false,
                1 => true,
                else => return error.BadValueType,
            } },
            .string => .{ .string = try r.string() },
            .array => .{ .array = try r.array() },
            .u64 => .{ .u64 = try r.int(u64) },
            .i64 => .{ .i64 = try r.int(i64) },
            .f64 => .{ .f64 = @bitCast(try r.int(u64)) },
            _ => error.BadValueType,
        };
    }

    fn array(r: *Reader) ParseError!Array {
        const elem_type = try r.valueType();
        const len = try r.int(u64);
        const start = r.pos;
        if (elem_type.scalarSize()) |size| {
            const n = std.math.mul(u64, len, size) catch return error.Truncated;
            _ = try r.take(n);
        } else switch (elem_type) {
            .string => for (0..len) |_| {
                _ = try r.string();
            },
            // Nested arrays are legal GGUF but no known producer emits them;
            // walking them would need recursion over untrusted depth.
            else => return error.BadValueType,
        }
        return .{ .elem_type = elem_type, .len = len, .bytes = r.bytes[start..r.pos] };
    }
};

// ---------------------------------------------------------------------------
// Tests: a minimal writer builds files covering each encoding path.

const TestWriter = struct {
    buf: std.ArrayList(u8) = .empty,

    fn int(w: *TestWriter, comptime T: type, v: T) void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        w.buf.appendSlice(std.testing.allocator, &b) catch unreachable;
    }

    fn str(w: *TestWriter, s: []const u8) void {
        w.int(u64, s.len);
        w.buf.appendSlice(std.testing.allocator, s) catch unreachable;
    }

    fn pad(w: *TestWriter, alignment: usize) void {
        while (w.buf.items.len % alignment != 0) w.buf.append(std.testing.allocator, 0) catch unreachable;
    }
};

fn buildTestFile(w: *TestWriter, alignment: ?u32) void {
    w.buf.appendSlice(std.testing.allocator, magic) catch unreachable;
    w.int(u32, 3);
    w.int(u64, 2); // tensors
    w.int(u64, if (alignment != null) 5 else 4); // kv

    w.str("general.architecture");
    w.int(u32, @backingInt(ValueType.string));
    w.str("qwen35");

    w.str("qwen35.block_count");
    w.int(u32, @backingInt(ValueType.u32));
    w.int(u32, 24);

    w.str("tokenizer.ggml.tokens");
    w.int(u32, @backingInt(ValueType.array));
    w.int(u32, @backingInt(ValueType.string));
    w.int(u64, 3);
    w.str("a");
    w.str("bc");
    w.str("");

    w.str("tokenizer.ggml.token_type");
    w.int(u32, @backingInt(ValueType.array));
    w.int(u32, @backingInt(ValueType.i32));
    w.int(u64, 3);
    w.int(i32, 1);
    w.int(i32, -1);
    w.int(i32, 3);

    if (alignment) |a| {
        w.str("general.alignment");
        w.int(u32, @backingInt(ValueType.u32));
        w.int(u32, a);
    }

    w.str("w");
    w.int(u32, 2);
    w.int(u64, 4);
    w.int(u64, 2);
    w.int(u32, @backingInt(TensorType.f32));
    w.int(u64, 0);

    w.str("b");
    w.int(u32, 1);
    w.int(u64, 32);
    w.int(u32, @backingInt(TensorType.q8_0));
    w.int(u64, alignment orelse default_alignment);

    w.pad(alignment orelse default_alignment);
    for (0..8) |i| w.int(u32, @bitCast(@as(f32, @floatFromInt(i))));
    w.pad(alignment orelse default_alignment);
    for (0..34) |_| w.buf.append(std.testing.allocator, 0) catch unreachable;
}

test "parse metadata, arrays and tensor index" {
    const gpa = std.testing.allocator;
    var w: TestWriter = .{};
    defer w.buf.deinit(gpa);
    buildTestFile(&w, null);

    var f = try parse(gpa, w.buf.items, true);
    defer f.deinit(gpa);

    try std.testing.expectEqualStrings("qwen35", f.get("general.architecture").?.string);
    try std.testing.expectEqual(@as(u32, 24), f.get("qwen35.block_count").?.u32);
    try std.testing.expectEqual(null, f.get("missing"));

    const tokens = f.get("tokenizer.ggml.tokens").?.array;
    try std.testing.expectEqual(@as(u64, 3), tokens.len);
    var it = tokens.iterator();
    try std.testing.expectEqualStrings("a", it.nextString().?);
    try std.testing.expectEqualStrings("bc", it.nextString().?);
    try std.testing.expectEqualStrings("", it.nextString().?);
    try std.testing.expectEqual(null, it.nextString());

    const types = f.get("tokenizer.ggml.token_type").?.array;
    try std.testing.expectEqual(ValueType.i32, types.elem_type);
    try std.testing.expectEqual(@as(i32, -1), @as(i32, @bitCast(types.get(u32, 1))));
    try std.testing.expectEqual(@as(u32, 3), types.get(u32, 2));

    const t = f.tensor("w").?;
    try std.testing.expectEqualSlices(u64, &.{ 4, 2 }, t.shape());
    try std.testing.expectEqual(@as(u64, 32), t.byteSize());
    const data = f.tensorBytes(w.buf.items, t);
    try std.testing.expectEqual(@as(f32, 7), @as(f32, @bitCast(std.mem.readInt(u32, data[28..32], .little))));
    try std.testing.expectEqual(@as(u64, 34), f.tensor("b").?.byteSize());
    try std.testing.expectEqual(@as(u64, 0), f.data_offset % default_alignment);
}

test "custom alignment moves the data section" {
    const gpa = std.testing.allocator;
    var w: TestWriter = .{};
    defer w.buf.deinit(gpa);
    buildTestFile(&w, 64);

    var f = try parse(gpa, w.buf.items, true);
    defer f.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 64), f.alignment);
    try std.testing.expectEqual(@as(u64, 0), f.data_offset % 64);
}

test "every truncation is rejected, never read out of bounds" {
    const gpa = std.testing.allocator;
    var w: TestWriter = .{};
    defer w.buf.deinit(gpa);
    buildTestFile(&w, null);

    for (0..w.buf.items.len) |n| {
        if (parse(gpa, w.buf.items[0..n], true)) |f| {
            var ff = f;
            ff.deinit(gpa);
            return error.TestUnexpectedResult;
        } else |_| {}
    }
}

test "rejects bad magic, version and tensor types" {
    const gpa = std.testing.allocator;
    var w: TestWriter = .{};
    defer w.buf.deinit(gpa);
    buildTestFile(&w, null);
    const bytes = w.buf.items;

    bytes[0] = 'X';
    try std.testing.expectError(error.BadMagic, parse(gpa, bytes, true));
    bytes[0] = 'G';

    bytes[4] = 2;
    try std.testing.expectError(error.UnsupportedVersion, parse(gpa, bytes, true));
    bytes[4] = 3;

    // Find the q8_0 type tag of tensor "b" and replace it with an unknown id.
    const q8: u32 = @backingInt(TensorType.q8_0);
    var needle: [4]u8 = undefined;
    std.mem.writeInt(u32, &needle, q8, .little);
    const at = std.mem.lastIndexOf(u8, bytes[0..dataOffset(bytes)], &needle).?;
    std.mem.writeInt(u32, bytes[at..][0..4], 31, .little);
    try std.testing.expectError(error.BadTensorType, parse(gpa, bytes, true));
}

/// End of the header region of the test file: the first data byte.
fn dataOffset(bytes: []const u8) usize {
    var f = parse(std.testing.allocator, bytes, true) catch unreachable;
    defer f.deinit(std.testing.allocator);
    return @intCast(f.data_offset);
}
