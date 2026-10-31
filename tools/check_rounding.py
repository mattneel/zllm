"""Compare activation rounding and RMSNorm with the pinned actual x86 ggml CPU.

A temporary Zig test imports the production reference directly. Q8_0's oracle
is libggml-cpu quantize_row_q8_0, NOT the scalar file quantizer in libggml-base;
Q8_K still uses libggml-base quantize_row_q8_K_ref. Both decode through base.
RMSNorm expected bits explicitly reproduce ggml-cpu/ops.cpp's f32 squares,
sequential f64 sum/division, f32 mean/sqrt/reciprocal and ordered x*scale*w.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import fixture_paths, parse_args

ROOT = Path(__file__).resolve().parent.parent


def probes() -> np.ndarray:
    rng = np.random.default_rng(743)
    rows = rng.standard_normal((32, 256)).astype(np.float32)
    # Exact half steps, their immediate neighbors, both signs and both parities.
    ties = np.arange(-126.5, 127, 1, dtype=np.float32)
    for i, values in enumerate((ties, np.nextafter(ties, -np.inf), np.nextafter(ties, np.inf))):
        rows[i, : len(values)] = values
        rows[i, 255] = 127
    # Ensure each Q8_0 block has the same exactly representable unit scale too.
    for start in range(0, 256, 32):
        rows[3:6, start : start + 32] = np.arange(-15.5, 16, 1, dtype=np.float32)
        rows[3:6, start] = 127
    rows[4] = np.nextafter(rows[4], -np.inf)
    rows[5] = np.nextafter(rows[5], np.inf)
    rows[6] = 0
    rows[7] = -rows[3]
    # Non-unit scale where 127/amax and 1/(amax/127) straddle a half step.
    rows[8] = 0
    rows[8, 0] = np.array(0x3F80000F, dtype=np.uint32).view(np.float32)
    rows[8, 1] = np.array(0x3CA14298, dtype=np.uint32).view(np.float32)
    rows[8, 2] = -rows[8, 1]
    # Exhaust every Q8_0 half-integer and its immediate neighbors at exact
    # unit scale, with an untouched 127 anchor in every 32-element block.
    blocks = rows[9:15].reshape(-1, 32)
    for variant, samples in enumerate((ties, np.nextafter(ties, -np.inf), np.nextafter(ties, np.inf))):
        for index, start in enumerate(range(0, len(samples), 31)):
            block = blocks[variant * 9 + index]
            block[:] = 0
            block[0] = 127
            chunk = samples[start : start + 31]
            block[1 : 1 + len(chunk)] = chunk
    return rows.reshape(-1)


def oracle(quantize, decode, values: np.ndarray, block: int, block_bytes: int) -> np.ndarray:
    encoded = np.zeros(values.size // block * block_bytes, dtype=np.uint8)
    for fn in (quantize, decode):
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    quantize(values.ctypes.data, encoded.ctypes.data, values.size)
    expected = np.empty_like(values)
    decode(encoded.ctypes.data, expected.ctypes.data, values.size)
    return expected


def norm_probes(width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7406 + width)
    values = rng.standard_normal((8, width)).astype(np.float32)
    weights = rng.uniform(-2, 2, values.shape).astype(np.float32)
    values[0] = 0
    # Small squares disappear in an f32 reduction but not ggml's f64 sum.
    values[1] = np.float32(0.25)
    values[1, 0] = np.float32(4096)
    # Non-dyadic values expose incorrectly promoted f64 squares.
    values[2] = np.linspace(-1.7, 2.3, width, dtype=np.float32)
    return values, weights


def norm_oracle(values: np.ndarray, weights: np.ndarray, eps: np.float32) -> np.ndarray:
    expected = np.empty_like(values)
    for row, weight, out in zip(values, weights, expected):
        squares = np.multiply(row, row, dtype=np.float32)
        total = sum(float(square) for square in squares)
        mean = np.float32(total / row.size)
        root = np.sqrt(np.float32(mean + eps))
        scale = np.float32(np.float32(1) / root)
        np.multiply(np.multiply(row, scale, dtype=np.float32), weight, out=out)
    return expected


def main() -> int:
    args = parse_args(__doc__)
    paths = fixture_paths(args.fixtures)
    base = ctypes.CDLL(str(paths.llama_bin / "libggml-base.so"))
    cpu = ctypes.CDLL(str(paths.llama_bin / "libggml-cpu.so"))
    values = probes()
    with tempfile.TemporaryDirectory(prefix=".rounding-", dir=ROOT) as work:
        temp = Path(work)
        (temp / "input.f32").write_bytes(values.tobytes())
        tests = []
        for suffix, weight, block, size, quantize in (
            ("q8_K", "q4_k", 256, 292, base.quantize_row_q8_K_ref),
            ("q8_0", "q8_0", 32, 34, cpu.quantize_row_q8_0),
        ):
            decode = getattr(base, f"dequantize_row_{suffix}")
            expected = oracle(quantize, decode, values, block, size)
            if suffix == "q8_0":
                scalar = oracle(base.quantize_row_q8_0_ref, decode, values, block, size)
                if np.array_equal(expected.view(np.uint32), scalar.view(np.uint32)):
                    raise AssertionError("Q8_0 probes must distinguish active x86 CPU from scalar file quantization")
                if expected[3 * 256 + 18] != 2:
                    raise AssertionError("active x86 Q8_0 oracle must round unit-scale +2.5 to even")
            (temp / f"{suffix}.f32").write_bytes(expected.tobytes())
            tests.append(f'''test "{suffix} activation rounding is bitwise pinned ggml" {{
    const input = @embedFile("{temp.name}/input.f32");
    const expected = @embedFile("{temp.name}/{suffix}.f32");
    var values: [{values.size}]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, input[i * 4 ..][0..4], .little));
    mm.roundLikeLlama(.{weight}, &values);
    for (values, 0..) |v, i| try std.testing.expectEqual(std.mem.readInt(u32, expected[i * 4 ..][0..4], .little), @as(u32, @bitCast(v)));
}}
''')
        for width in (17, 32, 256, 2048, 4096):
            norm_values, weights = norm_probes(width)
            expected = norm_oracle(norm_values, weights, np.float32(1e-5))
            for suffix, array in (("input", norm_values), ("weight", weights), ("expected", expected)):
                (temp / f"norm-{width}-{suffix}.f32").write_bytes(array.tobytes())
            tests.append(f'''test "RMSNorm width {width} is bitwise pinned ggml" {{
    var values = floats({norm_values.size}, @embedFile("{temp.name}/norm-{width}-input.f32"));
    const weights = floats({weights.size}, @embedFile("{temp.name}/norm-{width}-weight.f32"));
    const expected = @embedFile("{temp.name}/norm-{width}-expected.f32");
    for (0..{norm_values.shape[0]}) |row| {{
        ops.rmsNorm(values[row * {width} ..][0..{width}], weights[row * {width} ..][0..{width}], 1e-5);
    }}
    for (values, 0..) |v, i| try std.testing.expectEqual(std.mem.readInt(u32, expected[i * 4 ..][0..4], .little), @as(u32, @bitCast(v)));
}}
''')
        with tempfile.NamedTemporaryFile(mode="w", suffix=".zig", prefix=".rounding-", dir=ROOT, delete=False) as driver:
            driver.write('''const std = @import("std");
const mm = @import("src/ref/matmul.zig");
const ops = @import("src/ref/ops.zig");
fn floats(comptime count: usize, bytes: []const u8) [count]f32 {
    var values: [count]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return values;
}
''' + "\n".join(tests))
            driver_path = Path(driver.name)
        try:
            subprocess.run(["zig", "test", str(driver_path), "-O", "ReleaseSafe", "--test-filter", "bitwise pinned ggml"], cwd=ROOT, check=True)
        finally:
            driver_path.unlink()
    print(f"PASS bitwise pinned ggml: active x86 Q8_0 and base-ref Q8_K ({values.size} values each), RMSNorm widths 17/32/256/2048/4096")
    return 0


if __name__ == "__main__":
    sys.exit(main())
