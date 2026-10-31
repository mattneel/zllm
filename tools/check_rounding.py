"""Compare activation rounding with pinned ggml, including every half-integer tie.

A temporary Zig test imports the production reference directly; this avoids adding
an inference CLI operation solely for a diagnostic. Its inputs and expected f32
bits come from ggml's exported quantize/dequantize functions, not from zllm.
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
    return rows.reshape(-1)


def oracle(lib, values: np.ndarray, suffix: str, block: int, block_bytes: int) -> np.ndarray:
    encoded = np.zeros(values.size // block * block_bytes, dtype=np.uint8)
    quantize = getattr(lib, f"quantize_row_{suffix}_ref")
    decode = getattr(lib, f"dequantize_row_{suffix}")
    for fn in (quantize, decode):
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    quantize(values.ctypes.data, encoded.ctypes.data, values.size)
    expected = np.empty_like(values)
    decode(encoded.ctypes.data, expected.ctypes.data, values.size)
    return expected


def main() -> int:
    args = parse_args(__doc__)
    paths = fixture_paths(args.fixtures)
    lib = ctypes.CDLL(str(paths.llama_bin / "libggml-base.so"))
    values = probes()
    with tempfile.TemporaryDirectory(prefix=".rounding-", dir=ROOT) as work:
        temp = Path(work)
        (temp / "input.f32").write_bytes(values.tobytes())
        tests = []
        for suffix, weight, block, size in (("q8_K", "q4_k", 256, 292), ("q8_0", "q8_0", 32, 34)):
            expected = oracle(lib, values, suffix, block, size)
            (temp / f"{suffix}.f32").write_bytes(expected.tobytes())
            tests.append(f'''test "{suffix} activation rounding is bitwise ggml" {{
    const input = @embedFile("{temp.name}/input.f32");
    const expected = @embedFile("{temp.name}/{suffix}.f32");
    var values: [{values.size}]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, input[i * 4 ..][0..4], .little));
    mm.roundLikeLlama(.{weight}, &values);
    for (values, 0..) |v, i| try std.testing.expectEqual(std.mem.readInt(u32, expected[i * 4 ..][0..4], .little), @as(u32, @bitCast(v)));
}}
''')
        with tempfile.NamedTemporaryFile(mode="w", suffix=".zig", prefix=".rounding-", dir=ROOT, delete=False) as driver:
            driver.write('const std = @import("std");\nconst mm = @import("src/ref/matmul.zig");\n' + "\n".join(tests))
            driver_path = Path(driver.name)
        try:
            subprocess.run(["zig", "test", str(driver_path), "-O", "ReleaseSafe", "--test-filter", "activation rounding is bitwise ggml"], cwd=ROOT, check=True)
        finally:
            driver_path.unlink()
    print(f"PASS activation rounding: Q8_K and Q8_0, {values.size} values each, including halfway neighbors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
