"""Compare zllm's CPU dequantization with ggml's, bit for bit.

For each quantized type the GPU path will read (Q8_0, Q3_K, Q4_K, Q5_K,
Q6_K), random blocks are decoded by `zllm dequant` and by the pinned
llama.cpp build's dequantize_row_* (libggml-base, through ctypes). Every
byte of every block is random, so every quant, scale and high-bit pattern
occurs, except the f16 super-scales (d, dmin), which are drawn as finite
values: NaN payloads are not part of the format's meaning.

The two must agree exactly: zllm's decoders follow ggml's order of f32
operations.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from common import Failures, fixture_paths, parse_args

BLOCKS = 4096
SEED = 1234

# type: (block bytes, elements per block, byte offsets of f16 super-scales)
TYPES = {
    "q8_0": (34, 32, [0]),
    "q3_k": (110, 256, [108]),
    "q4_k": (144, 256, [0, 2]),
    "q5_k": (176, 256, [0, 2]),
    "q6_k": (210, 256, [208]),
}
GGML_NAMES = {"q8_0": "q8_0", "q3_k": "q3_K", "q4_k": "q4_K", "q5_k": "q5_K", "q6_k": "q6_K"}


def random_blocks(rng: np.random.Generator, block_bytes: int, scale_offsets: list[int]) -> np.ndarray:
    blocks = rng.integers(0, 256, size=(BLOCKS, block_bytes), dtype=np.uint8)
    for off in scale_offsets:
        # Finite f16 super-scales across magnitudes, both signs.
        scales = (rng.standard_normal(BLOCKS) * 10.0 ** rng.integers(-4, 2, BLOCKS)).astype(np.float16)
        blocks[:, off : off + 2] = scales.view(np.uint8).reshape(BLOCKS, 2)
    return blocks


def main() -> int:
    args = parse_args(__doc__)
    paths = fixture_paths(args.fixtures)
    lib_path = paths.llama_bin / "libggml-base.so"
    if not lib_path.is_file():
        raise SystemExit(f"{lib_path} not found; run scripts/fixtures.sh llama")
    ggml = ctypes.CDLL(str(lib_path))
    rng = np.random.default_rng(SEED)
    failures = Failures()

    with tempfile.TemporaryDirectory() as tmp:
        for name, (block_bytes, block_elems, scale_offsets) in TYPES.items():
            blocks = random_blocks(rng, block_bytes, scale_offsets)
            n = BLOCKS * block_elems

            want = np.empty(n, dtype=np.float32)
            fn = getattr(ggml, f"dequantize_row_{GGML_NAMES[name]}")
            fn.restype = None
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            fn(blocks.ctypes.data, want.ctypes.data, n)

            src = Path(tmp) / f"{name}.bin"
            dst = Path(tmp) / f"{name}.f32"
            src.write_bytes(blocks.tobytes())
            subprocess.run([args.zllm, "dequant", name, str(src), str(dst)], check=True)
            got = np.fromfile(dst, dtype=np.float32)

            same = got.shape == want.shape and np.array_equal(got.view(np.uint32), want.view(np.uint32))
            detail = ""
            if not same and got.shape == want.shape:
                bad = np.flatnonzero(got.view(np.uint32) != want.view(np.uint32))
                i = int(bad[0])
                detail = f"{bad.size} of {n} values differ; first at {i} (block {i // block_elems}): zllm {got[i]!r}, ggml {want[i]!r}"
            failures.expect(f"{name}: {n} values bitwise identical to ggml", same, detail)
            print(f"{'PASS' if same else 'FAIL'} {name:<5} {BLOCKS} random blocks, {n} values" + ("" if same else f": {detail}"))

    return failures.report()


if __name__ == "__main__":
    sys.exit(main())
