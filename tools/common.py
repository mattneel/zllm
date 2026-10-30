"""Shared plumbing for the parity checks: arguments, pinned fixture paths, failure reporting."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Mirrors the pins in scripts/fixtures.sh.
GGUF_FILE = "Qwen3.5-0.8B-BF16.gguf"
HEADER_FILE = "Qwen3.5-0.8B-BF16.header.gguf"
# The quantized model check_quant.py uses by default (fixtures.sh model27b).
GGUF27_FILE = "RVN-Q3_K_M-mtp.gguf"
HF_DIR = "hf/Qwen3.5-0.8B"
CHAT_TEMPLATE = "hf/Qwen-Fixed-Chat-Templates/chat_template.jinja"


@dataclass(frozen=True)
class DenseModel:
    """One of scripts/fixtures.sh's dense_models."""

    name: str
    file: str

    def gguf(self, root: Path) -> Path:
        return root / "models" / self.file

    def gguf_header(self, root: Path) -> Path:
        return root / "models" / self.file.replace(".gguf", ".header.gguf")

    def hf_tokenizer(self, root: Path) -> Path:
        return root / "hf" / self.name / "tokenizer.json"


# Mirrors dense_models in scripts/fixtures.sh.
DENSE_MODELS = (
    DenseModel("qwen3", "Qwen3-0.6B-BF16.gguf"),
    DenseModel("llama", "Llama-3.2-1B-Instruct-BF16.gguf"),
    DenseModel("granite", "granite-4.2-3b-bf16.gguf"),
)


@dataclass(frozen=True)
class FixturePaths:
    root: Path

    @property
    def gguf(self) -> Path:
        return self.root / "models" / GGUF_FILE

    @property
    def gguf_header(self) -> Path:
        return self.root / "models" / HEADER_FILE

    @property
    def hf_tokenizer(self) -> Path:
        return self.root / HF_DIR / "tokenizer.json"

    @property
    def chat_template(self) -> Path:
        return self.root / CHAT_TEMPLATE

    @property
    def llama_bin(self) -> Path:
        return self.root / "llama.cpp" / "build" / "bin"

    @property
    def gguf_py(self) -> Path:
        return self.root / "llama.cpp" / "gguf-py"


def gguf_metadata(path: Path) -> dict:
    """A GGUF's metadata (key → value), from its header alone (gguf-py's reader maps tensor data too)."""
    import struct

    data = path.read_bytes()
    pos = 0
    scalars = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def read(fmt: str):
        nonlocal pos
        (v,) = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return v

    def string() -> str:
        nonlocal pos
        n = read("<Q")
        s = data[pos : pos + n].decode("utf-8", "surrogateescape")
        pos += n
        return s

    def value(t: int):
        if t == 8:
            return string()
        if t == 9:
            et, n = read("<I"), read("<Q")
            return [value(et) for _ in range(n)]
        return read(scalars[t])

    if data[:4] != b"GGUF":
        raise SystemExit(f"{path} is not a GGUF file")
    pos = 8
    read("<Q")
    n_kv = read("<Q")
    meta = {}
    for _ in range(n_kv):
        key = string()
        meta[key] = value(read("<I"))
    return meta


def import_pinned_gguf(paths: FixturePaths):
    """gguf-py from the pinned llama.cpp checkout, not whatever is on PyPI."""
    if not paths.gguf_py.is_dir():
        raise SystemExit(f"llama.cpp checkout not found at {paths.gguf_py.parent}; run scripts/fixtures.sh llama")
    sys.path.insert(0, str(paths.gguf_py))
    import gguf

    return gguf


def parse_args(description: str | None, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--zllm", required=True, help="path to the zllm executable")
    p.add_argument("--fixtures", required=True, type=Path, help="fixture directory from scripts/fixtures.sh")
    if extra is not None:
        extra(p)
    return p.parse_args()


def fixture_paths(root: Path) -> FixturePaths:
    paths = FixturePaths(root)
    if not root.is_dir():
        raise SystemExit(f"fixtures not found at {root}; run scripts/fixtures.sh")
    return paths


class Failures:
    """Collects mismatches so one run reports all of them, capped in output."""

    max_shown = 20

    def __init__(self) -> None:
        self.items: list[str] = []

    def expect(self, what: str, ok: bool, detail: str = "") -> None:
        if not ok:
            self.items.append(f"{what}: {detail}")

    def expect_eq(self, what: str, got, want) -> None:
        self.expect(what, got == want, f"got {got!r}, want {want!r}")

    def report(self) -> int:
        if not self.items:
            print("PASS")
            return 0
        for item in self.items[: self.max_shown]:
            print(f"FAIL {item}")
        if len(self.items) > self.max_shown:
            print(f"... and {len(self.items) - self.max_shown} more")
        print(f"FAIL ({len(self.items)} mismatches)")
        return 1
