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
