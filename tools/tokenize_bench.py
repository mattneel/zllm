"""Warm encode-only zllm/Hugging Face/llama.cpp comparison; emit one JSON report.

Run with uv run --project tools tools/tokenize_bench.py --help.
Model loading, corpus parsing, compilation, IPC and parity checks are untimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tools" / "corpus" / "tokenizer.jsonl"
SAMPLE_NS = 500_000_000


def read_corpus(path: Path) -> list[str]:
    # Only LF delimits records: str.splitlines() corrupts U+2028 and NEL cases.
    texts = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError(f"{path}:{number}: expected an object with string text")
        record["text"].encode("utf-8")  # Reject unpaired JSON surrogates.
        texts.append(record["text"])
    if not texts or not sum(len(text.encode("utf-8")) for text in texts):
        raise ValueError("corpus must contain at least one nonempty UTF-8 text")
    return texts


def validate_ids(got: object, expected: list[list[int]]) -> dict:
    """Reject malformed output and report exact per-record id mismatches."""
    if not isinstance(got, list) or len(got) != len(expected):
        raise ValueError("token id record count differs from the corpus")
    mismatches = []
    for number, (ids, want) in enumerate(zip(got, expected, strict=True), 1):
        if not isinstance(ids, list) or any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in ids):
            raise ValueError(f"record {number}: malformed token ids")
        if ids != want:
            mismatches.append(number)
    return {"ids_match_hf": not mismatches, "mismatched_records": mismatches}


def validate_report(report: object, engine: str, texts: list[str], corpus_tokens: int) -> dict:
    """Fail closed on fabricated rates, unit drift, or incorrect work counts."""
    if not isinstance(report, dict) or report.get("engine") != engine:
        raise ValueError(f"expected {engine} benchmark object")
    positive = ("corpus_records", "corpus_bytes", "repetitions", "elapsed_ns", "bytes")
    for key in (*positive, "corpus_tokens", "tokens"):
        value = report.get(key)
        if type(value) is not int or value < (1 if key in positive else 0):
            raise ValueError(f"{engine}: invalid {key}")
    expected = {
        "corpus_records": len(texts),
        "corpus_bytes": sum(len(text.encode("utf-8")) for text in texts),
        "corpus_tokens": corpus_tokens,
    }
    expected["bytes"] = expected["corpus_bytes"] * report["repetitions"]
    expected["tokens"] = corpus_tokens * report["repetitions"]
    for key, value in expected.items():
        if report[key] != value:
            raise ValueError(f"{engine}: {key} is {report[key]}, expected {value}")
    if report["elapsed_ns"] < SAMPLE_NS:
        raise ValueError(f"{engine}: sample shorter than 0.5 seconds")
    seconds = report["elapsed_ns"] / 1e9
    for key, want in (("mb_per_second", report["bytes"] / seconds / 1e6), ("tokens_per_second", report["tokens"] / seconds)):
        value = report.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or not math.isclose(value, want, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"{engine}: invalid or inconsistent {key}")
    return report


def benchmark_hf(tokenizer, texts: list[str], expected: list[list[int]]) -> dict:
    corpus_tokens = sum(map(len, expected))  # Computing expected ids was the warm pass.
    repetitions = tokens = elapsed_ns = 0
    start = time.perf_counter_ns()
    while elapsed_ns < SAMPLE_NS:
        for text in texts:
            tokens += len(tokenizer.encode(text).ids)
        repetitions += 1
        elapsed_ns = time.perf_counter_ns() - start
    corpus_bytes = sum(len(text.encode("utf-8")) for text in texts)
    seconds = elapsed_ns / 1e9
    return {
        "engine": "huggingface", "corpus_records": len(texts),
        "corpus_bytes": corpus_bytes, "corpus_tokens": corpus_tokens,
        "repetitions": repetitions, "elapsed_ns": elapsed_ns,
        "bytes": corpus_bytes * repetitions, "tokens": tokens,
        "mb_per_second": corpus_bytes * repetitions / seconds / 1e6,
        "tokens_per_second": tokens / seconds,
    }


def run_json(command: list[str]) -> object:
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


def build_llama(checkout: Path, cache: Path, cxx: str) -> Path:
    """Build the library if absent, then our warm-loop executable (not llama-tokenize)."""
    if not (checkout / "include" / "llama.h").is_file():
        raise FileNotFoundError(f"missing llama.cpp checkout: {checkout}")
    compiler = shlex.split(cxx)
    if not compiler or shutil.which(compiler[0]) is None:
        raise FileNotFoundError(f"C++ compiler unavailable: {cxx}")
    build = checkout / "build"
    lib = build / "bin"
    library_name = "libllama.dylib" if sys.platform == "darwin" else "libllama.so"
    if not (lib / library_name).is_file():
        subprocess.run(["cmake", "-S", str(checkout), "-B", str(build), "-DBUILD_SHARED_LIBS=ON", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DLLAMA_CURL=OFF"], check=True, capture_output=True, text=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "llama", "-j2"], check=True, capture_output=True, text=True)
    cache.mkdir(parents=True, exist_ok=True)
    executable = cache / "llama-tokenize-bench"
    subprocess.run([*compiler, "-O3", "-std=c++17", str(ROOT / "tools" / "tokenize-bench.cpp"), "-I", str(checkout / "include"), "-I", str(checkout / "ggml" / "include"), "-L", str(lib), f"-Wl,-rpath,{lib}", "-lllama", "-o", str(executable)], check=True, capture_output=True, text=True)
    return executable


def benchmark_llama(args, texts: list[str], expected: list[list[int]]) -> dict:
    try:
        executable = build_llama(args.llama_cpp.resolve(), args.cache.resolve(), args.cxx)
    except (OSError, subprocess.CalledProcessError) as error:
        reason = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        return {"engine": "llama.cpp", "status": "unavailable", "reason": reason or str(error)}
    # Length-prefixing preserves embedded NUL, newline, NEL, and empty records.
    with tempfile.TemporaryDirectory(prefix="tokenize-bench-") as directory:
        corpus = Path(directory) / "corpus.bin"
        with corpus.open("wb") as output:
            for text in texts:
                raw = text.encode("utf-8")
                output.write(struct.pack("<I", len(raw)))
                output.write(raw)
        result = run_json([str(executable), str(args.model.resolve()), str(corpus)])
    if not isinstance(result, dict):
        raise ValueError("expected llama.cpp benchmark object")
    ids = result.pop("ids", None)
    parity = validate_ids(ids, expected)
    report = validate_report(result, "llama.cpp", texts, sum(map(len, ids)))
    return {**report, **parity, "status": "measured"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="GGUF model or metadata-only header")
    parser.add_argument("hf_tokenizer", type=Path, help="matching Hugging Face tokenizer.json")
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--zllm", type=Path, default=ROOT / "zig-out" / "bin" / "zllm")
    parser.add_argument("--label", default="zllm", help="identify the old/new binary in saved results")
    parser.add_argument("--llama-cpp", type=Path, default=ROOT / ".fixtures" / "llama.cpp")
    parser.add_argument("--cache", type=Path, default=ROOT / ".fixtures" / "cache" / "tokenize-bench")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--skip-llama", action="store_true", help="explicitly omit llama.cpp instead of attempting its build")
    args = parser.parse_args()
    from tokenizers import Tokenizer, __version__ as tokenizers_version

    texts = read_corpus(args.corpus)
    hf = Tokenizer.from_file(str(args.hf_tokenizer))
    expected = [hf.encode(text).ids for text in texts]
    corpus_tokens = sum(map(len, expected))
    # This separate invocation checks ids outside the executable's timed loop.
    output = subprocess.run([str(args.zllm.resolve()), "tokenize", str(args.model.resolve()), str(args.corpus.resolve())], check=True, capture_output=True, text=True, encoding="utf-8").stdout
    ids = [json.loads(line)["ids"] for line in output.split("\n") if line]
    parity = validate_ids(ids, expected)
    zllm = run_json([str(args.zllm.resolve()), "tokenize-bench", str(args.model.resolve()), str(args.corpus.resolve())])
    zllm = validate_report(zllm, "zllm", texts, sum(map(len, ids)))
    hf_result = validate_report(benchmark_hf(hf, texts, expected), "huggingface", texts, corpus_tokens)
    results = [{**zllm, **parity, "label": args.label, "status": "measured"}, {**hf_result, "version": tokenizers_version, "status": "measured", "ids_match_hf": True}]
    results.append({"engine": "llama.cpp", "status": "skipped", "reason": "--skip-llama"} if args.skip_llama else benchmark_llama(args, texts, expected))
    report = {
        "model": str(args.model.resolve()), "hf_tokenizer": str(args.hf_tokenizer.resolve()),
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "hf_tokenizer_sha256": hashlib.sha256(args.hf_tokenizer.read_bytes()).hexdigest(),
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "sample_target_ns": SAMPLE_NS, "results": results,
    }
    print(json.dumps(report, allow_nan=False))
    return int(any(result.get("ids_match_hf") is False for result in results))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"tokenize-bench: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        sys.exit(1)
