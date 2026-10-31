"""Compare the GPU forward pass with the CPU f32 reference.

For each eval text in tools/corpus/eval.jsonl, the same token ids run through
`zllm forward` (CPU reference, itself checked against llama.cpp by
check_forward.py) and `zllm gpu-forward` on a Dawn host, which processes them
as steps against the KV cache and the recurrent state.

With a model that has an MTP block, every run also produces `mtp_logits` (the
block's drafts, one row per position but the last), held to the same bounds as
the logits.

Pass criteria (docs/src/validation.md):
  * f32 KV cache, 512-token steps: both use the same bf16 weights and f32
    activations, so they may differ only by reduction order and
    transcendental-function rounding. Top-1 agreement 100%; every layer
    output, the final hidden state and the logits: cosine >= 0.999999 and
    relative L2 error <= 1e-4 at every position.
  * The prefill kernels in 7-token steps: bitwise identical to the 512-token
    run. Every row's arithmetic is independent of how a sequence is split.
  * The decode kernels (matrix-vector multiplies, split attention) in
    1-token steps, as decoding runs, over the first DECODE_TOKENS positions:
    the same bounds and top-1 agreement as the 512-token run; and in 3-token
    steps, bitwise identical to the 1-token run.
  * f16 and FP8 KV caches (KV_GATES: their top-1 agreement and min cosine
    at every position), on every eval text and each dense model's first.
  * `zllm engine-check`: requests that reuse the resident sequence or a
    checkpoint give the logits of fresh runs (bitwise where both runs use the
    same kernels for each position).
  * Each dense model (scripts/fixtures.sh dense), on the first eval text: the
    512-token run's bounds and top-1 agreement on both kernel paths (the
    decode kernels over its first DECODE_TOKENS positions).

Hosts: `native` runs the Linux build on the Vulkan adapter Dawn finds
(llvmpipe in CI); `d3d12` runs the Windows build through scripts/d3d12.sh;
`velocity` runs it there on the Velocity backend (ZLLM_BACKEND=velocity).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from check_forward import read_records, row_stats
from common import DENSE_MODELS, Failures, cached_reference, fixture_paths, parse_args

EVAL = Path(__file__).parent / "corpus" / "eval.jsonl"
ROOT = Path(__file__).resolve().parent.parent

COSINE_MIN = 0.999999
REL_L2_MAX = 1e-4
# KV cache format -> (top-1 agreement min, cosine min at every position),
# against the f32 reference. FP8 (E4M3, 3 mantissa bits) measured 97.4%
# top-1 and a 0.982 worst cosine over these texts and models on the RTX
# 5090: its bounds catch a broken path; whether a model may default to it
# is the precision ladder's gate (docs/src/spec.md §9).
KV_GATES = {"f16": (0.99, 0.99998), "fp8": (0.95, 0.975)}
# (tokens per step, token limit) for the runs that must match bitwise.
# Positions checked through the decode kernels (one step per token).
DECODE_TOKENS = 40


def win(path: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(path)], check=True, capture_output=True, text=True).stdout.strip()


# Where GPU checks run: this build on Dawn's adapter, the Windows build on
# Dawn's D3D12, or the Windows build on Velocity.
HOSTS = ["native", "d3d12", "velocity"]


def zllm_on_host(args, paths, command: str, *argv: Path | str) -> subprocess.CompletedProcess:
    """Runs a zllm command on the selected host; Path arguments are mapped to the host."""
    if args.host in HOSTS[1:]:
        mapped = [win(a) if isinstance(a, Path) else a for a in argv]
        cmd = [str(ROOT / "scripts" / "d3d12.sh"), command, *mapped]
        env = {**os.environ, "ZLLM_BACKEND": "velocity" if args.host == "velocity" else "webgpu"}
    else:
        cmd = [args.zllm, command, *map(str, argv)]
        env = {**os.environ, "ZLLM_DAWN": str(paths.root / "dawn" / "libwebgpu_dawn.so")}
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


class GpuRuns:
    """gpu-forward runs on one model, executed in one process: the model
    loads once for all of them (`gpu-forward --runs`). On the Windows hosts
    the outputs go to a scratch directory beside the Windows stage: a
    Windows process writes WSL's filesystem slowly (9P), and the outputs
    hold whole logits rows, hundreds of MB a run. Use as a context manager:
    the scratch directory goes on exit."""

    def __init__(self, args, paths, model: Path, tmp: Path):
        self.args, self.paths, self.model, self.tmp = args, paths, model, tmp
        self.runs: list[tuple[Path, Path, tuple[str, ...]]] = []
        self.windows = args.host in HOSTS[1:]
        if self.windows:
            stage = Path(subprocess.run(["wslpath", "-u", (paths.root / "dawn" / "win_stage").read_text().strip()], check=True, capture_output=True, text=True).stdout.strip())
            self.out_dir = Path(tempfile.mkdtemp(prefix="check-", dir=stage))
        else:
            self.out_dir = tmp

    def __enter__(self) -> GpuRuns:
        return self

    def __exit__(self, *_) -> None:
        if self.windows:
            shutil.rmtree(self.out_dir, ignore_errors=True)

    def add(self, tokens: Path, name: str, *opts: str) -> Path:
        """Adds a run writing records named `name`; returns where they will be."""
        out = self.out_dir / name
        self.runs.append((tokens, out, opts))
        return out

    def run(self) -> dict[Path, str]:
        """Executes the runs; returns each output's timing line."""
        host_path = win if self.windows else str
        runs_file = self.tmp / f"runs-{self.model.stem}.txt"
        runs_file.write_text("".join(" ".join([host_path(t), host_path(o), *opts]) + "\n" for t, o, opts in self.runs))
        proc = zllm_on_host(self.args, self.paths, "gpu-forward", self.model, "--runs", runs_file)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout + proc.stderr)
            raise SystemExit(f"gpu-forward failed ({self.args.host})")
        timing = [line for line in proc.stderr.replace("\r", "").splitlines() if line.startswith("gpu-forward:") and "loaded in" not in line]
        return {o: t for (_, o, _), t in zip(self.runs, timing, strict=True)}


def main() -> int:
    args = parse_args(__doc__, lambda p: p.add_argument("--host", choices=HOSTS, default="native"))
    paths = fixture_paths(args.fixtures)
    hf = Tokenizer.from_file(str(paths.hf_tokenizer))
    evals = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").split("\n") if line]

    fail = Failures()
    total = agree = decode_total = decode_agree = 0
    worst = {"prefill": [1.0, 0.0], "decode": [1.0, 0.0]}
    kv_stats = {fmt: {"agree": 0, "total": 0, "cos": 1.0} for fmt in KV_GATES}

    def compare(label: str, path: str, gpu: dict, ref: dict, rows: int) -> int:
        """Bounds against the reference for the first `rows` positions; returns top-1 agreement."""
        print(f"  {'tensor':<14} {'min cos':>10} {'max rel L2':>11}   [{path} kernels]")
        names = sorted((n for n in ref if n.startswith("l_out-")), key=lambda n: int(n.split("-")[1]))
        for name in [*names, "result_norm", "result_output", "mtp_logits"]:
            if name not in gpu:
                continue
            cos, rel, _ = row_stats(gpu[name], ref[name][: len(gpu[name])])
            print(f"  {name:<14} {cos.min():>10.7f} {rel.max():>11.2e}")
            worst[path][0] = min(worst[path][0], float(cos.min()))
            worst[path][1] = max(worst[path][1], float(rel.max()))
            for p in np.flatnonzero(cos < COSINE_MIN):
                fail.expect(f"{label}: {name} cosine at {p}", False, f"{cos[p]:.7f} < {COSINE_MIN}")
            for p in np.flatnonzero(rel > REL_L2_MAX):
                fail.expect(f"{label}: {name} rel L2 at {p}", False, f"{rel[p]:.2e} > {REL_L2_MAX}")
        top = int((gpu["result_output"].argmax(axis=1) == ref["result_output"][:rows].argmax(axis=1)).sum())
        print(f"  top-1 agreement {top}/{rows}")
        if "mtp_logits" in gpu:
            draft = gpu["mtp_logits"]
            top_mtp = int((draft.argmax(axis=1) == ref["mtp_logits"][: len(draft)].argmax(axis=1)).sum())
            print(f"  MTP top-1 agreement {top_mtp}/{len(draft)}")
            fail.expect(f"{label}: MTP top-1 agreement", top_mtp == len(draft), f"{top_mtp}/{len(draft)}")
        return top

    def kv_formats(label: str, ref: dict, rows: int, outs: dict[str, Path], timing: dict[Path, str]) -> None:
        """The 512-token prefill runs with each KV_GATES cache format, against the reference."""
        for fmt, (_, cos_min) in KV_GATES.items():
            got = read_records(outs[fmt])
            cos = min(float(row_stats(got[n], ref[n])[0].min()) for n in ref)
            top = int((got["result_output"].argmax(axis=1) == ref["result_output"][:rows].argmax(axis=1)).sum())
            s = kv_stats[fmt]
            s["agree"] += top
            s["total"] += rows
            s["cos"] = min(s["cos"], cos)
            print(f"  {fmt} KV: top-1 agreement {top}/{rows}, min cosine {cos:.7f}  [{timing[outs[fmt]]}]")
            fail.expect(f"{label}: {fmt} KV cosine", cos >= cos_min, f"{cos:.7f} < {cos_min}")

    def bitwise(a: dict, b: dict, rows: int) -> bool:
        return all(np.array_equal(a[n].view(np.uint32), b[n][: len(a[n])].view(np.uint32)) for n in b)

    with tempfile.TemporaryDirectory() as tmp_name, contextlib.ExitStack() as stack:
        tmp = Path(tmp_name)
        # Every eval text's runs on the 0.8B in one gpu-forward process.
        runs = stack.enter_context(GpuRuns(args, paths, paths.gguf, tmp))
        cases = []
        for ev in evals:
            ids = hf.encode(ev["text"]).ids
            name = ev["name"]
            tok = tmp / f"{name}.i32"
            tok.write_bytes(np.asarray(ids, dtype="<i4").tobytes())
            ref_path = tmp / f"{name}-ref.bin"
            cached_reference(paths, "zllm-forward --mtp", paths.gguf, tok, ref_path, [args.zllm, "forward", str(paths.gguf), str(tok), str(ref_path), "--mtp"])
            mtp = ("--mtp",) if "mtp_logits" in read_records(ref_path) else ()
            part = tmp / f"{name}-part.i32"
            part.write_bytes(np.asarray(ids[:DECODE_TOKENS], dtype="<i4").tobytes())
            outs = {
                "gpu": runs.add(tok, f"{name}-gpu.bin", "--chunk", "512", "--kv", "f32", "--path", "prefill", *mtp),
                "split": runs.add(tok, f"{name}-split.bin", "--chunk", "7", "--kv", "f32", "--path", "prefill", *mtp),
                "one": runs.add(part, f"{name}-one.bin", "--chunk", "1", "--kv", "f32", "--path", "decode", *mtp),
                "three": runs.add(part, f"{name}-three.bin", "--chunk", "3", "--kv", "f32", "--path", "decode", *mtp),
            }
            for fmt in KV_GATES:
                outs[fmt] = runs.add(tok, f"{name}-{fmt}.bin", "--chunk", "512", "--kv", fmt, "--path", "prefill", *mtp)
            cases.append((name, ids, ref_path, outs))
        timing = runs.run()
        for name, ids, ref_path, outs in cases:
            ref = read_records(ref_path)
            gpu = read_records(outs["gpu"])
            fail.expect_eq(f"{name}: tensor names", sorted(gpu), sorted(ref))
            print(f"\n{name}: {len(ids)} tokens  [{timing[outs['gpu']]}]")
            agree += compare(name, "prefill", gpu, ref, len(ids))
            total += len(ids)
            same = bitwise(read_records(outs["split"]), gpu, len(ids))
            print(f"  prefill kernels, steps of 7: {'bitwise identical' if same else 'DIFFERENT'}  [{timing[outs['split']]}]")
            fail.expect(f"{name}: prefill steps of 7 bitwise", same, "outputs differ from 512-token steps")
            one = read_records(outs["one"])
            print(f"  decode kernels, steps of 1, first {DECODE_TOKENS} tokens  [{timing[outs['one']]}]")
            decode_agree += compare(f"{name} (decode)", "decode", one, ref, DECODE_TOKENS)
            decode_total += DECODE_TOKENS
            same = bitwise(read_records(outs["three"]), one, DECODE_TOKENS)
            print(f"  decode kernels, steps of 3: {'bitwise identical' if same else 'DIFFERENT'}  [{timing[outs['three']]}]")
            fail.expect(f"{name}: decode steps of 3 bitwise", same, "outputs differ from 1-token steps")
            kv_formats(name, ref, len(ids), outs, timing)

        # The dense models (graph/dense.zig) run the same program with
        # attention layers only: the first eval sequence on each kernel path,
        # each model's runs in one process.
        for model in DENSE_MODELS:
            gguf = model.gguf(paths.root)
            if not gguf.is_file():
                raise SystemExit(f"missing {gguf}; run scripts/fixtures.sh dense")
            ids = Tokenizer.from_file(str(model.hf_tokenizer(paths.root))).encode(evals[0]["text"]).ids
            tok = tmp / f"{model.name}.i32"
            tok.write_bytes(np.asarray(ids, dtype="<i4").tobytes())
            ref_path = tmp / f"{model.name}-ref.bin"
            cached_reference(paths, "zllm-forward", gguf, tok, ref_path, [args.zllm, "forward", str(gguf), str(tok), str(ref_path)])
            ref = read_records(ref_path)
            part = tmp / f"{model.name}-part.i32"
            part.write_bytes(np.asarray(ids[:DECODE_TOKENS], dtype="<i4").tobytes())
            runs = stack.enter_context(GpuRuns(args, paths, gguf, tmp))
            outs = {
                "gpu": runs.add(tok, f"{model.name}-gpu.bin", "--chunk", "512", "--kv", "f32", "--path", "prefill"),
                "one": runs.add(part, f"{model.name}-one.bin", "--chunk", "1", "--kv", "f32", "--path", "decode"),
            }
            for fmt in KV_GATES:
                outs[fmt] = runs.add(tok, f"{model.name}-{fmt}.bin", "--chunk", "512", "--kv", fmt, "--path", "prefill")
            timing = runs.run()
            label = f"{model.name} {evals[0]['name']}"
            print(f"\n{gguf.name}, {evals[0]['name']}: {len(ids)} tokens  [{timing[outs['gpu']]}]")
            agree += compare(label, "prefill", read_records(outs["gpu"]), ref, len(ids))
            total += len(ids)
            print(f"  decode kernels, steps of 1, first {DECODE_TOKENS} tokens  [{timing[outs['one']]}]")
            decode_agree += compare(f"{label} (decode)", "decode", read_records(outs["one"]), ref, DECODE_TOKENS)
            decode_total += DECODE_TOKENS
            kv_formats(label, ref, len(ids), outs, timing)

    for path, n, a in (("prefill", total, agree), ("decode", decode_total, decode_agree)):
        print(f"\n{args.host}, f32 KV, {path} kernels: top-1 agreement {a}/{n}; min cosine {worst[path][0]:.7f} (min {COSINE_MIN}); max rel L2 {worst[path][1]:.2e} (max {REL_L2_MAX})", end="")
    print()
    for fmt, (top_min, cos_min) in KV_GATES.items():
        s = kv_stats[fmt]
        print(f"{args.host}, {fmt} KV: top-1 agreement {s['agree']}/{s['total']} (min {top_min:.0%}); min cosine {s['cos']:.7f} (min {cos_min})")
        fail.expect(f"{fmt} KV top-1 agreement", s["agree"] >= top_min * s["total"], f"{s['agree']}/{s['total']}")
    proc = zllm_on_host(args, paths, "engine-check", paths.gguf)
    print(f"\n{args.host}, engine-check:")
    sys.stdout.write("".join(line + "\n" for line in proc.stdout.replace("\r", "").splitlines() if line.startswith(("PASS", "FAIL"))))
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    fail.expect("engine-check", proc.returncode == 0, f"exit code {proc.returncode}")
    fail.expect("top-1 agreement", agree == total, f"{agree}/{total}")
    fail.expect("decode top-1 agreement", decode_agree == decode_total, f"{decode_agree}/{decode_total}")
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
