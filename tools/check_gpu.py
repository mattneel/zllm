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

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from check_forward import read_records, row_stats
from common import DENSE_MODELS, Failures, fixture_paths, parse_args

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


def run_gpu(args, paths, model: Path, tokens: Path, out: Path, *opts: str) -> str:
    proc = zllm_on_host(args, paths, "gpu-forward", model, tokens, out, *opts)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(f"gpu-forward failed ({args.host})")
    timing = [line for line in proc.stderr.replace("\r", "").splitlines() if line.startswith("gpu-forward:")]
    return timing[-1] if timing else ""


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

    def kv_formats(label: str, gguf: Path, tok: Path, ref: dict, rows: int, *opts: str) -> None:
        """The 512-token prefill run with each KV_GATES cache format, against the reference."""
        for fmt, (_, cos_min) in KV_GATES.items():
            out = Path(tmp) / f"{fmt}.bin"
            timing = run_gpu(args, paths, gguf, tok, out, "--chunk", "512", "--kv", fmt, "--path", "prefill", *opts)
            got = read_records(out)
            cos = min(float(row_stats(got[n], ref[n])[0].min()) for n in ref)
            top = int((got["result_output"].argmax(axis=1) == ref["result_output"][:rows].argmax(axis=1)).sum())
            s = kv_stats[fmt]
            s["agree"] += top
            s["total"] += rows
            s["cos"] = min(s["cos"], cos)
            print(f"  {fmt} KV: top-1 agreement {top}/{rows}, min cosine {cos:.7f}  [{timing}]")
            fail.expect(f"{label}: {fmt} KV cosine", cos >= cos_min, f"{cos:.7f} < {cos_min}")

    def bitwise(a: dict, b: dict, rows: int) -> bool:
        return all(np.array_equal(a[n].view(np.uint32), b[n][: len(a[n])].view(np.uint32)) for n in b)

    with tempfile.TemporaryDirectory() as tmp:
        for ev in evals:
            ids = hf.encode(ev["text"]).ids
            tok = Path(tmp) / f"{ev['name']}.i32"
            tok.write_bytes(np.asarray(ids, dtype="<i4").tobytes())
            ref_path, gpu_path = Path(tmp) / "ref.bin", Path(tmp) / "gpu.bin"
            subprocess.run([args.zllm, "forward", str(paths.gguf), str(tok), str(ref_path), "--mtp"], check=True)
            ref = read_records(ref_path)
            mtp = ("--mtp",) if "mtp_logits" in ref else ()
            timing = run_gpu(args, paths, paths.gguf, tok, gpu_path, "--chunk", "512", "--kv", "f32", "--path", "prefill", *mtp)
            gpu = read_records(gpu_path)
            fail.expect_eq(f"{ev['name']}: tensor names", sorted(gpu), sorted(ref))
            print(f"\n{ev['name']}: {len(ids)} tokens  [{timing}]")
            agree += compare(ev["name"], "prefill", gpu, ref, len(ids))
            total += len(ids)

            split_path = Path(tmp) / "split.bin"
            timing = run_gpu(args, paths, paths.gguf, tok, split_path, "--chunk", "7", "--kv", "f32", "--path", "prefill", *mtp)
            same = bitwise(read_records(split_path), gpu, len(ids))
            print(f"  prefill kernels, steps of 7: {'bitwise identical' if same else 'DIFFERENT'}  [{timing}]")
            fail.expect(f"{ev['name']}: prefill steps of 7 bitwise", same, "outputs differ from 512-token steps")

            part = Path(tmp) / "part.i32"
            part.write_bytes(np.asarray(ids[:DECODE_TOKENS], dtype="<i4").tobytes())
            one_path, three_path = Path(tmp) / "one.bin", Path(tmp) / "three.bin"
            timing = run_gpu(args, paths, paths.gguf, part, one_path, "--chunk", "1", "--kv", "f32", "--path", "decode", *mtp)
            one = read_records(one_path)
            print(f"  decode kernels, steps of 1, first {DECODE_TOKENS} tokens  [{timing}]")
            decode_agree += compare(f"{ev['name']} (decode)", "decode", one, ref, DECODE_TOKENS)
            decode_total += DECODE_TOKENS
            timing = run_gpu(args, paths, paths.gguf, part, three_path, "--chunk", "3", "--kv", "f32", "--path", "decode", *mtp)
            same = bitwise(read_records(three_path), one, DECODE_TOKENS)
            print(f"  decode kernels, steps of 3: {'bitwise identical' if same else 'DIFFERENT'}  [{timing}]")
            fail.expect(f"{ev['name']}: decode steps of 3 bitwise", same, "outputs differ from 1-token steps")

            kv_formats(ev["name"], paths.gguf, tok, ref, len(ids), *mtp)

        # The dense models (graph/dense.zig) run the same program with
        # attention layers only: the first eval sequence on each kernel path.
        for model in DENSE_MODELS:
            gguf = model.gguf(paths.root)
            if not gguf.is_file():
                raise SystemExit(f"missing {gguf}; run scripts/fixtures.sh dense")
            ids = Tokenizer.from_file(str(model.hf_tokenizer(paths.root))).encode(evals[0]["text"]).ids
            tok = Path(tmp) / f"{model.name}.i32"
            tok.write_bytes(np.asarray(ids, dtype="<i4").tobytes())
            ref_path, gpu_path = Path(tmp) / "ref.bin", Path(tmp) / "gpu.bin"
            subprocess.run([args.zllm, "forward", str(gguf), str(tok), str(ref_path)], check=True)
            ref = read_records(ref_path)
            timing = run_gpu(args, paths, gguf, tok, gpu_path, "--chunk", "512", "--kv", "f32", "--path", "prefill")
            print(f"\n{gguf.name}, {evals[0]['name']}: {len(ids)} tokens  [{timing}]")
            agree += compare(f"{model.name} {evals[0]['name']}", "prefill", read_records(gpu_path), ref, len(ids))
            total += len(ids)
            part = Path(tmp) / "part.i32"
            part.write_bytes(np.asarray(ids[:DECODE_TOKENS], dtype="<i4").tobytes())
            timing = run_gpu(args, paths, gguf, part, gpu_path, "--chunk", "1", "--kv", "f32", "--path", "decode")
            print(f"  decode kernels, steps of 1, first {DECODE_TOKENS} tokens  [{timing}]")
            decode_agree += compare(f"{model.name} {evals[0]['name']} (decode)", "decode", read_records(gpu_path), ref, DECODE_TOKENS)
            decode_total += DECODE_TOKENS
            kv_formats(f"{model.name} {evals[0]['name']}", gguf, tok, ref, len(ids))

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
