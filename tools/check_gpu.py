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
  * f16 KV cache (the default): top-1 agreement >= F16_TOP1_MIN, and cosine
    >= F16_COSINE_MIN at every position.
  * `zllm engine-check`: requests that reuse the resident sequence or a
    checkpoint give the logits of fresh runs (bitwise where both runs use the
    same kernels for each position).
  * Each dense model (scripts/fixtures.sh dense), on the first eval text: the
    512-token run's bounds and top-1 agreement on both kernel paths (the
    decode kernels over its first DECODE_TOKENS positions).

Hosts: `native` runs the Linux build on the Vulkan adapter Dawn finds
(llvmpipe in CI); `d3d12` runs the Windows build through scripts/d3d12.sh.
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
F16_TOP1_MIN = 0.99
F16_COSINE_MIN = 0.99998
# (tokens per step, token limit) for the runs that must match bitwise.
# Positions checked through the decode kernels (one step per token).
DECODE_TOKENS = 40


def win(path: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(path)], check=True, capture_output=True, text=True).stdout.strip()


def zllm_on_host(args, paths, command: str, *argv: Path | str) -> subprocess.CompletedProcess:
    """Runs a zllm command on the selected host; Path arguments are mapped to the host."""
    if args.host == "d3d12":
        mapped = [win(a) if isinstance(a, Path) else a for a in argv]
        cmd = [str(ROOT / "scripts" / "d3d12.sh"), command, *mapped]
        env = os.environ.copy()
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
    args = parse_args(__doc__, lambda p: p.add_argument("--host", choices=["native", "d3d12"], default="native"))
    paths = fixture_paths(args.fixtures)
    hf = Tokenizer.from_file(str(paths.hf_tokenizer))
    evals = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").split("\n") if line]

    fail = Failures()
    total = agree = agree16 = total16 = decode_total = decode_agree = 0
    worst = {"prefill": [1.0, 0.0], "decode": [1.0, 0.0]}
    worst_cos16 = 1.0

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

            f16_path = Path(tmp) / "f16.bin"
            timing = run_gpu(args, paths, paths.gguf, tok, f16_path, "--chunk", "512", "--kv", "f16", "--path", "prefill", *mtp)
            f16 = read_records(f16_path)
            cos16 = min(float(row_stats(f16[n], ref[n])[0].min()) for n in ref)
            top16 = int((f16["result_output"].argmax(axis=1) == ref["result_output"].argmax(axis=1)).sum())
            agree16 += top16
            total16 += len(ids)
            worst_cos16 = min(worst_cos16, cos16)
            print(f"  f16 KV: top-1 agreement {top16}/{len(ids)}, min cosine {cos16:.7f}  [{timing}]")
            fail.expect(f"{ev['name']}: f16 KV cosine", cos16 >= F16_COSINE_MIN, f"{cos16:.7f} < {F16_COSINE_MIN}")

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

    for path, n, a in (("prefill", total, agree), ("decode", decode_total, decode_agree)):
        print(f"\n{args.host}, f32 KV, {path} kernels: top-1 agreement {a}/{n}; min cosine {worst[path][0]:.7f} (min {COSINE_MIN}); max rel L2 {worst[path][1]:.2e} (max {REL_L2_MAX})", end="")
    print()
    print(f"{args.host}, f16 KV: top-1 agreement {agree16}/{total16} (min {F16_TOP1_MIN:.0%}); min cosine {worst_cos16:.7f} (min {F16_COSINE_MIN})")
    proc = zllm_on_host(args, paths, "engine-check", paths.gguf)
    print(f"\n{args.host}, engine-check:")
    sys.stdout.write("".join(line + "\n" for line in proc.stdout.replace("\r", "").splitlines() if line.startswith(("PASS", "FAIL"))))
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    fail.expect("engine-check", proc.returncode == 0, f"exit code {proc.returncode}")
    fail.expect("top-1 agreement", agree == total, f"{agree}/{total}")
    fail.expect("decode top-1 agreement", decode_agree == decode_total, f"{decode_agree}/{decode_total}")
    fail.expect("f16 KV top-1 agreement", agree16 >= F16_TOP1_MIN * total16, f"{agree16}/{total16}")
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
