"""Compare the GPU forward pass with the CPU f32 reference.

For each eval text in tools/corpus/eval.jsonl, the same token ids run through
`zllm forward` (CPU reference, itself checked against llama.cpp by
check_forward.py) and `zllm gpu-forward` on a Dawn host, which processes them
as steps against the KV cache and the recurrent state.

Pass criteria (docs/src/validation.md):
  * f32 KV cache, 512-token steps: both use the same bf16 weights and f32
    activations, so they may differ only by reduction order and
    transcendental-function rounding. Top-1 agreement 100%; every layer
    output, the final hidden state and the logits: cosine >= 0.999999 and
    relative L2 error <= 1e-4 at every position.
  * 7-token steps, and 1-token steps (as decoding runs) over the first 40
    tokens: bitwise identical to the 512-token run. Every row's arithmetic is
    independent of how the sequence is split into steps.
  * f16 KV cache (the default): top-1 agreement >= F16_TOP1_MIN, and cosine
    >= F16_COSINE_MIN at every position.
  * `zllm engine-check`: requests that reuse the resident sequence or a
    checkpoint give logits bitwise identical to fresh runs.

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
from common import Failures, fixture_paths, parse_args

EVAL = Path(__file__).parent / "corpus" / "eval.jsonl"
ROOT = Path(__file__).resolve().parent.parent

COSINE_MIN = 0.999999
REL_L2_MAX = 1e-4
F16_TOP1_MIN = 0.99
F16_COSINE_MIN = 0.99998
# (tokens per step, token limit) for the runs that must match bitwise.
SPLITS = [(7, None), (1, 40)]


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
    total = agree = agree16 = 0
    worst_cos, worst_rel, worst_cos16 = 1.0, 0.0, 1.0
    with tempfile.TemporaryDirectory() as tmp:
        for ev in evals:
            ids = hf.encode(ev["text"]).ids
            tok = Path(tmp) / f"{ev['name']}.i32"
            tok.write_bytes(np.asarray(ids, dtype="<i4").tobytes())
            ref_path, gpu_path = Path(tmp) / "ref.bin", Path(tmp) / "gpu.bin"
            subprocess.run([args.zllm, "forward", str(paths.gguf), str(tok), str(ref_path)], check=True)
            timing = run_gpu(args, paths, paths.gguf, tok, gpu_path, "--chunk", "512", "--kv", "f32")

            ref, gpu = read_records(ref_path), read_records(gpu_path)
            fail.expect_eq(f"{ev['name']}: tensor names", sorted(gpu), sorted(ref))
            print(f"\n{ev['name']}: {len(ids)} tokens  [{timing}]")
            print(f"  {'tensor':<14} {'min cos':>10} {'max rel L2':>11}")
            names = sorted((n for n in ref if n.startswith("l_out-")), key=lambda n: int(n.split("-")[1]))
            for name in [*names, "result_norm", "result_output"]:
                if name not in gpu:
                    continue
                cos, rel, _ = row_stats(gpu[name], ref[name])
                print(f"  {name:<14} {cos.min():>10.7f} {rel.max():>11.2e}")
                worst_cos = min(worst_cos, float(cos.min()))
                worst_rel = max(worst_rel, float(rel.max()))
                for p in np.flatnonzero(cos < COSINE_MIN):
                    fail.expect(f"{ev['name']}: {name} cosine at {p}", False, f"{cos[p]:.7f} < {COSINE_MIN}")
                for p in np.flatnonzero(rel > REL_L2_MAX):
                    fail.expect(f"{ev['name']}: {name} rel L2 at {p}", False, f"{rel[p]:.2e} > {REL_L2_MAX}")
            ref_top = ref["result_output"].argmax(axis=1)
            top = int((gpu["result_output"].argmax(axis=1) == ref_top).sum())
            agree += top
            total += len(ids)
            print(f"  top-1 agreement {top}/{len(ids)}")

            for chunk, limit in SPLITS:
                part = tok
                if limit is not None:
                    part = Path(tmp) / "part.i32"
                    part.write_bytes(np.asarray(ids[:limit], dtype="<i4").tobytes())
                split_path = Path(tmp) / "split.bin"
                timing = run_gpu(args, paths, paths.gguf, part, split_path, "--chunk", str(chunk), "--kv", "f32")
                split = read_records(split_path)
                rows = len(ids) if limit is None else limit
                same = all(np.array_equal(split[n].view(np.uint32), gpu[n][:rows].view(np.uint32)) for n in gpu)
                print(f"  steps of {chunk}, {rows} tokens: {'bitwise identical' if same else 'DIFFERENT'}  [{timing}]")
                fail.expect(f"{ev['name']}: steps of {chunk} bitwise", same, "outputs differ from 512-token steps")

            f16_path = Path(tmp) / "f16.bin"
            timing = run_gpu(args, paths, paths.gguf, tok, f16_path, "--chunk", "512", "--kv", "f16")
            f16 = read_records(f16_path)
            cos16 = min(float(row_stats(f16[n], ref[n])[0].min()) for n in ref)
            top16 = int((f16["result_output"].argmax(axis=1) == ref_top).sum())
            agree16 += top16
            worst_cos16 = min(worst_cos16, cos16)
            print(f"  f16 KV: top-1 agreement {top16}/{len(ids)}, min cosine {cos16:.7f}  [{timing}]")
            fail.expect(f"{ev['name']}: f16 KV cosine", cos16 >= F16_COSINE_MIN, f"{cos16:.7f} < {F16_COSINE_MIN}")

    print(f"\n{args.host}, f32 KV: top-1 agreement {agree}/{total}; min cosine {worst_cos:.7f} (min {COSINE_MIN}); max rel L2 {worst_rel:.2e} (max {REL_L2_MAX})")
    print(f"{args.host}, f16 KV: top-1 agreement {agree16}/{total} (min {F16_TOP1_MIN:.0%}); min cosine {worst_cos16:.7f} (min {F16_COSINE_MIN})")
    proc = zllm_on_host(args, paths, "engine-check", paths.gguf)
    print(f"\n{args.host}, engine-check:")
    sys.stdout.write("".join(line + "\n" for line in proc.stdout.replace("\r", "").splitlines() if line.startswith(("PASS", "FAIL"))))
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    fail.expect("engine-check", proc.returncode == 0, f"exit code {proc.returncode}")
    fail.expect("top-1 agreement", agree == total, f"{agree}/{total}")
    fail.expect("f16 KV top-1 agreement", agree16 >= F16_TOP1_MIN * total, f"{agree16}/{total}")
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
