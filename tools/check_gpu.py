"""Compare the GPU forward pass with the CPU f32 reference.

For each eval text in tools/corpus/eval.jsonl, the same token ids run through
`zllm forward` (CPU reference, itself checked against llama.cpp by
check_forward.py) and `zllm gpu-forward` on a Dawn host, from an empty state
in one batch. Both use the same bf16 weights and f32 activations, so they may
differ only by reduction order and transcendental-function rounding.

Pass criteria (docs/src/validation.md):
  * top-1 token agreement with the reference = 100% of positions
  * every layer output, the final hidden state and the logits: cosine >= 0.999999
    and relative L2 error <= 1e-4 at every position

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


def win(path: Path) -> str:
    return subprocess.run(["wslpath", "-w", str(path)], check=True, capture_output=True, text=True).stdout.strip()


def run_gpu(args, paths, model: Path, tokens: Path, out: Path) -> str:
    if args.host == "d3d12":
        cmd = [str(ROOT / "scripts" / "d3d12.sh"), "gpu-forward", win(model), win(tokens), win(out)]
        env = os.environ.copy()
    else:
        cmd = [args.zllm, "gpu-forward", str(model), str(tokens), str(out)]
        env = {**os.environ, "ZLLM_DAWN": str(paths.root / "dawn" / "libwebgpu_dawn.so")}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
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
    total = agree = 0
    worst_cos, worst_rel = 1.0, 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for ev in evals:
            ids = hf.encode(ev["text"]).ids
            tok = Path(tmp) / f"{ev['name']}.i32"
            tok.write_bytes(np.asarray(ids, dtype="<i4").tobytes())
            ref_path, gpu_path = Path(tmp) / "ref.bin", Path(tmp) / "gpu.bin"
            subprocess.run([args.zllm, "forward", str(paths.gguf), str(tok), str(ref_path)], check=True)
            timing = run_gpu(args, paths, paths.gguf, tok, gpu_path)

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
            top = int((gpu["result_output"].argmax(axis=1) == ref["result_output"].argmax(axis=1)).sum())
            agree += top
            total += len(ids)
            print(f"  top-1 agreement {top}/{len(ids)}")

    print(f"\n{args.host}: top-1 agreement {agree}/{total}; min cosine {worst_cos:.7f} (min {COSINE_MIN}); max rel L2 {worst_rel:.2e} (max {REL_L2_MAX})")
    fail.expect("top-1 agreement", agree == total, f"{agree}/{total}")
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
