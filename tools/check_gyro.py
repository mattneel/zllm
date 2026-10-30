"""The Qwen3.8-Flash-Next Gyro-Coder (Qwen4Exp) on a GPU host against zllm's
CPU reference and its reference runtime's greedy tokens.

`--model` is the 75.8 GB `Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf`, read in
place. The prompts and greedy tokens (tools/corpus/gyro_*.json) come from the
reference runtime, AgentionAI's llama.cpp fork.

1. The ref1 prompt and the runtime's 40 greedy tokens, teacher-forced: `zllm
   forward` (CPU) and `zllm gpu-forward` on the host, both writing every
   layer's streams (l_last) and chosen experts (ffn_moe_topk) and the logits.
   Every layer must choose the same experts; streams and logits must stay
   within the bounds below; every position's top-1 must agree, so the GPU's
   greedy choices are the runtime's 40 tokens. One-token and four-token steps
   must give bitwise identical results.
2. The runtime's 6,430-token prompt (a fact buried in filler, past the sparse
   attention's 2,051-cell width), greedy through `zllm generate` on the host:
   the runtime's 10 tokens.

Hosts as in check_gpu.py: `native` (this build, ZLLM_DAWN's adapter) or
`d3d12` (the Windows build through scripts/d3d12.sh).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from check_forward import read_records
from check_gpu import run_gpu, zllm_on_host
from common import Failures, fixture_paths, parse_args

CORPUS = Path(__file__).parent / "corpus"

# Measured on the RTX 5090 (D3D12): logits 1.9e-5 median, 1.5e-4 max;
# streams 1.3e-4 max. The bounds leave room for another backend's summation
# order, not for a wrong kernel (a routing flip moves a row by 1e-2).
LOGITS_MEDIAN_MAX = 1e-4
LOGITS_MAX = 1e-3
STREAMS_MAX = 1e-3


def rel_l2(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    return np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)


def main() -> int:
    args = parse_args(
        __doc__,
        lambda p: (
            p.add_argument("--host", choices=["native", "d3d12"], default="native"),
            p.add_argument("--model", type=Path, required=True, help="Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf"),
        ),
    )
    paths = fixture_paths(args.fixtures)
    if str(args.model) in ("", "."):
        raise SystemExit("check-gyro needs the model: -Dgyro-model=<path to Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf>")
    if not args.model.is_file():
        raise SystemExit(f"model not found at {args.model}")
    fail = Failures()
    ref1 = json.loads((CORPUS / "gyro_ref1.json").read_text())
    tokens = np.array(ref1["prompt"] + ref1["greedy"], dtype=np.int32)
    n_prompt = len(ref1["prompt"])

    with tempfile.TemporaryDirectory(prefix="zllm-gyro-") as tmp:
        work = Path(tmp)
        tok = work / "ref1.i32"
        tokens.tofile(tok)
        cpu_path = work / "cpu.bin"
        subprocess.run([args.zllm, "forward", str(args.model), str(tok), str(cpu_path)], check=True)
        cpu = read_records(cpu_path)
        runs = {}
        for chunk in ("4", "1"):
            out = work / f"gpu{chunk}.bin"
            timing = run_gpu(args, paths, args.model, tok, out, "--chunk", chunk, "--kv", "f16")
            print(f"  [{timing}]")
            runs[chunk] = read_records(out)
        gpu = runs["4"]

        n = len(tokens)
        layers = sorted({int(k.rsplit("-", 1)[1]) for k in gpu if k.startswith("l_last-")})
        fail.expect("layers", len(layers) > 0 and all(f"l_last-{i}" in cpu for i in layers), "missing layer records")
        streams_max = 0.0
        flips = 0
        for i in layers:
            g = gpu[f"l_last-{i}"].reshape(n, -1)
            c = cpu[f"l_last-{i}"].reshape(n, -1)
            streams_max = max(streams_max, float(rel_l2(g, c).max()))
            ge = gpu[f"ffn_moe_topk-{i}"].reshape(n, -1).astype(int)
            ce = cpu[f"ffn_moe_topk-{i}"].reshape(n, -1).astype(int)
            flips += sum(set(ge[t]) != set(ce[t]) for t in range(n))
        g = gpu["result_output"].reshape(n, -1)
        c = cpu["result_output"].reshape(n, -1)
        logits = rel_l2(g, c)
        top1 = int((g.argmax(1) == c.argmax(1)).sum())
        greedy = int((g[n_prompt - 1 : -1].argmax(1) == tokens[n_prompt:]).sum())
        same = sorted(runs["1"]) == sorted(gpu) and all(np.array_equal(runs["1"][k].view(np.uint32), gpu[k].view(np.uint32)) for k in gpu)
        print(f"{args.host}, ref1 ({n} positions, {len(layers)} layers):")
        print(f"  expert sets differing from the CPU reference: {flips} (want 0)")
        print(f"  streams max rel L2 {streams_max:.2e} (max {STREAMS_MAX})")
        print(f"  logits rel L2 median {np.median(logits):.2e} (max {LOGITS_MEDIAN_MAX}), max {logits.max():.2e} (max {LOGITS_MAX})")
        print(f"  top-1 agreement {top1}/{n}; greedy {greedy}/{len(ref1['greedy'])} of the runtime's tokens")
        print(f"  1-token vs 4-token steps: {'bitwise identical' if same else 'DIFFER'}")
        fail.expect("expert routing", flips == 0, f"{flips} (position, layer) expert sets differ")
        fail.expect("streams", streams_max <= STREAMS_MAX, f"{streams_max:.2e}")
        fail.expect("logits median", float(np.median(logits)) <= LOGITS_MEDIAN_MAX, f"{np.median(logits):.2e}")
        fail.expect("logits max", float(logits.max()) <= LOGITS_MAX, f"{logits.max():.2e}")
        fail.expect_eq("top-1", top1, n)
        fail.expect_eq("greedy tokens", greedy, len(ref1["greedy"]))
        fail.expect("step splits", same, "1-token and 4-token steps differ")

    # The long prompt, greedy, end to end.
    long = json.loads((CORPUS / "gyro_reflong.json").read_text())
    hf = Tokenizer.from_file(str(paths.hf_tokenizer))
    prompt = hf.decode(long["prompt"], skip_special_tokens=False)
    fail.expect("reflong prompt", hf.encode(prompt, add_special_tokens=False).ids == long["prompt"], "the prompt text does not encode back to its ids")
    # The last greedy token ends the turn: a stop token, not part of the text.
    want = hf.decode(long["greedy"][:-1], skip_special_tokens=False)
    context = str(1 << (len(long["prompt"]) + len(long["greedy"])).bit_length())
    proc = zllm_on_host(args, paths, "generate", args.model, prompt, "--max-tokens", str(len(long["greedy"])), "--temperature", "0", "--ctx", context)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        raise SystemExit(f"generate failed ({args.host})")
    # generate echoes the prompt, then the generated text and a newline: the
    # text after the echoed prompt's end.
    out = proc.stdout.replace("\r", "")
    tail = prompt[-100:]
    at = out.rfind(tail)
    got = out[at + len(tail) :] if at >= 0 else out
    stats = [line for line in proc.stderr.replace("\r", "").splitlines() if line.startswith("[stop") or line.startswith("[length")]
    print(f"{args.host}, reflong ({len(long['prompt'])} prompt tokens): {got.strip()!r} (want {want!r}) {stats[-1] if stats else ''}")
    fail.expect_eq("reflong greedy", got.strip(), want)
    fail.expect("reflong prompt tokens", any(f"prompt {len(long['prompt'])} tokens" in s for s in stats), "the prompt encoded to another length")
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
