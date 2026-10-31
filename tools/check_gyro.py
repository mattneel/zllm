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
   greedy choices are the runtime's 40 tokens. Forced prefill and decode
   paths each use `--precision exact --kv f16` and match the CPU's f16 KV
   with the existing bounds below.
   Step splits and fusion must be bitwise invariant separately within
   each path (the two paths may round differently). Prefill covers full
   and partial tiles, 7-token steps, and forced 1-token steps.
2. The runtime's 6,430-token prompt (a fact buried in filler, past the sparse
   attention's 2,051-cell width), greedy through `zllm generate` on the host:
   the runtime's 10 tokens.

Hosts as in check_gpu.py: `native` (this build, ZLLM_DAWN's adapter) or
`d3d12` (the Windows build through scripts/d3d12.sh) or `agility` (that build on
Agility).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from check_forward import read_records
from check_gpu import HOSTS, GpuRuns, zllm_on_host
from common import Failures, cached_reference, fixture_paths, parse_args

CORPUS = Path(__file__).parent / "corpus"

# Measured on the RTX 5090 (D3D12): logits 1.9e-5 median, 1.5e-4 max;
# streams 1.3e-4 max. The bounds leave room for another backend's summation
# order, not for a wrong kernel (a routing flip moves a row by 1e-2).
LOGITS_MEDIAN_MAX = 1e-4
LOGITS_MAX = 1e-3
STREAMS_MAX = 1e-3

# name, forced path, chunk, KV format, unfused. All runs share the complete
# ref1 sequence, including all 40 greedy tokens; the model loads only once.
FORWARD_RUNS = (
    ("prefill", "prefill", 512, "f16", False),
    ("prefill-7", "prefill", 7, "f16", False),
    ("prefill-1", "prefill", 1, "f16", False),
    ("prefill-unfused", "prefill", 512, "f16", True),
    ("decode", "decode", 4, "f16", False),
    ("decode-1", "decode", 1, "f16", False),
    ("decode-unfused", "decode", 4, "f16", True),
)


def rel_l2(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    return np.linalg.norm(got - want, axis=1) / np.linalg.norm(want, axis=1)


def compare_reference(fail: Failures, label: str, gpu: dict, cpu: dict, tokens: np.ndarray, n_prompt: int) -> None:
    """Check every layer and position, retaining routing and greedy gates."""
    # The CPU observer also records intermediate activations that the GPU
    # deliberately does not read back. Require the complete shared boundary.
    layers = sorted(int(k.rsplit("-", 1)[1]) for k in cpu if k.startswith("l_last-"))
    expected = {"result_output"} | {f"{kind}-{i}" for i in layers for kind in ("l_last", "ffn_moe_topk")}
    names_match = set(gpu) == expected and expected <= set(cpu)
    fail.expect(f"{label}: tensor names", names_match, f"expected {len(expected)} shared records, got {len(gpu)}")
    if not names_match:
        return
    shapes_match = all(gpu[k].shape == cpu[k].shape for k in expected)
    fail.expect(f"{label}: tensor shapes", shapes_match, "record shapes differ from the CPU")
    if not shapes_match:
        return
    n = len(tokens)
    fail.expect(f"{label}: layers", bool(layers), "missing layer records")
    streams_max = 0.0
    flips = 0
    for i in layers:
        g = gpu[f"l_last-{i}"].reshape(n, -1)
        c = cpu[f"l_last-{i}"].reshape(n, -1)
        rel = rel_l2(g, c)
        streams_max = max(streams_max, float(rel.max()))
        fail.expect(f"{label}: streams layer {i}", bool(np.all(rel <= STREAMS_MAX)), f"max rel L2 {rel.max():.2e} > {STREAMS_MAX}")
        ge = gpu[f"ffn_moe_topk-{i}"].reshape(n, -1)
        ce = cpu[f"ffn_moe_topk-{i}"].reshape(n, -1)
        flips += sum(set(ge[t]) != set(ce[t]) for t in range(n))
    g = gpu["result_output"].reshape(n, -1)
    c = cpu["result_output"].reshape(n, -1)
    logits = rel_l2(g, c)
    top1 = int((g.argmax(1) == c.argmax(1)).sum())
    greedy = int((g[n_prompt - 1 : -1].argmax(1) == tokens[n_prompt:]).sum())
    fail.expect(f"{label}: expert routing", flips == 0, f"{flips} (position, layer) expert sets differ")
    fail.expect(f"{label}: logits median", float(np.median(logits)) <= LOGITS_MEDIAN_MAX, f"{np.median(logits):.2e}")
    fail.expect(f"{label}: logits max", bool(np.all(logits <= LOGITS_MAX)), f"{logits.max():.2e} > {LOGITS_MAX}")
    fail.expect_eq(f"{label}: top-1", top1, n)
    fail.expect_eq(f"{label}: greedy tokens", greedy, n - n_prompt)
    print(f"  {label}: expert flips {flips}; streams max rel L2 {streams_max:.2e}; logits median {np.median(logits):.2e}, max {logits.max():.2e}")
    print(f"    top-1 {top1}/{n}; greedy {greedy}/{n - n_prompt} of the runtime's tokens")


def compare_runs(fail: Failures, runs: dict, cpu: dict, tokens: np.ndarray, n_prompt: int) -> None:
    """Only compare bit patterns within the same forced kernel path."""
    for path, splits in (("prefill", ("prefill-7", "prefill-1")), ("decode", ("decode-1",))):
        base = runs[path]
        compare_reference(fail, path, base, cpu, tokens, n_prompt)
        for name in (*splits, f"{path}-unfused"):
            other = runs[name]
            names_match = sorted(base) == sorted(other)
            differ = [k for k in base if k not in other or base[k].shape != other[k].shape or not np.array_equal(base[k].view(np.uint32), other[k].view(np.uint32))]
            same = names_match and not differ
            kind = "fusion" if name.endswith("-unfused") else "step splits"
            fail.expect(f"{name}: {kind}", same, f"records differ from {path}: {differ or 'tensor names'}")
            print(f"  {name} vs {path}: {'bitwise identical' if same else 'DIFFER'}")


def main() -> int:
    args = parse_args(
        __doc__,
        lambda p: (
            p.add_argument("--host", choices=HOSTS, default="native"),
            p.add_argument("--model", type=Path, required=True, help="Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf"),
            p.add_argument("--fast", action="store_true", help="the fast tier: ref1 only, without the long prompt"),
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
        cached_reference(paths, "zllm-forward", args.model, tok, cpu_path, [args.zllm, "forward", str(args.model), str(tok), str(cpu_path)])
        cpu = read_records(cpu_path)
        with GpuRuns(args, paths, args.model, work) as batch:
            outs = {
                name: batch.add(tok, f"gpu-{name}.bin", "--path", path, "--chunk", str(chunk), "--kv", kv, *(("--no-fuse",) if unfused else ()))
                for name, path, chunk, kv, unfused in FORWARD_RUNS
            }
            timing = batch.run()
            runs = {}
            for name, out in outs.items():
                print(f"  {name} [{timing[out]}]")
                runs[name] = read_records(out)
        print(f"{args.host}, ref1 ({len(tokens)} positions):")
        compare_runs(fail, runs, cpu, tokens, n_prompt)

    if args.fast:
        return fail.report()
    # The long prompt, greedy, end to end.
    long = json.loads((CORPUS / "gyro_reflong.json").read_text())
    hf = Tokenizer.from_file(str(paths.hf_tokenizer))
    prompt = hf.decode(long["prompt"], skip_special_tokens=False)
    fail.expect("reflong prompt", hf.encode(prompt, add_special_tokens=False).ids == long["prompt"], "the prompt text does not encode back to its ids")
    # The last greedy token ends the turn: a stop token, not part of the text.
    want = hf.decode(long["greedy"][:-1], skip_special_tokens=False)
    context = str(1 << (len(long["prompt"]) + len(long["greedy"])).bit_length())
    proc = zllm_on_host(args, paths, "generate", args.model, prompt, "--precision", "exact", "--kv", "f16", "--max-tokens", str(len(long["greedy"])), "--temperature", "0", "--ctx", context)
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
