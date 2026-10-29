"""Check a quantized model: the GPU against the CPU reference, and the CPU
reference against pinned llama.cpp.

For each eval text in tools/corpus/eval.jsonl, teacher-forced from an empty
state, on the same GGUF:

  R   `zllm forward`: the CPU f32 reference. Quantized weights are decoded
      bitwise as ggml decodes them (check-dequant); activations stay f32.
  RL  `zllm forward --llama-activations`: the same, with each multiply's
      input rounded as llama.cpp's CPU backend rounds it (Q8_0 or Q8_K
      blocks).
  L   `llama-dump`: pinned llama.cpp on the CPU.
  G   `zllm gpu-forward`: the prefill kernels in PREFILL_STEP-token steps (a
      27B step must stay well under Windows' 2 s GPU watchdog), and the
      decode kernels in 1-token steps over the first DECODE_TOKENS positions.
      The quantized prefill multiplies round their input as llama.cpp does
      (8-bit dot products), so that path is another implementation of RL;
      the decode multiplies keep their input f32, so that one implements R.

Pass criteria (docs/src/validation.md):
  * GPU = reference, decode kernels: G against R, top-1 agreement 100%,
    and at every position:
      - the first layer's output: relative L2 <= GPU_FIRST_LAYER_MAX (the
        kernels agree up to f32 summation order);
      - every layer output, the final hidden state and the logits: cosine
        >= GPU_COSINE_MIN and relative L2 <= GPU_REL_L2_MAX. The model
        amplifies small differences: on the 27B one position's 1e-7 grows
        smoothly to 4e-4 over 64 layers. The bound still sits 30x below the
        gap llama.cpp's activation rounding leaves.
  * Reference = llama.cpp, up to llama.cpp's activation rounding:
      - the first layer of RL against L: median relative L2 <= FIRST_LAYER_MAX
        (only the emulation's own rounding ties differ);
      - at every layer, the median gap between R and L is at most
        SENSITIVITY_RATIO times the median gap between R and RL, the model's
        own response to that rounding. A semantic difference (a wrong head
        mapping, norm or rotation) adds a gap the rounding cannot explain,
        from the first layer of the affected kind on.
  * GPU = reference up to rounding, prefill kernels: G against RL and R by
    the two bounds above, as llama.cpp. Rounding to 8 bits is discontinuous:
    the GPU's f32 activations differ from the reference's by summation
    order, about 1e-7, and a value that close to a rounding boundary rounds
    the other way and moves by a whole step. So positions differ by a few
    such ties, and the bounds take medians over positions. Each kernel is
    checked exactly by gpu-check.
    The spec's bounds against llama.cpp (cosine >= 0.999, top-1 >= 99%)
    are reported, not enforced: on the 27B, R and RL alone already miss
    them.

Per-layer statistics are written to <fixtures>/reports/<model>.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from check_forward import read_records, row_stats
from check_gpu import run_gpu
from common import GGUF27_FILE, Failures, fixture_paths, parse_args

EVAL = Path(__file__).parent / "corpus" / "eval.jsonl"

GPU_COSINE_MIN = 0.99999
GPU_FIRST_LAYER_MAX = 1e-5
GPU_REL_L2_MAX = 1e-3
FIRST_LAYER_MAX = 1e-4
SENSITIVITY_RATIO = 1.5
DECODE_TOKENS = 40
PREFILL_STEP = 32


def layer_names(records: dict) -> list[str]:
    return sorted((n for n in records if n.startswith("l_out-")), key=lambda n: int(n.split("-")[1]))


def main() -> int:
    def extra(p):
        p.add_argument("--host", choices=["native", "d3d12"], default="d3d12")
        p.add_argument("--model", type=Path, help=f"the quantized GGUF (default: <fixtures>/models/{GGUF27_FILE})")

    args = parse_args(__doc__, extra)
    paths = fixture_paths(args.fixtures)
    dump = paths.llama_bin / "llama-dump"
    if not dump.is_file():
        raise SystemExit(f"missing {dump}; run scripts/fixtures.sh llama")
    model = (args.model or paths.root / "models" / GGUF27_FILE).resolve()
    if not model.is_file():
        raise SystemExit(f"{model} not found; run scripts/fixtures.sh model27b")
    hf = Tokenizer.from_file(str(paths.hf_tokenizer))
    evals = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").split("\n") if line]

    fail = Failures()
    report: dict = {"model": model.name, "sequences": []}
    gpu_agree = {"decode": 0}
    gpu_total = {"decode": 0}
    gpu_worst = {"decode": [1.0, 0.0]}
    gpu_first = {"decode": 0.0}
    gpu_rounded = {"ratio": 0.0, "first": 0.0, "agree": 0, "total": 0}
    llama_agree = llama_total = 0
    llama_min_cos = 1.0
    worst_ratio = 0.0
    worst_first = 0.0

    def rounded_vs_ref(label: str, other: dict, r: dict, rl: dict) -> tuple[float, float, int]:
        """An implementation of the rounded activations, `other`, against RL:
        the first layer's median gap, and at every layer the median gap to R
        within SENSITIVITY_RATIO of RL's own. Returns the worst ratio, the
        first-layer gap and top-1 agreement with RL."""
        worst = 0.0
        for name in [*layer_names(r), "result_norm"]:
            gap = float(np.median(row_stats(r[name], other[name])[1]))
            own = float(np.median(row_stats(r[name], rl[name])[1]))
            ratio = gap / own if own > 0 else float("inf")
            worst = max(worst, ratio)
            fail.expect(f"{label}: {name} gap vs rounding sensitivity", ratio <= SENSITIVITY_RATIO, f"{gap:.2e} / {own:.2e} = {ratio:.2f} > {SENSITIVITY_RATIO}")
        first = float(np.median(row_stats(rl["l_out-0"], other["l_out-0"])[1]))
        fail.expect(f"{label}: first layer against the rounded reference", first <= FIRST_LAYER_MAX, f"median rel L2 {first:.2e} > {FIRST_LAYER_MAX}")
        top = int((other["result_output"].argmax(axis=1) == rl["result_output"].argmax(axis=1)).sum())
        return worst, first, top

    def gpu_vs_ref(label: str, path: str, g: dict, r: dict, rows: int) -> None:
        _, first, _ = row_stats(g["l_out-0"][:rows], r["l_out-0"][:rows])
        gpu_first[path] = max(gpu_first[path], float(first.max()))
        for p in np.flatnonzero(first > GPU_FIRST_LAYER_MAX):
            fail.expect(f"{label}: l_out-0 at position {p}", False, f"rel L2 {first[p]:.2e} > {GPU_FIRST_LAYER_MAX}")
        for name in [*layer_names(r), "result_norm", "result_output"]:
            cos, rel, _ = row_stats(g[name][:rows], r[name][:rows])
            gpu_worst[path][0] = min(gpu_worst[path][0], float(cos.min()))
            gpu_worst[path][1] = max(gpu_worst[path][1], float(rel.max()))
            for p in np.flatnonzero((cos < GPU_COSINE_MIN) | (rel > GPU_REL_L2_MAX)):
                fail.expect(f"{label}: {name} at position {p}", False, f"cosine {cos[p]:.7f}, rel L2 {rel[p]:.2e}")
        top = int((g["result_output"][:rows].argmax(axis=1) == r["result_output"][:rows].argmax(axis=1)).sum())
        gpu_agree[path] += top
        gpu_total[path] += rows
        fail.expect(f"{label}: top-1 agreement", top == rows, f"{top}/{rows}")
        cos, rel, _ = row_stats(g["result_output"][:rows], r["result_output"][:rows])
        print(f"  GPU vs reference, {path} kernels: top-1 {top}/{rows}; logits min cosine {cos.min():.7f}, max rel L2 {rel.max():.2e}")

    with tempfile.TemporaryDirectory() as tmp:
        for ev in evals:
            ids = hf.encode(ev["text"]).ids
            n = len(ids)
            tok = Path(tmp) / f"{ev['name']}.i32"
            tok.write_bytes(np.asarray(ids, dtype="<i4").tobytes())
            out = {k: Path(tmp) / f"{k}.bin" for k in ("L", "R", "RL", "G", "G1")}
            subprocess.run([str(dump), str(model), str(tok), str(out["L"])], check=True, capture_output=True)
            subprocess.run([args.zllm, "forward", str(model), str(tok), str(out["R"])], check=True)
            subprocess.run([args.zllm, "forward", str(model), str(tok), str(out["RL"]), "--llama-activations"], check=True)
            print(f"\n{ev['name']}: {n} tokens")
            print(f"  [{run_gpu(args, paths, model, tok, out['G'], '--chunk', str(PREFILL_STEP))}]")
            part = Path(tmp) / f"{ev['name']}.part.i32"
            part.write_bytes(np.asarray(ids[:DECODE_TOKENS], dtype="<i4").tobytes())
            print(f"  [{run_gpu(args, paths, model, part, out['G1'], '--chunk', '1')}]")
            rec = {k: read_records(p) for k, p in out.items()}
            r, rl, ll = rec["R"], rec["RL"], rec["L"]

            gpu_vs_ref(f"{ev['name']} (decode)", "decode", rec["G1"], r, DECODE_TOKENS)
            g_ratio, g_first, g_top = rounded_vs_ref(f"{ev['name']}: GPU prefill", rec["G"], r, rl)
            gpu_rounded["ratio"] = max(gpu_rounded["ratio"], g_ratio)
            gpu_rounded["first"] = max(gpu_rounded["first"], g_first)
            gpu_rounded["agree"] += g_top
            gpu_rounded["total"] += n
            print(f"  GPU prefill vs reference with llama.cpp's rounding: top-1 {g_top}/{n}; first layer median rel L2 {g_first:.2e}; worst gap / rounding sensitivity {g_ratio:.2f}")

            layers = []
            print(f"  {'tensor':<14} {'R vs L':>10} {'R vs RL':>10} {'ratio':>6}   (median rel L2)")
            for i, name in enumerate([*layer_names(r), "result_norm"]):
                gap = float(np.median(row_stats(r[name], ll[name])[1]))
                own = float(np.median(row_stats(r[name], rl[name])[1]))
                ratio = gap / own if own > 0 else float("inf")
                worst_ratio = max(worst_ratio, ratio)
                layers.append({"tensor": name, "ref_vs_llama": gap, "ref_vs_rounded": own})
                if i < 2 or i % 16 == 15 or name == "result_norm":
                    print(f"  {name:<14} {gap:>10.2e} {own:>10.2e} {ratio:>6.2f}")
                fail.expect(f"{ev['name']}: {name} gap to llama.cpp vs rounding sensitivity", ratio <= SENSITIVITY_RATIO, f"{gap:.2e} / {own:.2e} = {ratio:.2f} > {SENSITIVITY_RATIO}")
            first = float(np.median(row_stats(rl["l_out-0"], ll["l_out-0"])[1]))
            worst_first = max(worst_first, first)
            fail.expect(f"{ev['name']}: first layer with llama.cpp's rounding", first <= FIRST_LAYER_MAX, f"median rel L2 {first:.2e} > {FIRST_LAYER_MAX}")
            cos, _, _ = row_stats(r["result_norm"], ll["result_norm"])
            llama_min_cos = min(llama_min_cos, float(cos.min()))
            top = int((r["result_output"].argmax(axis=1) == ll["result_output"].argmax(axis=1)).sum())
            own_top = int((r["result_output"].argmax(axis=1) == rl["result_output"].argmax(axis=1)).sum())
            llama_agree += top
            llama_total += n
            print(f"  first layer with llama.cpp's rounding: median rel L2 {first:.2e}")
            print(f"  reference vs llama.cpp: top-1 {top}/{n} (vs its own rounded run: {own_top}/{n}); final hidden min cosine {cos.min():.6f}")
            report["sequences"].append({"name": ev["name"], "tokens": n, "first_layer_emulated": first, "layers": layers, "top1_vs_llama": top, "top1_vs_rounded": own_top})

    print()
    c, rel = gpu_worst["decode"]
    print(f"{args.host}, GPU vs reference, decode kernels: top-1 {gpu_agree['decode']}/{gpu_total['decode']}; first layer max rel L2 {gpu_first['decode']:.2e} (max {GPU_FIRST_LAYER_MAX}); min cosine {c:.7f} (min {GPU_COSINE_MIN}); max rel L2 {rel:.2e} (max {GPU_REL_L2_MAX})")
    print(
        f"{args.host}, GPU vs reference with llama.cpp's rounding, prefill kernels: top-1 {gpu_rounded['agree']}/{gpu_rounded['total']} (reported); "
        f"first layer worst median rel L2 {gpu_rounded['first']:.2e} (max {FIRST_LAYER_MAX}); worst gap / rounding sensitivity {gpu_rounded['ratio']:.2f} (max {SENSITIVITY_RATIO})"
    )
    print(f"reference vs llama.cpp: first layer with its rounding, worst median rel L2 {worst_first:.2e} (max {FIRST_LAYER_MAX}); worst gap / rounding sensitivity {worst_ratio:.2f} (max {SENSITIVITY_RATIO})")
    print(f"reference vs llama.cpp, spec §12 (reported): top-1 {llama_agree}/{llama_total} = {llama_agree / llama_total:.4f}; final hidden min cosine {llama_min_cos:.6f}")
    report.update({"gpu": gpu_worst, "gpu_rounded": gpu_rounded, "worst_ratio": worst_ratio, "worst_first_layer": worst_first, "top1_vs_llama": llama_agree / llama_total, "min_cos_vs_llama": llama_min_cos})
    report_path = paths.root / "reports" / f"{model.stem}.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))
    print(f"report: {report_path}")
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
