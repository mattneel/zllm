"""Compare the zllm CPU f32 reference forward pass with pinned llama.cpp.

Teacher-forced: each eval text in tools/corpus/eval.jsonl is tokenized once
with the pinned Hugging Face tokenizer, and both implementations consume the
same token ids from an empty state in one batch. Activations are compared at
every position: each layer's output (`l_out-<i>`), the final normed hidden
state (`result_norm`) and the logits (`result_output`).

Pass criteria (docs/src/validation.md):
  * top-1 token agreement over all positions >= 99%
  * final hidden state cosine similarity >= 0.999 at every position
  * final hidden state relative norm error | |z| / |r| - 1 | <= 1% at every position

Per-layer statistics are printed and written to <fixtures>/reports/forward.json.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from common import Failures, fixture_paths, parse_args

EVAL = Path(__file__).parent / "corpus" / "eval.jsonl"

TOP1_MIN = 0.99
COSINE_MIN = 0.999
NORM_ERR_MAX = 0.01


def read_records(path: Path) -> dict[str, np.ndarray]:
    """Activation records written by llama-dump and `zllm forward`, as [n_tokens, dim]."""
    data = path.read_bytes()
    out: dict[str, np.ndarray] = {}
    pos = 0
    while pos < len(data):
        (name_len,) = struct.unpack_from("<I", data, pos)
        pos += 4
        name = data[pos : pos + name_len].decode()
        pos += name_len
        (n_dims,) = struct.unpack_from("<I", data, pos)
        pos += 4
        ne = struct.unpack_from(f"<{n_dims}q", data, pos)
        pos += 8 * n_dims
        count = int(np.prod(ne))
        arr = np.frombuffer(data, dtype="<f4", count=count, offset=pos)
        pos += 4 * count
        out[name] = arr.reshape(ne[1], ne[0])
    return out


def row_stats(z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-position cosine, relative L2 error and relative norm error."""
    z64, r64 = z.astype(np.float64), r.astype(np.float64)
    zn, rn = np.linalg.norm(z64, axis=1), np.linalg.norm(r64, axis=1)
    cos = (z64 * r64).sum(axis=1) / (zn * rn)
    rel = np.linalg.norm(z64 - r64, axis=1) / rn
    norm_err = np.abs(zn / rn - 1)
    return cos, rel, norm_err


def main() -> int:
    args = parse_args(__doc__)
    paths = fixture_paths(args.fixtures)
    dump = paths.llama_bin / "llama-dump"
    if not dump.is_file():
        raise SystemExit(f"missing {dump}; run scripts/fixtures.sh llama")
    hf = Tokenizer.from_file(str(paths.hf_tokenizer))
    evals = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").split("\n") if line]

    fail = Failures()
    report: dict = {"model": paths.gguf.name, "sequences": []}
    agree = total = 0
    worst_cos, worst_norm = 1.0, 0.0

    with tempfile.TemporaryDirectory() as tmp:
        for ev in evals:
            ids = hf.encode(ev["text"]).ids
            tok_path = Path(tmp) / f"{ev['name']}.i32"
            tok_path.write_bytes(np.asarray(ids, dtype="<i4").tobytes())
            ref_path, z_path = Path(tmp) / "ref.bin", Path(tmp) / "zllm.bin"

            t0 = time.monotonic()
            subprocess.run([str(dump), str(paths.gguf), str(tok_path), str(ref_path)], check=True, capture_output=True)
            t1 = time.monotonic()
            subprocess.run([args.zllm, "forward", str(paths.gguf), str(tok_path), str(z_path)], check=True)
            t2 = time.monotonic()

            ref, z = read_records(ref_path), read_records(z_path)
            fail.expect_eq(f"{ev['name']}: tensor names", sorted(z), sorted(ref))

            print(f"\n{ev['name']}: {len(ids)} tokens (llama.cpp {t1 - t0:.1f}s, zllm {t2 - t1:.1f}s)")
            print(f"  {'tensor':<14} {'min cos':>10} {'mean cos':>10} {'max rel L2':>11} {'max |norm-1|':>13}")
            layers = []
            names = sorted((n for n in ref if n.startswith("l_out-")), key=lambda n: int(n.split("-")[1]))
            for name in [*names, "result_norm"]:
                if name not in z:
                    continue
                fail.expect_eq(f"{ev['name']}: {name} shape", z[name].shape, ref[name].shape)
                cos, rel, norm_err = row_stats(z[name], ref[name])
                layers.append(
                    {
                        "tensor": name,
                        "min_cos": float(cos.min()),
                        "mean_cos": float(cos.mean()),
                        "max_rel_l2": float(rel.max()),
                        "max_norm_err": float(norm_err.max()),
                    }
                )
                print(f"  {name:<14} {cos.min():>10.6f} {cos.mean():>10.6f} {rel.max():>11.2e} {norm_err.max():>13.2e}")

            cos, _, norm_err = row_stats(z["result_norm"], ref["result_norm"])
            worst_cos = min(worst_cos, float(cos.min()))
            worst_norm = max(worst_norm, float(norm_err.max()))
            for p in np.flatnonzero(cos < COSINE_MIN):
                fail.expect(f"{ev['name']}: result_norm cosine at position {p}", False, f"{cos[p]:.6f} < {COSINE_MIN}")
            for p in np.flatnonzero(norm_err > NORM_ERR_MAX):
                fail.expect(f"{ev['name']}: result_norm norm error at position {p}", False, f"{norm_err[p]:.2e} > {NORM_ERR_MAX}")

            z_top, r_top = z["result_output"].argmax(axis=1), ref["result_output"].argmax(axis=1)
            seq_agree = int((z_top == r_top).sum())
            agree += seq_agree
            total += len(ids)
            lcos, _, _ = row_stats(z["result_output"], ref["result_output"])
            print(f"  top-1 agreement {seq_agree}/{len(ids)}; logits min cos {lcos.min():.6f}")
            report["sequences"].append(
                {"name": ev["name"], "tokens": len(ids), "top1_agree": seq_agree, "logits_min_cos": float(lcos.min()), "layers": layers}
            )

    top1 = agree / total
    report.update({"top1": top1, "positions": total, "min_final_cos": worst_cos, "max_final_norm_err": worst_norm})
    report_path = paths.root / "reports" / "forward.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1))

    print(f"\ntop-1 agreement {agree}/{total} = {top1:.4f} (min {TOP1_MIN})")
    print(f"final hidden state: min cosine {worst_cos:.6f} (min {COSINE_MIN}), max norm error {worst_norm:.2e} (max {NORM_ERR_MAX})")
    print(f"report: {report_path}")
    fail.expect("top-1 agreement", top1 >= TOP1_MIN, f"{top1:.4f} < {TOP1_MIN}")
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
