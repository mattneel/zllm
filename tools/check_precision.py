"""Check the default precision ladder against the independent CPU reference.

`precision-check` tokenizes the model's fixed calibration text (at most 64
positions), writes little-endian i32 tokens and row-major f32 chosen-rung
logits, and reports its selection, gate and calibration observations as JSON.
The same tokens run through `forward`; its result_output records must pass
>=99% top-1 agreement and the model's KL / maximum relative-L2 policy.

Runs the Qwen3.5-0.8B fixture and, with --gyro-model, the supplied Gyro-Coder.
GPU hosts and Windows path mapping are shared with check_gpu.py. All model
and CPU/GPU jobs run sequentially; this harness never launches parallel work.

Both CPU oracles explicitly use f16 attention KV, the automatic ladder's
reference and ceiling. Explicit f32 KV remains supported but is never an
automatic candidate. No quality threshold changes with the reference format.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

from check_forward import read_records
from check_gpu import HOSTS, zllm_on_host
from common import Failures, cached_reference, fixture_paths, parse_args

TOP1_MIN = 0.99
MEAN_KL_MAX = 0.001
MAX_REL_L2 = 0.1
GATE = {"min_top1_agreement": TOP1_MIN, "max_mean_kl": MEAN_KL_MAX, "max_rel_l2": MAX_REL_L2}
METRIC_FIELDS = {"positions", "top1_matches", "mean_kl", "max_rel_l2"}


def compute_metrics(chosen: np.ndarray, reference: np.ndarray) -> dict:
    """Stable f64 KL(P_reference || P_chosen) and worst per-row relative L2."""
    if chosen.shape != reference.shape or reference.ndim != 2 or not all(reference.shape):
        raise ValueError(f"logit shapes must match nonempty [positions, vocab]: {chosen.shape}, {reference.shape}")
    if not np.isfinite(chosen).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite logits")
    got, ref = chosen.astype(np.float64), reference.astype(np.float64)

    def log_softmax(x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    log_ref, log_got = log_softmax(ref), log_softmax(got)
    kl = (np.exp(log_ref) * (log_ref - log_got)).sum(axis=1)
    ref_norm = np.linalg.norm(ref, axis=1)
    delta_norm = np.linalg.norm(got - ref, axis=1)
    rel = np.divide(delta_norm, ref_norm, out=np.full_like(ref_norm, np.inf), where=ref_norm != 0)
    rel[(ref_norm == 0) & (delta_norm == 0)] = 0
    if not np.isfinite(rel).all():
        raise ValueError("nonfinite relative L2 (nonzero chosen logits against a zero reference row)")
    return {
        "positions": len(ref),
        "top1_matches": int((got.argmax(axis=1) == ref.argmax(axis=1)).sum()),
        "mean_kl": max(0.0, float(kl.mean())),
        "max_rel_l2": float(rel.max()),
    }


def compare_metrics(fail: Failures, label: str, metrics: dict, gate: dict) -> None:
    """Fail closed for malformed/nonfinite metrics, then apply each policy gate."""
    if not isinstance(metrics, dict) or set(metrics) != METRIC_FIELDS:
        fail.expect(f"{label}: metrics", False, "missing or unexpected metric fields")
        return
    n, matches = metrics["positions"], metrics["top1_matches"]
    valid = type(n) is int and n > 0 and type(matches) is int and 0 <= matches <= n
    for name in ("mean_kl", "max_rel_l2"):
        value = metrics[name]
        valid = valid and type(value) in (int, float) and math.isfinite(value) and value >= 0
    fail.expect(f"{label}: finite metrics", valid, repr(metrics))
    if not valid:
        return
    fail.expect(f"{label}: top-1", matches / n >= gate["min_top1_agreement"], f"{matches}/{n} < {gate['min_top1_agreement']}")
    fail.expect(f"{label}: mean KL", metrics["mean_kl"] <= gate["max_mean_kl"], f"{metrics['mean_kl']:.6g} > {gate['max_mean_kl']}")
    fail.expect(f"{label}: max rel L2", metrics["max_rel_l2"] <= gate["max_rel_l2"], f"{metrics['max_rel_l2']:.6g} > {gate['max_rel_l2']}")


def valid_selection(selection) -> bool:
    return (
        isinstance(selection, dict)
        and set(selection) == {"kv", "prefill", "weights"}
        and selection["kv"] in ("fp8", "f16")
        and selection["prefill"] in ("fp16_single", "fp16_compensated", "scalar_f32")
        and selection["weights"] == "stored"
    )


def validate_report(fail: Failures, label: str, report: dict, positions: int) -> bool:
    """Require the selected rung to be the measured, quality-approved rung."""
    valid = isinstance(report, dict) and {"selection", "gate", "observations", "metrics"} <= set(report)
    fail.expect(f"{label}: report fields", valid, "missing precision-check report fields")
    if not valid:
        return False
    try:
        json.dumps(report, allow_nan=False)
    except ValueError:
        fail.expect(f"{label}: finite report", False, "nonfinite precision-check report")
        return False
    valid = valid_selection(report["selection"]) and report["gate"] == GATE and report.get("reference_kv") == "f16"
    fail.expect(f"{label}: selection and policy", valid, repr((report["selection"], report["gate"])))
    if not valid:
        return False
    compare_metrics(fail, f"{label} calibration", report["metrics"], report["gate"])
    final = report["metrics"]
    fail.expect(f"{label}: calibration positions", isinstance(final, dict) and final.get("positions") == positions, f"want {positions} prompt positions")
    observations = report["observations"]
    valid = isinstance(observations, list) and bool(observations)
    if valid:
        valid = all(isinstance(o, dict) and set(o) == {"selection", "metrics"} and valid_selection(o["selection"]) and isinstance(o["metrics"], dict) for o in observations)
    fail.expect(f"{label}: observations", valid, "missing or malformed calibration observations")
    if not valid:
        return False
    selected = [o for o in observations if o["selection"] == report["selection"]]
    fail.expect(f"{label}: measured selection", any(o["metrics"] == final for o in selected), "selected rung/metrics absent from observations")
    return True


def compare_reference(fail: Failures, label: str, chosen: np.ndarray, reference: np.ndarray, gate: dict) -> dict | None:
    try:
        metrics = compute_metrics(chosen, reference)
    except ValueError as error:
        fail.expect(f"{label}: CPU logits", False, str(error))
        return None
    compare_metrics(fail, f"{label} CPU", metrics, gate)
    return metrics


def main() -> int:
    args = parse_args(
        __doc__,
        lambda p: (
            p.add_argument("--host", choices=HOSTS, default="native"),
            p.add_argument("--gyro-model", type=Path, help="optional Qwen3.8-Flash-Next-Gyro-Coder-TQ2_0.gguf, read in place"),
        ),
    )
    paths = fixture_paths(args.fixtures)
    models = [(paths.gguf, ("--kv", "f16"))]
    if args.gyro_model is not None:
        models.append((args.gyro_model.resolve(), ("--kv", "f16")))
    for model, _ in models:
        if not model.is_file():
            raise SystemExit(f"model not found at {model}")
    fail = Failures()
    reports = []
    with tempfile.TemporaryDirectory(prefix="zllm-precision-") as tmp:
        work = Path(tmp)
        for index, (model, cpu_options) in enumerate(models):
            label = model.name
            tokens, logits = work / f"{index}.i32", work / f"{index}.f32"
            print(f"[{label}: default precision ladder on {args.host}]", flush=True)
            proc = zllm_on_host(args, paths, "precision-check", model, tokens, logits)
            if proc.returncode != 0:
                sys.stderr.write(proc.stdout + proc.stderr)
                fail.expect(f"{label}: precision-check", False, f"exit code {proc.returncode}")
                continue
            try:
                report = json.loads(proc.stdout)
                ids = np.fromfile(tokens, dtype="<i4")
                chosen = np.fromfile(logits, dtype="<f4")
            except (ValueError, OSError) as error:
                fail.expect(f"{label}: precision-check output", False, str(error))
                continue
            n = len(ids)
            if tokens.stat().st_size % 4 or not 1 <= n <= 64 or np.any(ids < 0) or logits.stat().st_size % 4 or not chosen.size or chosen.size % n:
                fail.expect(f"{label}: calibration output shape", False, "expected 1..64 nonnegative i32 IDs and complete f32 logit rows")
                continue
            if not validate_report(fail, label, report, n):
                continue
            cpu_path = work / f"{index}-cpu.bin"
            print(f"[{label}: CPU reference for {n} calibration positions]", flush=True)
            cached_reference(paths, "zllm-forward precision --kv f16", model, tokens, cpu_path, [args.zllm, "forward", str(model), str(tokens), str(cpu_path), *cpu_options])
            cpu = read_records(cpu_path)
            if "result_output" not in cpu:
                fail.expect(f"{label}: CPU logits", False, "missing result_output")
                continue
            chosen = chosen.reshape(n, -1)
            metrics = compare_reference(fail, label, chosen, cpu["result_output"], report["gate"])
            reports.append({"model": str(model), "host": args.host, **report, "reference_kv": "f16", "cpu_metrics": metrics})
            if metrics is not None:
                print(f"  {report['selection']}: CPU top-1 {metrics['top1_matches']}/{n}, mean KL {metrics['mean_kl']:.6g}, max rel L2 {metrics['max_rel_l2']:.6g}")
    report_path = paths.root / "reports" / "precision.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=1, allow_nan=False) + "\n")
    print(f"report: {report_path}")
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
