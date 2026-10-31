"""Fixture-free mutations of the precision harness's real policy and I/O path."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import check_precision
from check_gpu import GpuRuns
from check_precision import GATE, MAX_REL_L2, MEAN_KL_MAX, TOP1_MIN, compare_metrics, compare_reference, compute_metrics, validate_report
from common import Failures


class PrecisionChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.tile(np.array([10, 9], dtype=np.float32), (64, 1))
        self.selection = {"kv": "f16", "prefill": "fp16_compensated", "weights": "stored"}
        self.metrics = compute_metrics(self.reference, self.reference)
        self.report = {
            "selection": self.selection.copy(),
            "gate": GATE.copy(),
            "reference_kv": "f16",
            "observations": [{"selection": self.selection.copy(), "metrics": self.metrics.copy()}],
            "metrics": self.metrics.copy(),
        }

    def failures(self, chosen: np.ndarray, reference: np.ndarray | None = None) -> list[str]:
        fail = Failures()
        compare_reference(fail, "model", chosen, self.reference if reference is None else reference, GATE)
        return fail.items

    def report_failures(self, report: dict) -> list[str]:
        fail = Failures()
        validate_report(fail, "model", report, 64)
        return fail.items

    def test_identical_logits_and_measured_selection_pass(self) -> None:
        self.assertFalse(self.failures(self.reference.copy()))
        self.assertFalse(self.report_failures(self.report))
        self.assertEqual((TOP1_MIN, MEAN_KL_MAX, MAX_REL_L2), (0.99, 0.001, 0.1))
        self.assertEqual(self.metrics, {"positions": 64, "top1_matches": 64, "mean_kl": 0.0, "max_rel_l2": 0.0})

    def test_single_tail_top_one_flip_fails_even_with_tiny_divergence(self) -> None:
        reference = np.tile(np.array([1.000001, 1], dtype=np.float32), (64, 1))
        chosen = reference.copy()
        chosen[-1] = chosen[-1, ::-1]
        metrics = compute_metrics(chosen, reference)
        self.assertLess(metrics["mean_kl"], MEAN_KL_MAX)
        self.assertLess(metrics["max_rel_l2"], MAX_REL_L2)
        failures = self.failures(chosen, reference)
        self.assertTrue(any("CPU: top-1" in item for item in failures), failures)

    def test_distribution_drift_fails_kl_without_top_one_or_l2_failure(self) -> None:
        chosen = np.tile(np.array([10.9, 8.1], dtype=np.float32), (64, 1))
        metrics = compute_metrics(chosen, self.reference)
        self.assertEqual(metrics["top1_matches"], 64)
        self.assertLess(metrics["max_rel_l2"], MAX_REL_L2)
        failures = self.failures(chosen)
        self.assertTrue(any("CPU: mean KL" in item for item in failures), failures)
        self.assertFalse(any("CPU: top-1" in item or "CPU: max rel L2" in item for item in failures), failures)

    def test_single_tail_common_shift_fails_max_l2_not_kl_or_top_one(self) -> None:
        chosen = self.reference.copy()
        chosen[-1] += 2
        failures = self.failures(chosen)
        self.assertTrue(any("CPU: max rel L2" in item for item in failures), failures)
        self.assertFalse(any("CPU: top-1" in item or "CPU: mean KL" in item for item in failures), failures)

    def test_nonfinite_cpu_or_chosen_logits_fail(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            for side in ("chosen", "reference"):
                with self.subTest(value=value, side=side):
                    chosen, reference = self.reference.copy(), self.reference.copy()
                    (chosen if side == "chosen" else reference)[-1, -1] = value
                    self.assertTrue(any("nonfinite logits" in item for item in self.failures(chosen, reference)))

    def test_shapes_empty_rows_and_zero_reference_fail_closed(self) -> None:
        for chosen in (self.reference[:-1], self.reference[:, :-1], np.empty((0, 2)), self.reference.ravel()):
            with self.subTest(shape=chosen.shape):
                self.assertTrue(self.failures(chosen))
        zeros = np.zeros((2, 3), dtype=np.float32)
        self.assertFalse(self.failures(zeros, zeros))
        nonzero = zeros.copy()
        nonzero[-1, -1] = 1
        self.assertTrue(any("nonfinite relative L2" in item for item in self.failures(nonzero, zeros)))

    def test_kl_is_reference_to_chosen_and_logsumexp_is_stable(self) -> None:
        reference = np.array([[0, 1]], dtype=np.float64)
        chosen = np.array([[0, 3]], dtype=np.float64)
        metrics = compute_metrics(chosen, reference)
        p = np.exp(reference[0]) / np.exp(reference[0]).sum()
        q = np.exp(chosen[0]) / np.exp(chosen[0]).sum()
        self.assertAlmostEqual(metrics["mean_kl"], float((p * np.log(p / q)).sum()), places=14)
        shifted = compute_metrics(chosen + 10000, reference + 10000)
        self.assertAlmostEqual(shifted["mean_kl"], metrics["mean_kl"], places=14)
        extreme = np.array([[10000, -10000], [-10000, 10000]], dtype=np.float32)
        self.assertEqual(compute_metrics(extreme, extreme)["mean_kl"], 0)
        ties = compute_metrics(np.array([[1, 1]], dtype=np.float32), np.array([[1, 1]], dtype=np.float32))
        self.assertEqual(ties["top1_matches"], 1)

    def test_metric_metadata_cannot_hide_nan_or_invalid_counts(self) -> None:
        mutations = (
            ("positions", 0), ("positions", True), ("top1_matches", 65),
            ("top1_matches", -1), ("top1_matches", 1.5),
            ("mean_kl", np.nan), ("mean_kl", -0.1),
            ("max_rel_l2", np.inf), ("max_rel_l2", -1),
        )
        for name, value in mutations:
            with self.subTest(name=name, value=value):
                metrics = {**self.metrics, name: value}
                fail = Failures()
                compare_metrics(fail, "model", metrics, GATE)
                self.assertTrue(fail.items)
        fail = Failures()
        compare_metrics(fail, "model", {}, GATE)
        self.assertTrue(fail.items)

    def test_calibration_report_rejects_weakened_gate_unmeasured_selection_and_nan(self) -> None:
        for mutate in (
            lambda r: r["gate"].update(min_top1_agreement=0.95),
            lambda r: r["gate"].update(max_mean_kl=0.1),
            lambda r: r["gate"].update(max_rel_l2=1),
            lambda r: r["selection"].update(kv="fp8"),
            lambda r: r["selection"].update(prefill="unknown"),
            lambda r: r["selection"].update(kv="f32"),
            lambda r: r.update(reference_kv="f32"),
            lambda r: r["metrics"].update(mean_kl=np.nan),
            lambda r: r["metrics"].update(positions=63),
            lambda r: r["metrics"].update(top1_matches=63),
            lambda r: r.update(observations=[]),
            lambda r: r.pop("metrics"),
        ):
            report = copy.deepcopy(self.report)
            mutate(report)
            self.assertTrue(self.report_failures(report), report)

    def test_main_runs_default_ladder_and_both_cpu_oracles_serially(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = root / "models" / "Qwen3.5-0.8B-BF16.gguf"
            model.parent.mkdir()
            model.touch()
            gyro = root / "gyro.gguf"
            gyro.touch()
            args = SimpleNamespace(zllm="zllm", fixtures=root, host="d3d12", gyro_model=gyro)
            calls = []

            def precision(args, paths, command, gguf, tokens, logits):
                self.assertEqual(command, "precision-check")
                calls.append(("gpu", gguf))
                tokens.write_bytes(np.arange(64, dtype="<i4").tobytes())
                logits.write_bytes(self.reference.astype("<f4").tobytes())
                return subprocess.CompletedProcess([], 0, json.dumps(self.report), "load logs stay on stderr")

            def cpu(paths, kind, gguf, tokens, out, cmd):
                calls.append(("cpu", gguf))
                options = ["--kv", "f16"]
                self.assertEqual(cmd, ["zllm", "forward", str(gguf), str(tokens), str(out), *options])
                self.assertEqual(kind, "zllm-forward precision --kv f16")
                self.assertEqual(np.fromfile(tokens, dtype="<i4").tolist(), list(range(64)))
                name = b"result_output"
                out.write_bytes(struct.pack("<I", len(name)) + name + struct.pack("<Iqq", 2, 2, 64) + self.reference.astype("<f4").tobytes())

            with patch.object(check_precision, "parse_args", return_value=args), patch.object(check_precision, "zllm_on_host", side_effect=precision), patch.object(check_precision, "cached_reference", side_effect=cpu), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(check_precision.main(), 0)
            self.assertEqual(calls, [("gpu", model), ("cpu", model), ("gpu", gyro), ("cpu", gyro)])
            report = json.loads((root / "reports" / "precision.json").read_text())
            self.assertEqual([r["reference_kv"] for r in report], ["f16", "f16"])
            self.assertTrue(all(r["cpu_metrics"]["top1_matches"] == 64 for r in report))

    def test_shared_gpu_batch_requests_exact_and_retains_run_kv(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args = SimpleNamespace(host="native", zllm="zllm")
            paths = SimpleNamespace(root=root)
            process = SimpleNamespace(stderr=io.StringIO("gpu-forward: one run\n"), stdout=io.StringIO(""), wait=lambda: 0)
            with GpuRuns(args, paths, root / "model.gguf", root) as runs:
                runs.add(root / "tokens.i32", "out.bin", "--kv", "f16", "--path", "prefill")
                with patch("check_gpu.subprocess.Popen", return_value=process) as popen, contextlib.redirect_stdout(io.StringIO()):
                    runs.run()
            command = popen.call_args.args[0]
            self.assertEqual(command[1:6], ["gpu-forward", str(root / "model.gguf"), "--precision", "exact", "--runs"])
            self.assertIn("--kv f16 --path prefill", (root / "runs-model.txt").read_text())


if __name__ == "__main__":
    unittest.main()
