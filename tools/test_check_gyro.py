"""Fixture-free regression mutations of check_gyro's actual comparison gates."""

from __future__ import annotations

import contextlib
import io
import json
import unittest

import numpy as np

from check_gyro import CORPUS, FORWARD_RUNS, LOGITS_MEDIAN_MAX, LOGITS_MAX, STREAMS_MAX, compare_runs
from common import Failures


class GyroChecks(unittest.TestCase):
    def setUp(self) -> None:
        ref1 = json.loads((CORPUS / "gyro_ref1.json").read_text())
        self.tokens = np.asarray(ref1["prompt"] + ref1["greedy"], dtype=np.int32) % 7 + 1
        self.n_prompt = len(ref1["prompt"])
        self.n = len(self.tokens)
        vocab = int(self.tokens[self.n_prompt :].max()) + 1
        # Keep the fixture's real row counts with small synthetic token ids.
        # Each teacher-forced position predicts the next synthetic greedy id.
        logits = np.zeros((self.n, vocab), dtype=np.float32)
        logits[:, 0] = 1
        for pos, token in enumerate(self.tokens[self.n_prompt :], self.n_prompt - 1):
            logits[pos, token] = 2
        self.cpu = {
            "l_last-0": np.tile(np.array([1, 2], dtype=np.float32), self.n),
            "ffn_moe_topk-0": np.tile(np.array([0, 1], dtype=np.float32), self.n),
            "result_output": logits.ravel(),
        }
        self.runs = {name: {k: v.copy() for k, v in self.cpu.items()} for name, *_ in FORWARD_RUNS}

    def failures(self) -> list[str]:
        fail = Failures()
        with contextlib.redirect_stdout(io.StringIO()):
            compare_runs(fail, self.runs, self.cpu, self.tokens, self.n_prompt)
        return fail.items

    def test_run_matrix_keeps_both_paths_partial_tiles_and_small_prefill(self) -> None:
        self.assertEqual(len({name for name, *_ in FORWARD_RUNS}), len(FORWARD_RUNS))
        for path in ("prefill", "decode"):
            cases = [case for case in FORWARD_RUNS if case[1] == path and case[3] == "f16"]
            self.assertTrue(any(case[4] for case in cases), path)
            chunks = {case[2] for case in cases if not case[4]}
            self.assertIn(1, chunks)
            self.assertTrue(any(chunk > 4 for chunk in chunks) if path == "prefill" else all(chunk <= 4 for chunk in chunks))
        prefill = next(case for case in FORWARD_RUNS if case[0] == "prefill")
        self.assertGreaterEqual(prefill[2], 64)
        self.assertGreater(self.n, 64)
        self.assertNotEqual(self.n % 64, 0)
        self.assertIn(("prefill-7", "prefill", 7, "f16", False), FORWARD_RUNS)
        self.assertEqual((LOGITS_MEDIAN_MAX, LOGITS_MAX, STREAMS_MAX), (1e-4, 1e-3, 1e-3))

    def test_distinct_path_rounding_is_not_compared_bitwise(self) -> None:
        for name, path, _, kv, _ in FORWARD_RUNS:
            if path == "prefill" and kv == "f16":
                for key in ("l_last-0", "result_output"):
                    self.runs[name][key] *= np.float32(1.00001)
        self.assertFalse(self.failures())

    def test_each_step_split_and_fusion_rejects_one_bit_in_partial_tail(self) -> None:
        for name in ("prefill-7", "prefill-1", "decode-1", "prefill-unfused", "decode-unfused"):
            with self.subTest(run=name):
                bits = self.runs[name]["l_last-0"].view(np.uint32)
                bits[-1] ^= np.uint32(1)
                self.assertTrue(any(item.startswith(f"{name}:") for item in self.failures()))
                bits[-1] ^= np.uint32(1)

    def test_both_paths_reject_single_row_cpu_error_above_existing_bound(self) -> None:
        for path in ("prefill", "decode"):
            for key, gate in (("l_last-0", "streams layer 0"), ("result_output", "logits max")):
                with self.subTest(path=path, record=key):
                    record = self.runs[path][key]
                    row = record.reshape(self.n, -1)[-1]
                    original = row.copy()
                    row *= np.float32(1.002)
                    self.assertTrue(any(item.startswith(f"{path}: {gate}") for item in self.failures()))
                    row[:] = original

    def test_nonfinite_stream_and_logit_rows_fail(self) -> None:
        for path in ("prefill", "decode"):
            for key, gate in (("l_last-0", "streams layer 0"), ("result_output", "logits max")):
                with self.subTest(path=path, record=key):
                    record = self.runs[path][key]
                    original = record[-1]
                    record[-1] = np.nan
                    self.assertTrue(any(item.startswith(f"{path}: {gate}") for item in self.failures()))
                    record[-1] = original

    def test_both_paths_reject_expert_routing_flip(self) -> None:
        for path in ("prefill", "decode"):
            with self.subTest(path=path):
                self.runs[path]["ffn_moe_topk-0"][-1] = 2
                self.assertTrue(any(item.startswith(f"{path}: expert routing") for item in self.failures()))
                self.runs[path]["ffn_moe_topk-0"][-1] = 1

    def test_both_paths_keep_cpu_top_one_and_runtime_greedy_checks(self) -> None:
        for path in ("prefill", "decode"):
            with self.subTest(path=path):
                record = self.runs[path]["result_output"].reshape(self.n, -1)
                row = record[self.n_prompt - 1]
                original = row.copy()
                row[0] = 3
                failures = self.failures()
                self.assertTrue(any(item.startswith(f"{path}: top-1") for item in failures))
                self.assertTrue(any(item.startswith(f"{path}: greedy tokens") for item in failures))
                row[:] = original

    def test_missing_and_extra_split_records_fail(self) -> None:
        record = self.runs["prefill-7"].pop("l_last-0")
        self.assertTrue(any(item.startswith("prefill-7: step splits") for item in self.failures()))
        self.runs["prefill-7"]["l_last-0"] = record
        self.runs["prefill-7"]["unexpected"] = np.ones(1, dtype=np.float32)
        self.assertTrue(any(item.startswith("prefill-7: step splits") for item in self.failures()))

    def test_cpu_only_intermediates_are_not_gpu_readback_requirements(self) -> None:
        self.cpu["hc_norm-0"] = np.ones(self.n * 8, dtype=np.float32)
        self.assertFalse(self.failures())
        self.runs["prefill"].pop("ffn_moe_topk-0")
        self.assertTrue(any(item.startswith("prefill: tensor names") for item in self.failures()))


if __name__ == "__main__":
    unittest.main()
