"""Fixture-free mutation tests for tokenizer benchmark inputs and reported work."""

from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tokenize_bench as bench


class BenchmarkValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.texts = ["a", "é\n\u2028\u0085\0", ""]
        self.ids = [[1], [2, 3], []]
        corpus_bytes = sum(len(text.encode("utf-8")) for text in self.texts)
        self.report = {
            "engine": "zllm", "corpus_records": 3, "corpus_bytes": corpus_bytes,
            "corpus_tokens": 3, "repetitions": 2, "elapsed_ns": bench.SAMPLE_NS,
            "bytes": corpus_bytes * 2, "tokens": 6,
            "mb_per_second": corpus_bytes * 4 / 1e6, "tokens_per_second": 12.0,
        }

    def validate(self, report: object) -> dict:
        return bench.validate_report(report, "zllm", self.texts, 3)

    def test_correct_work_and_exact_ids_pass(self) -> None:
        self.assertEqual(self.validate(self.report), self.report)
        self.assertTrue(bench.validate_ids(self.ids, self.ids)["ids_match_hf"])

    def test_each_missing_field_fails(self) -> None:
        for key in self.report:
            with self.subTest(key=key):
                report = self.report.copy()
                del report[key]
                with self.assertRaises(ValueError):
                    self.validate(report)

    def test_work_count_mutations_fail(self) -> None:
        mutations = {
            "engine": ["huggingface", None],
            "corpus_records": [2, 4, True, 3.0],
            "corpus_bytes": [len("".join(self.texts)), 0, -1, True],
            "corpus_tokens": [2, 4, -1, True, 3.0],
            "repetitions": [0, -1, True, 1.5, 3],
            "elapsed_ns": [0, -1, True, bench.SAMPLE_NS - 1, bench.SAMPLE_NS * 2],
            "bytes": [0, self.report["bytes"] + 1, True],
            "tokens": [0, 5, 7, True],
        }
        for key, values in mutations.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        self.validate({**self.report, key: value})

    def test_nonfinite_negative_wrong_units_and_boolean_rates_fail(self) -> None:
        for key in ("mb_per_second", "tokens_per_second"):
            for value in (float("nan"), float("inf"), -float("inf"), -1, True, "12", self.report[key] * 1.048576):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        self.validate({**self.report, key: value})

    def test_zero_token_corpus_is_not_rejected_when_work_is_real(self) -> None:
        report = {**self.report, "corpus_tokens": 0, "tokens": 0, "tokens_per_second": 0}
        bench.validate_report(report, "zllm", self.texts, 0)

    def test_id_order_content_and_record_boundary_mutations_are_reported(self) -> None:
        for ids in ([[9], [2, 3], []], [[1], [3, 2], []], [[1, 2], [3], []]):
            with self.subTest(ids=ids):
                parity = bench.validate_ids(ids, self.ids)
                self.assertFalse(parity["ids_match_hf"])
                self.assertTrue(parity["mismatched_records"])
        for ids in (None, [], self.ids[:-1], [[True], [2, 3], []], [[-1], [2, 3], []], [[2**32], [2, 3], []], [[1.0], [2, 3], []], ["1", [2, 3], []]):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    bench.validate_ids(ids, self.ids)

    def test_corpus_preserves_exact_independent_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text("\n" + "\n".join(json.dumps({"name": "ignored", "text": text}, ensure_ascii=False) for text in self.texts) + "\n", encoding="utf-8")
            self.assertEqual(bench.read_corpus(path), self.texts)
            for content in ("", '{"text":""}\n', '{"text":1}\n', '{}\n', '[]\n', '{"text":"\\ud800"}\n', '{"text":'):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises((ValueError, UnicodeError)):
                        bench.read_corpus(path)

    def test_hf_times_repeated_individual_encode_calls(self) -> None:
        calls = []

        def encode(text):
            calls.append(text)
            return SimpleNamespace(ids=self.ids[self.texts.index(text)])

        tokenizer = SimpleNamespace(encode=encode)
        with patch.object(bench.time, "perf_counter_ns", side_effect=[100, 100 + bench.SAMPLE_NS // 2, 100 + bench.SAMPLE_NS]):
            report = bench.benchmark_hf(tokenizer, self.texts, self.ids)
        bench.validate_report(report, "huggingface", self.texts, 3)
        self.assertEqual(calls, self.texts * 2)
        self.assertEqual(report["repetitions"], 2)

    def test_llama_build_failure_has_no_throughput(self) -> None:
        args = SimpleNamespace(llama_cpp=Path("llama.cpp"), cache=Path("cache"), cxx="c++")
        for error in (FileNotFoundError("compiler absent"), subprocess.CalledProcessError(1, ["c++"], stderr="compile failed")):
            with self.subTest(error=error), patch.object(bench, "build_llama", side_effect=error):
                report = bench.benchmark_llama(args, self.texts, self.ids)
            self.assertEqual(report["status"], "unavailable")
            self.assertNotIn("mb_per_second", report)
            self.assertNotIn("elapsed_ns", report)
            self.assertTrue(report["reason"])

    def test_llama_uses_inner_clock_and_lossless_length_prefixed_records(self) -> None:
        args = SimpleNamespace(llama_cpp=Path("llama.cpp"), cache=Path("cache"), cxx="c++", model=Path("model.gguf"))
        report = {**copy.deepcopy(self.report), "engine": "llama.cpp", "ids": self.ids}
        captured = []

        def run_json(command):
            captured.append(Path(command[2]).read_bytes())
            return report

        with patch.object(bench, "build_llama", return_value=Path("helper")), patch.object(bench, "run_json", side_effect=run_json), patch.object(bench.time, "perf_counter_ns", side_effect=AssertionError("subprocess startup must not be timed")):
            result = bench.benchmark_llama(args, self.texts, self.ids)
        expected = b"".join(len(raw).to_bytes(4, "little") + raw for raw in (text.encode("utf-8") for text in self.texts))
        self.assertEqual(captured, [expected])
        self.assertEqual(result["elapsed_ns"], bench.SAMPLE_NS)
        self.assertTrue(result["ids_match_hf"])
        self.assertNotIn("ids", result)


if __name__ == "__main__":
    unittest.main()
