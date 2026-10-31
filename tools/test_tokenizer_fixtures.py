"""Fixture-free mutations for tokenizer-only coverage and required small files."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import check_chat
import check_tokenizer
from common import DENSE_MODELS, TOKENIZER_MODELS, TOKENIZER_ONLY_MODELS, FixturePaths


class TokenizerFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = FixturePaths(Path(self.directory.name))
        self.corpus = self.paths.root / "corpus.jsonl"
        self.corpus.write_text(json.dumps({"name": "single user", "text": "a\u2028\u0085\0", "messages": [{"role": "user", "content": "a"}]}) + "\n", encoding="utf-8")
        self.args = SimpleNamespace(zllm="unused", fixtures=self.paths.root)
        for path in [self.paths.gguf_header, self.paths.hf_tokenizer, self.paths.chat_template, *(p for model in TOKENIZER_MODELS for p in (model.gguf_header(self.paths.root), model.hf_tokenizer(self.paths.root)))]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def tokenizer_check(self):
        calls = Mock()
        with patch.object(check_tokenizer, "parse_args", return_value=self.args), patch.object(check_tokenizer, "CORPUS", self.corpus), patch.object(check_tokenizer, "Tokenizer", Mock()), patch.object(check_tokenizer, "check", calls), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_tokenizer.main(), 0)
        return calls.call_args_list

    def test_tokenizer_only_models_never_join_dense_weight_checks(self) -> None:
        self.assertEqual({model.name for model in TOKENIZER_ONLY_MODELS}, {"nanbeige", "gemma4"})
        self.assertEqual({model.name for model in DENSE_MODELS}, {"qwen3", "llama", "granite"})
        self.assertEqual(TOKENIZER_MODELS, DENSE_MODELS + TOKENIZER_ONLY_MODELS)
        for model in TOKENIZER_ONLY_MODELS:
            self.assertTrue(model.gguf_header(self.paths.root).name.endswith(".header.gguf"))
            self.assertEqual(model.hf_tokenizer(self.paths.root), self.paths.root / "hf" / model.name / "tokenizer.json")

    def test_all_headers_checked_without_weights_or_text_boundary_changes(self) -> None:
        calls = self.tokenizer_check()
        self.assertEqual([call.args[2] for call in calls], [self.paths.gguf_header, *(model.gguf_header(self.paths.root) for model in TOKENIZER_MODELS)])
        self.assertTrue(all(call.args[3] == ["a\u2028\u0085\0"] for call in calls))
        self.assertFalse(any(model.gguf(self.paths.root).exists() for model in TOKENIZER_ONLY_MODELS))

    def test_present_full_tokenizer_model_is_checked_too(self) -> None:
        model = TOKENIZER_ONLY_MODELS[0]
        model.gguf(self.paths.root).touch()
        calls = self.tokenizer_check()
        self.assertEqual(sum(call.args[2] == model.gguf(self.paths.root) for call in calls), 1)

    def test_each_missing_new_header_or_hf_file_fails_instead_of_skipping(self) -> None:
        for model in TOKENIZER_ONLY_MODELS:
            for path in (model.gguf_header(self.paths.root), model.hf_tokenizer(self.paths.root)):
                with self.subTest(path=path):
                    path.unlink()
                    with self.assertRaisesRegex(SystemExit, "fixtures.sh vocab"):
                        self.tokenizer_check()
                    path.touch()

    def chat_check(self):
        calls = Mock()
        env = Mock()
        env.filters = {}
        env.globals = {}
        env.from_string.return_value.render.return_value = "prompt"
        proc = SimpleNamespace(stdout='{"text":"prompt"}\n')
        with patch.object(check_chat, "parse_args", return_value=self.args), patch.object(check_chat, "CORPUS", self.corpus), patch.object(check_chat, "ImmutableSandboxedEnvironment", return_value=env), patch.object(check_chat.subprocess, "run", return_value=proc), patch.object(check_chat, "check_model_template", calls), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_chat.main(), 0)
        return calls.call_args_list

    def test_chat_checks_every_model_template_without_silent_exclusions(self) -> None:
        calls = self.chat_check()
        self.assertEqual([call.args[1] for call in calls], [model.gguf_header(self.paths.root) for model in TOKENIZER_MODELS])

    def test_missing_new_chat_header_fails(self) -> None:
        for model in TOKENIZER_ONLY_MODELS:
            path = model.gguf_header(self.paths.root)
            with self.subTest(model=model.name):
                path.unlink()
                with self.assertRaisesRegex(SystemExit, "fixtures.sh vocab"):
                    self.chat_check()
                path.touch()


if __name__ == "__main__":
    unittest.main()
