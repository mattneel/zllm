"""Compare zllm's chat prompts with the pinned fixed Qwen chat template.

For every conversation in tools/corpus/chat.jsonl, `zllm chat-render` (the
transcription in src/chat/chat.zig) must produce exactly the text the Jinja
template renders, with Jinja set up as Hugging Face transformers sets it up
for apply_chat_template: a sandboxed environment with trim_blocks and
lstrip_blocks, the loop-controls extension, and raise_exception.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from common import DENSE_MODELS, Failures, fixture_paths, gguf_metadata, parse_args

CORPUS = Path(__file__).parent / "corpus" / "chat.jsonl"


def raise_exception(message: str):
    raise ValueError(message)


def openai_messages(case: dict) -> list[dict]:
    """The case's messages as requests send them (reasoning as reasoning_content)."""
    messages = []
    for m in case["messages"]:
        msg = {"role": m["role"], "content": m.get("content", "")}
        if "reasoning" in m:
            msg["reasoning_content"] = m["reasoning"]
        if "tool_calls" in m:
            msg["tool_calls"] = m["tool_calls"]
        messages.append(msg)
    return messages


def render(template, case: dict) -> str:
    options = dict(case.get("options", {}))
    add_generation_prompt = options.pop("add_generation_prompt", True)
    return template.render(messages=openai_messages(case), add_generation_prompt=add_generation_prompt, **options)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """transformers' tojson filter."""
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


# The time both sides' strftime_now formats: 2026-10-30 00:00 UTC.
NOW = 1793318400


def check_model_template(args, header: Path, cases: list[dict], fail: Failures) -> None:
    """A model's own template, rendered by zllm.chat.jinja and by jinja2 as transformers sets it up."""
    meta = gguf_metadata(header)
    tokens = meta["tokenizer.ggml.tokens"]
    special = {}
    for name in ("bos", "eos"):
        if f"tokenizer.ggml.{name}_token_id" in meta:
            special[f"{name}_token"] = tokens[meta[f"tokenizer.ggml.{name}_token_id"]]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.fromtimestamp(NOW, timezone.utc).strftime(fmt)
    template = env.from_string(meta["tokenizer.chat_template"])

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8", delete=False) as f:
        for case in cases:
            f.write(json.dumps({**case, "messages": openai_messages(case)}, ensure_ascii=False) + "\n")
        converted = Path(f.name)
    try:
        proc = subprocess.run([args.zllm, "chat-render", str(converted), str(header), "--now", str(NOW)], check=True, capture_output=True, text=True, encoding="utf-8")
    finally:
        converted.unlink()
    got = [json.loads(line) for line in proc.stdout.split("\n") if line]
    fail.expect_eq(f"{header.name}: case count", len(got), len(cases))
    matched = refused = 0
    for case, out in zip(cases, got):
        options = dict(case.get("options", {}))
        add_generation_prompt = options.pop("add_generation_prompt", True)
        tools = options.pop("tools", None)
        try:
            want = template.render(messages=openai_messages(case), tools=tools, documents=None, add_generation_prompt=add_generation_prompt, **special, **options)
        except ValueError as e:  # raise_exception
            want = None
            fail.expect_eq(f"{header.name} {case['name']}: raised", out.get("error"), str(e))
            refused += out.get("error") == str(e)
            continue
        except Exception as e:  # a runtime error in the template
            fail.expect(f"{header.name} {case['name']}: error", "error" in out, f"jinja2 raised {type(e).__name__}: {e}; zllm rendered")
            refused += "error" in out
            continue
        if out.get("text") == want:
            matched += 1
        else:
            fail.expect_eq(f"{header.name} {case['name']}: prompt", out.get("text", out.get("error")), want)
    print(f"{header.name}: {matched} conversations match, {refused} refused by both, of {len(cases)}")


def main() -> int:
    args = parse_args(__doc__)
    paths = fixture_paths(args.fixtures)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(paths.chat_template.read_text(encoding="utf-8"))

    cases = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").split("\n") if line]
    proc = subprocess.run([args.zllm, "chat-render", str(CORPUS)], check=True, capture_output=True, text=True, encoding="utf-8")
    got = [json.loads(line) for line in proc.stdout.split("\n") if line]

    fail = Failures()
    fail.expect_eq("case count", len(got), len(cases))
    for case, out in zip(cases, got):
        want = render(template, case)
        fail.expect_eq(f"{case['name']}: prompt", out["text"], want)
    print(f"chat template: {len(cases) - len(fail.items)}/{len(cases)} conversations match")
    for model in DENSE_MODELS:
        header = model.gguf_header(paths.root)
        if not header.is_file():
            raise SystemExit(f"missing {header}; run scripts/fixtures.sh vocab")
        check_model_template(args, header, cases, fail)
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
