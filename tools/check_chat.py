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
from pathlib import Path

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from common import Failures, fixture_paths, parse_args

CORPUS = Path(__file__).parent / "corpus" / "chat.jsonl"


def raise_exception(message: str):
    raise ValueError(message)


def render(template, case: dict) -> str:
    messages = []
    for m in case["messages"]:
        msg = {"role": m["role"], "content": m["content"]}
        if "reasoning" in m:
            msg["reasoning_content"] = m["reasoning"]
        messages.append(msg)
    options = dict(case.get("options", {}))
    add_generation_prompt = options.pop("add_generation_prompt", True)
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt, **options)


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
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
