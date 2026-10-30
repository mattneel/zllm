import { expect, test } from "bun:test";
import {
  parseToolCalls,
  type Tool,
  type ToolCallResult,
  ToolCallSplitter,
  type ToolFormat,
} from "../zllm/tools";

const weather: Tool = {
  type: "function",
  function: {
    name: "get_weather",
    parameters: {
      type: "object",
      properties: {
        city: { type: "string" },
        days: { type: "integer" },
        scale: { type: "number" },
        metric: { type: "boolean" },
        filters: { type: "object" },
        note: { type: ["string", "null"] },
      },
    },
  },
};

const call = (name: string, params: Record<string, string>) =>
  `<tool_call>\n<function=${name}>\n${Object.entries(params)
    .map(([k, v]) => `<parameter=${k}>\n${v}\n</parameter>\n`)
    .join("")}</function>\n</tool_call>`;

/** Feeds `text` to a splitter in pieces of `n` characters. */
function split(
  text: string,
  n: number,
  format: ToolFormat = "xml",
): { content: string; calls: ToolCallResult } {
  const s = new ToolCallSplitter([weather], format);
  let content = "";
  for (let i = 0; i < text.length; i += n) {
    for (const part of s.push(text.slice(i, i + n))) {
      content += part;
    }
  }
  return { content, calls: s.finish() };
}

test("values convert by the tool's schema", () => {
  const [c] = parseToolCalls(
    call("get_weather", {
      city: "007",
      days: "3",
      scale: "1.5",
      metric: "true",
      filters: '{"rain": [1, 2]}',
      note: "line one\nline two",
    }),
    [weather],
  );
  expect(c?.function.name).toBe("get_weather");
  expect(JSON.parse(c?.function.arguments ?? "")).toEqual({
    city: "007",
    days: 3,
    scale: 1.5,
    metric: true,
    filters: { rain: [1, 2] },
    note: "line one\nline two",
  });
});

test("values that don't fit their type, and unknown tools, fall back to JSON or text", () => {
  const [a, b] = parseToolCalls(
    call("get_weather", { days: "three", metric: "yes" }) + call("other", { n: "42", s: "hi" }),
    [weather],
  );
  expect(JSON.parse(a?.function.arguments ?? "")).toEqual({ days: "three", metric: "yes" });
  expect(JSON.parse(b?.function.arguments ?? "")).toEqual({ n: 42, s: "hi" });
});

test("text before a call streams; the calls come at the end, whatever the chunking", () => {
  const reply = `Let me check.\n\n${call("get_weather", { city: "Paris" })}\n${call("get_weather", { city: "Oslo" })}`;
  for (const n of [1, 3, 7, 1000]) {
    const { content, calls } = split(reply, n);
    expect(content).toBe("Let me check.");
    expect(calls.rest).toBe("");
    expect(calls.calls.map((c) => JSON.parse(c.function.arguments).city)).toEqual([
      "Paris",
      "Oslo",
    ]);
  }
});

test("text without calls passes through, including a partial tag at the end", () => {
  for (const n of [1, 4, 1000]) {
    const { content, calls } = split("Use <tool_ca", n);
    expect(content + calls.rest).toBe("Use <tool_ca");
    expect(calls.calls).toEqual([]);
  }
});

test("a call that doesn't parse is returned as text", () => {
  const { content, calls } = split("Hm <tool_call>\n<function=x>\nbroken", 5);
  expect(content).toBe("Hm");
  expect(calls).toEqual({ calls: [], rest: "<tool_call>\n<function=x>\nbroken" });
});

const args = (c: { function: { arguments: string } } | undefined): unknown =>
  JSON.parse(c?.function.arguments ?? "");

test("hermes: JSON calls in tool_call tags, after text, whatever the chunking (Qwen 3)", () => {
  const reply =
    'Checking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "days": 2}}\n</tool_call>\n' +
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>';
  for (const n of [1, 3, 7, 1000]) {
    const { content, calls } = split(reply, n, "hermes");
    expect(content).toBe("Checking.");
    expect(calls.calls.map((c) => c.function.name)).toEqual(["get_weather", "get_weather"]);
    expect(calls.calls.map(args)).toEqual([{ city: "Paris", days: 2 }, { city: "Oslo" }]);
  }
  // An unclosed last call still parses, as vLLM's hermes parser takes it.
  expect(parseToolCalls('<tool_call>{"name": "f", "arguments": {}}', [], "hermes").length).toBe(1);
  // A call that isn't JSON is no call: the reply comes back as text.
  const bad = split("<tool_call>{name: f}</tool_call>", 1000, "hermes");
  expect(bad.calls).toEqual({ calls: [], rest: "<tool_call>{name: f}</tool_call>" });
});

test("llama3_json: a reply that is a JSON call, or several, is calls (Llama 3)", () => {
  const reply =
    '{"name": "get_weather", "parameters": {"city": "Paris; France", "tags": ["a}"]}}; {"name": "get_weather", "parameters": {"city": "Oslo"}}';
  for (const text of [reply, ` \n${reply}`, `<|python_tag|>${reply}`]) {
    for (const n of [1, 3, 7, 1000]) {
      const { content, calls } = split(text, n, "llama3_json");
      expect(content).toBe("");
      expect(calls.calls.map(args)).toEqual([
        { city: "Paris; France", tags: ["a}"] },
        { city: "Oslo" },
      ]);
    }
  }
});

test("llama3_json: a reply that starts with text is text, braces or tags later included", () => {
  for (const n of [1, 3, 1000]) {
    const { content, calls } = split(
      'The answer is {"name": "x"} <|python_tag|>',
      n,
      "llama3_json",
    );
    expect(content).toBe('The answer is {"name": "x"} <|python_tag|>');
    expect(calls.calls).toEqual([]);
  }
});
