import { expect, test } from "bun:test";
import { parseToolCalls, type Tool, type ToolCallResult, ToolCallSplitter } from "../zllm/tools";

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
function split(text: string, n: number): { content: string; calls: ToolCallResult } {
  const s = new ToolCallSplitter([weather]);
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
