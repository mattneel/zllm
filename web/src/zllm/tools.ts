// Tool calling in the OpenAI chat format, over the calls each model family
// writes (src/chat/chat.zig `ToolFormat`, named as vLLM's tool parsers):
//
//   xml          <tool_call>                  the fixed Qwen template, Granite 4
//                <function=get_weather>
//                <parameter=city>
//                Paris
//                </parameter>
//                </function>
//                </tool_call>
//   hermes       <tool_call>                  Qwen 3
//                {"name": "get_weather", "arguments": {"city": "Paris"}}
//                </tool_call>
//   llama3_json  {"name": "get_weather", "parameters": {"city": "Paris"}}
//                the whole reply, optionally after <|python_tag|>, calls
//                separated by ";" (Llama 3)

/** How a model writes calls, as zllm_info's `tool_format` names it. */
export type ToolFormat = "xml" | "hermes" | "llama3_json";

export function isToolFormat(v: unknown): v is ToolFormat {
  return v === "xml" || v === "hermes" || v === "llama3_json";
}

/** A function the model may call. */
export interface Tool {
  type: "function";
  function: {
    name: string;
    description?: string;
    /** JSON Schema for the arguments object. */
    parameters?: Record<string, unknown>;
  };
}

/** A call the model made; `arguments` is a JSON object, as text. */
export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

/** The end of a reply: its calls, and text still held back. */
export interface ToolCallResult {
  calls: ToolCall[];
  rest: string;
}

/**
 * Splits reply content into text and the calls after it: text streams until
 * a `<tool_call>` tag, and everything from the first tag on is parsed at the
 * end. Text that could be the start of a tag, and whitespace before one, is
 * held back until it can't be. In `llama3_json`, a reply is calls if it
 * starts with `{` or `<|python_tag|>`, else text.
 */
export class ToolCallSplitter {
  static readonly START = "<tool_call>";
  static readonly PYTHON_TAG = "<|python_tag|>";
  #held = "";
  #calls: string | null = null;
  /** llama3_json: no text seen yet but whitespace. */
  #leading = true;

  constructor(
    readonly tools: readonly Tool[],
    readonly format: ToolFormat = "xml",
  ) {}

  *push(text: string): Generator<string> {
    if (this.#calls !== null) {
      this.#calls += text;
      return;
    }
    if (this.format === "llama3_json") {
      yield* this.#pushJson(text);
      return;
    }
    this.#held += text;
    const at = this.#held.indexOf(ToolCallSplitter.START);
    if (at >= 0) {
      const before = this.#held.slice(0, at).trimEnd();
      this.#calls = this.#held.slice(at);
      this.#held = "";
      if (before !== "") {
        yield before;
      }
      return;
    }
    let keep = 0;
    for (let n = Math.min(ToolCallSplitter.START.length - 1, this.#held.length); n > 0; n--) {
      if (ToolCallSplitter.START.startsWith(this.#held.slice(-n))) {
        keep = n;
        break;
      }
    }
    const ready = this.#held.slice(0, this.#held.length - keep);
    const body = ready.trimEnd();
    this.#held = this.#held.slice(body.length);
    if (body !== "") {
      yield body;
    }
  }

  *#pushJson(text: string): Generator<string> {
    if (!this.#leading) {
      yield text;
      return;
    }
    this.#held += text;
    const lead = this.#held.trimStart();
    if (lead === "" || ToolCallSplitter.PYTHON_TAG.startsWith(lead)) {
      return;
    }
    this.#leading = false;
    const held = this.#held;
    this.#held = "";
    if (lead.startsWith("{") || lead.startsWith(ToolCallSplitter.PYTHON_TAG)) {
      this.#calls = held;
    } else {
      yield held;
    }
  }

  /**
   * The calls, or none if the reply made none (or none that parse); text
   * still held back is returned as `rest`.
   */
  finish(): ToolCallResult {
    if (this.#calls === null) {
      const rest = this.#held;
      this.#held = "";
      return { calls: [], rest };
    }
    const calls = parseToolCalls(this.#calls, this.tools, this.format);
    return calls.length > 0 ? { calls, rest: "" } : { calls: [], rest: this.#calls };
  }
}

const CALL = /<tool_call>\s*<function=([^>\n]+)>([\s\S]*?)<\/function>\s*(?:<\/tool_call>|$)/g;
const PARAM = /<parameter=([^>\n]+)>\n?([\s\S]*?)\n?<\/parameter>/g;
const HERMES = /<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g;

/**
 * Every complete call in `text`, in `format`. XML parameter values are
 * converted by the tool's JSON Schema type, as vLLM's Qwen3 XML parser does:
 * strings stay as written; integers, numbers and booleans are parsed;
 * anything else is parsed as JSON where it is JSON and kept as text where it
 * isn't. JSON calls (vLLM's hermes and llama3_json parsers) that don't parse
 * are not calls.
 */
export function parseToolCalls(
  text: string,
  tools: readonly Tool[],
  format: ToolFormat = "xml",
): ToolCall[] {
  if (format === "hermes") {
    return [...text.matchAll(HERMES)].flatMap((m) => jsonCall(parseJson(m[1] ?? ""), "arguments"));
  }
  if (format === "llama3_json") {
    let rest = text.trimStart();
    if (rest.startsWith(ToolCallSplitter.PYTHON_TAG)) {
      rest = rest.slice(ToolCallSplitter.PYTHON_TAG.length);
    }
    return jsonObjects(rest).flatMap((o) => jsonCall(o, "parameters"));
  }
  const calls: ToolCall[] = [];
  for (const m of text.matchAll(CALL)) {
    const name = (m[1] ?? "").trim();
    const props = schemaProperties(tools.find((t) => t.function.name === name));
    const args: Record<string, unknown> = {};
    for (const p of (m[2] ?? "").matchAll(PARAM)) {
      const key = (p[1] ?? "").trim();
      args[key] = convert(p[2] ?? "", props[key]);
    }
    calls.push({
      id: newId(),
      type: "function",
      function: { name, arguments: JSON.stringify(args) },
    });
  }
  return calls;
}

function newId(): string {
  return `call_${crypto.randomUUID().replaceAll("-", "").slice(0, 24)}`;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text.trim());
  } catch {
    return undefined;
  }
}

/** A call from {"name", <args>}: `args` names the arguments' key ("arguments" is also taken). */
function jsonCall(v: unknown, args: "arguments" | "parameters"): ToolCall[] {
  if (!isRecord(v)) {
    return [];
  }
  const { name, arguments: fallback } = v;
  if (typeof name !== "string") {
    return [];
  }
  const a = v[args] ?? fallback ?? {};
  return [
    {
      id: newId(),
      type: "function",
      function: { name, arguments: typeof a === "string" ? a : JSON.stringify(a) },
    },
  ];
}

/**
 * The JSON objects `text` starts with, separated by whitespace or ";" (as
 * vLLM's llama3_json parser decodes them); parsing stops at what isn't one.
 */
function jsonObjects(text: string): unknown[] {
  const out: unknown[] = [];
  let i = 0;
  while (i < text.length) {
    while (i < text.length && /[\s;]/.test(text[i] ?? "")) {
      i++;
    }
    if (text[i] !== "{") {
      break;
    }
    const end = objectEnd(text, i);
    if (end < 0) {
      break;
    }
    const v = parseJson(text.slice(i, end));
    if (v === undefined) {
      break;
    }
    out.push(v);
    i = end;
  }
  return out;
}

/** The index after the JSON object at `start`, or -1 if it doesn't close. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inString) {
      if (c === "\\") {
        i++;
      } else if (c === '"') {
        inString = false;
      }
    } else if (c === '"') {
      inString = true;
    } else if (c === "{" || c === "[") {
      depth++;
    } else if (c === "}" || c === "]") {
      depth--;
      if (depth === 0) {
        return i + 1;
      }
    }
  }
  return -1;
}

function schemaProperties(tool: Tool | undefined): Record<string, unknown> {
  const { properties } = tool?.function.parameters ?? {};
  return isRecord(properties) ? properties : {};
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function convert(raw: string, schema: unknown): unknown {
  const { type } = isRecord(schema) ? schema : {};
  const types: unknown[] = Array.isArray(type) ? type : [type];
  if (types.includes("string")) {
    return raw;
  }
  const value = raw.trim();
  if (types.includes("integer") && /^-?\d+$/.test(value)) {
    return Number.parseInt(value, 10);
  }
  if (types.includes("number") && value !== "" && Number.isFinite(Number(value))) {
    return Number(value);
  }
  if (types.includes("boolean") && (value === "true" || value === "false")) {
    return value === "true";
  }
  if (value === "null" && types.includes("null")) {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch {
    return raw;
  }
}

/**
 * Arguments as the chat template takes them: an object, parsed from the
 * OpenAI format's JSON text (as vLLM does), or the text where it isn't JSON.
 */
export function argumentsValue(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
