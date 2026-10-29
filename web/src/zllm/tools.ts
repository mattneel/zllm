// Tool calling in the OpenAI chat format, over the Qwen template's XML calls:
//
//   <tool_call>
//   <function=get_weather>
//   <parameter=city>
//   Paris
//   </parameter>
//   </function>
//   </tool_call>

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
 * held back until it can't be.
 */
export class ToolCallSplitter {
  static readonly START = "<tool_call>";
  #held = "";
  #calls: string | null = null;

  constructor(readonly tools: readonly Tool[]) {}

  *push(text: string): Generator<string> {
    if (this.#calls !== null) {
      this.#calls += text;
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
    const calls = parseToolCalls(this.#calls, this.tools);
    return calls.length > 0 ? { calls, rest: "" } : { calls: [], rest: this.#calls };
  }
}

const CALL = /<tool_call>\s*<function=([^>\n]+)>([\s\S]*?)<\/function>\s*(?:<\/tool_call>|$)/g;
const PARAM = /<parameter=([^>\n]+)>\n?([\s\S]*?)\n?<\/parameter>/g;

/**
 * Every complete call in `text`. Parameter values are converted by the
 * tool's JSON Schema type, as vLLM's Qwen3 XML parser does: strings stay as
 * written; integers, numbers and booleans are parsed; anything else is
 * parsed as JSON where it is JSON and kept as text where it isn't.
 */
export function parseToolCalls(text: string, tools: readonly Tool[]): ToolCall[] {
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
      id: `call_${crypto.randomUUID().replaceAll("-", "").slice(0, 24)}`,
      type: "function",
      function: { name, arguments: JSON.stringify(args) },
    });
  }
  return calls;
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
