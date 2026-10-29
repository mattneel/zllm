// The zllm demo: load a GGUF over HTTP or from disk, then chat, streaming
// reasoning and content. The engine is also exposed as `window.zllmEngine`
// for scripted checks.

import {
  type ChatMessage,
  cacheUsage,
  clearCache,
  type Engine,
  type LoadOptions,
  load,
} from "../zllm/index";

declare global {
  interface Window {
    zllmEngine?: Engine;
  }
}

function el<T extends HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (e === null) {
    throw new Error(`#${id} missing`);
  }
  return e as T;
}

const status = el<HTMLParagraphElement>("status");
const progress = el<HTMLProgressElement>("progress");
const log = el<HTMLDivElement>("log");
const form = el<HTMLFormElement>("chat");
const promptBox = el<HTMLTextAreaElement>("prompt");
const think = el<HTMLInputElement>("think");
const send = el<HTMLButtonElement>("send");
const stop = el<HTMLButtonElement>("stop");
const stats = el<HTMLParagraphElement>("stats");

let engine: Engine | null = null;
let history: ChatMessage[] = [];
let controller: AbortController | null = null;

async function start(model: LoadOptions["model"], name: string): Promise<void> {
  el<HTMLFieldSetElement>("load-form").disabled = true;
  progress.hidden = false;
  status.textContent = `Loading ${name}…`;
  try {
    engine = await load({
      model,
      onProgress: (p) => {
        progress.max = p.total;
        progress.value = p.loaded;
        status.textContent = `Loading ${name}: ${(p.loaded / 2 ** 20).toFixed(0)} of ${(p.total / 2 ** 20).toFixed(0)} MiB`;
      },
    });
    window.zllmEngine = engine;
    const s = engine.stats();
    const from = { hit: "from the cache", filled: "and cached", off: "uncached" }[s.cache];
    status.textContent = `${name} on ${s.adapter}, loaded in ${(s.loadMs / 1000).toFixed(1)} s (${from}).`;
    progress.hidden = true;
    form.hidden = false;
    promptBox.focus();
  } catch (e) {
    status.textContent = `Load failed: ${e instanceof Error ? e.message : String(e)}`;
    el<HTMLFieldSetElement>("load-form").disabled = false;
    throw e;
  }
}

function bubble(cls: string, text = ""): HTMLDivElement {
  const div = document.createElement("div");
  div.className = `msg ${cls}`;
  div.textContent = text;
  log.append(div);
  return div;
}

async function chat(text: string): Promise<void> {
  if (engine === null) {
    return;
  }
  history.push({ role: "user", content: text });
  bubble("user", text);
  const reasoningBox = think.checked ? bubble("reasoning") : null;
  const answer = bubble("assistant");
  let content = "";
  let reasoning = "";
  controller = new AbortController();
  send.disabled = true;
  stop.disabled = false;
  try {
    const chunks = engine.chat.completions.create({
      messages: history,
      stream: true,
      enable_thinking: think.checked,
      max_tokens: 4096,
      signal: controller.signal,
    });
    for await (const c of chunks) {
      const delta = c.choices[0]?.delta;
      if (delta?.reasoning_content !== undefined && reasoningBox !== null) {
        reasoning += delta.reasoning_content;
        reasoningBox.textContent = reasoning;
      }
      if (delta?.content !== undefined) {
        content += delta.content;
        answer.textContent = content;
      }
      if (c.usage !== undefined) {
        const u = engine.stats().last;
        if (u !== null) {
          const prefilled = u.promptTokens - u.cachedTokens;
          stats.textContent =
            `${c.choices[0]?.finish_reason}; prompt ${u.promptTokens} tokens (${u.cachedTokens} reused), ` +
            `prefill ${prefilled} in ${u.prefillMs.toFixed(0)} ms ` +
            `(${((prefilled * 1000) / Math.max(u.prefillMs, 1e-9)).toFixed(0)} tok/s); ` +
            `${u.completionTokens} generated in ${u.decodeMs.toFixed(0)} ms ` +
            `(${(((u.completionTokens - 1) * 1000) / Math.max(u.decodeMs, 1e-9)).toFixed(1)} tok/s)`;
        }
      }
    }
    const turn: ChatMessage = { role: "assistant", content };
    if (reasoning !== "") {
      turn.reasoning_content = reasoning;
    }
    history.push(turn);
  } finally {
    controller = null;
    send.disabled = false;
    stop.disabled = true;
  }
}

el<HTMLButtonElement>("load-url").addEventListener("click", () => {
  const url = el<HTMLInputElement>("model-url").value;
  void start(url, url.split("/").pop() ?? url);
});
el<HTMLInputElement>("model-file").addEventListener("change", (ev) => {
  const file = (ev.target as HTMLInputElement).files?.[0];
  if (file !== undefined) {
    void start(file, file.name);
  }
});
form.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = promptBox.value.trim();
  if (text !== "") {
    promptBox.value = "";
    void chat(text);
  }
});
promptBox.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    form.requestSubmit();
  }
});
stop.addEventListener("click", () => controller?.abort());
el<HTMLButtonElement>("clear-cache").addEventListener("click", async () => {
  const before = await cacheUsage();
  await clearCache();
  status.textContent = `Cleared ${before.entries} cached model(s), ${(before.bytes / 2 ** 20).toFixed(0)} MiB.`;
});
el<HTMLButtonElement>("reset").addEventListener("click", () => {
  history = [];
  log.replaceChildren();
  stats.textContent = "";
});
