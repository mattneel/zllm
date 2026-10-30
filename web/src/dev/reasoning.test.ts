import { expect, test } from "bun:test";
import { ReasoningSplitter } from "../zllm/reasoning";

/** The reply fed in `chunks`, as (reasoning, content). */
function split(inside: boolean, chunks: string[]): [string, string] {
  const s = new ReasoningSplitter();
  s.start(inside);
  let reasoning = "";
  let content = "";
  const take = (d: { content?: string; reasoning_content?: string }): void => {
    reasoning += d.reasoning_content ?? "";
    content += d.content ?? "";
  };
  for (const c of chunks) {
    for (const d of s.push(c)) take(d);
  }
  for (const d of s.flush()) take(d);
  return [reasoning, content];
}

/** Every split of `text` into two chunks, and one character at a time. */
function chunkings(text: string): string[][] {
  const out = [[text], [...text]];
  for (let i = 1; i < text.length; i++) out.push([text.slice(0, i), text.slice(i)]);
  return out;
}

test("a reply inside the prompt's think block", () => {
  for (const c of chunkings("Let me see.</think>\n\nParis.")) {
    expect(split(true, c)).toEqual(["Let me see.", "Paris."]);
  }
});

test("a reply that opens its own think block (Qwen 3)", () => {
  for (const c of chunkings("<think>\nLet me see.\n</think>\n\nParis.")) {
    expect(split(false, c)).toEqual(["Let me see.\n", "Paris."]);
  }
});

test("a reply without one is all content (Llama)", () => {
  for (const c of chunkings("The capital is Paris.")) {
    expect(split(false, c)).toEqual(["", "The capital is Paris."]);
  }
  // Text that starts like the tag but is not it.
  for (const c of chunkings("<thinking>is fine</thinking>")) {
    expect(split(false, c)).toEqual(["", "<thinking>is fine</thinking>"]);
  }
  for (const c of chunkings("<b>bold</b>")) {
    expect(split(false, c)).toEqual(["", "<b>bold</b>"]);
  }
});

test("a reply that ends while it could still open the block", () => {
  expect(split(false, ["<thi"])).toEqual(["", "<thi"]);
  expect(split(false, ["\n"])).toEqual(["", ""]);
});
