/**
 * Splits a thinking reply into reasoning (up to `</think>`) and content, as
 * vLLM's Qwen3 reasoning parser does. Where the prompt opens the `<think>`
 * block (`start(true)`: Qwen 3.5's and Granite's templates), the reply starts
 * inside it; otherwise a reply that begins with `<think>` (Qwen 3's model
 * writes the tag itself) enters it, and any other reply is all content
 * (Llama).
 */
export class ReasoningSplitter {
  static readonly OPEN = "<think>";
  static readonly END = "</think>";
  #inReasoning = false;
  /** Before the first content: a reply may still open the block. */
  #opening = true;
  #held = "";
  #contentStarted = false;

  /** Where the reply starts: inside the think block the prompt opened, or not. */
  start(inside: boolean): void {
    this.#inReasoning = inside;
    this.#opening = !inside;
  }

  *push(text: string): Generator<{ content?: string; reasoning_content?: string }> {
    if (this.#opening) {
      this.#held += text;
      const lead = this.#held.trimStart();
      if (ReasoningSplitter.OPEN.startsWith(lead)) {
        // Still possibly "<think>" (or only whitespace so far).
        return;
      }
      const held = this.#held;
      this.#held = "";
      this.#opening = false;
      if (lead.startsWith(ReasoningSplitter.OPEN)) {
        this.#inReasoning = true;
        yield* this.push(lead.slice(ReasoningSplitter.OPEN.length).replace(/^\n/, ""));
      } else {
        yield* this.#content(held);
      }
      return;
    }
    if (!this.#inReasoning) {
      yield* this.#content(text);
      return;
    }
    this.#held += text;
    const end = this.#held.indexOf(ReasoningSplitter.END);
    if (end >= 0) {
      const reasoning = this.#held.slice(0, end);
      const rest = this.#held.slice(end + ReasoningSplitter.END.length);
      this.#held = "";
      this.#inReasoning = false;
      if (reasoning !== "") {
        yield { reasoning_content: reasoning };
      }
      yield* this.#content(rest);
      return;
    }
    // Hold back a tail that could start "</think>".
    let keep = 0;
    for (let n = Math.min(ReasoningSplitter.END.length - 1, this.#held.length); n > 0; n--) {
      if (ReasoningSplitter.END.startsWith(this.#held.slice(-n))) {
        keep = n;
        break;
      }
    }
    const ready = this.#held.slice(0, this.#held.length - keep);
    this.#held = this.#held.slice(this.#held.length - keep);
    if (ready !== "") {
      yield { reasoning_content: ready };
    }
  }

  *flush(): Generator<{ content?: string; reasoning_content?: string }> {
    const held = this.#held;
    this.#held = "";
    this.#opening = false;
    if (this.#inReasoning) {
      if (held !== "") {
        yield { reasoning_content: held };
      }
    } else {
      yield* this.#content(held);
    }
  }

  /** Content, without the blank lines the template puts after `</think>`. */
  *#content(text: string): Generator<{ content?: string }> {
    const out = this.#contentStarted ? text : text.replace(/^\n+/, "");
    if (out !== "") {
      this.#contentStarted = true;
      yield { content: out };
    }
  }
}
