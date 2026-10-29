// Renders ```mermaid fences in the zllm book.
//
// mdBook emits each fence as <pre><code class="language-mermaid">. This
// swaps them for <pre class="mermaid"> and runs Mermaid on them.
// Bundled to docs/theme/mermaid-init.js by `bun run build:book`.

import mermaid from "mermaid";

/** mdBook theme classes on <html> that use a dark background. */
const DARK_THEMES = ["coal", "navy", "ayu"] as const;

/** Replaces each fenced block with a Mermaid container; returns how many. */
function collectDiagrams(root: Document): number {
  let count = 0;
  for (const code of root.querySelectorAll<HTMLElement>("code.language-mermaid")) {
    const pre = code.parentElement;
    if (pre === null) {
      continue;
    }
    const container = root.createElement("pre");
    container.className = "mermaid";
    container.textContent = code.textContent;
    pre.replaceWith(container);
    count += 1;
  }
  return count;
}

function prefersDark(root: Document): boolean {
  const classes = root.documentElement.classList;
  return DARK_THEMES.some((theme) => classes.contains(theme));
}

async function main(): Promise<void> {
  if (collectDiagrams(document) === 0) {
    return;
  }
  mermaid.initialize({ startOnLoad: false, theme: prefersDark(document) ? "dark" : "default" });
  await mermaid.run({ querySelector: "pre.mermaid" });
}

main().catch((error: unknown) => {
  console.error("zllm book: diagram rendering failed", error);
});
