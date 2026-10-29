// Development server for the demo: bundles it, and serves zllm.wasm and
// model files (with HTTP range requests, which the shim needs).
//
//   bun run dev [--port 8080] [--models DIR]
//
// Models in DIR are served under /models/. Build zllm.wasm first:
// `zig build wasm` at the repository root.

import { parseArgs } from "node:util";

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    port: { type: "string", default: "8080" },
    models: { type: "string", default: "../.fixtures/models" },
  },
});

const root = new URL("../../", import.meta.url).pathname;
const wasm = new URL("../../../zig-out/web/zllm.wasm", import.meta.url).pathname;
const modelDir = values.models;

const built = await Bun.build({
  entrypoints: [`${root}src/demo/main.ts`],
  target: "browser",
  format: "esm",
  minify: true,
  sourcemap: "linked",
});
if (!built.success) {
  for (const log of built.logs) {
    console.error(log);
  }
  process.exit(1);
}
const assets = new Map<string, Blob>();
for (const out of built.outputs) {
  assets.set(`/${out.path.replace(/^\.\//, "")}`, out);
}

/** A file, or the byte range a Range header asks for. */
async function serveFile(req: Request, path: string): Promise<Response> {
  const file = Bun.file(path);
  if (!(await file.exists())) {
    return new Response("not found", { status: 404 });
  }
  // Last-Modified lets the shim cache the file (and notice when it changes).
  const lastModified = new Date(file.lastModified).toUTCString();
  const range = req.headers.get("Range")?.match(/^bytes=(\d+)-(\d*)$/);
  if (range?.[1] === undefined) {
    return new Response(file, {
      headers: { "Accept-Ranges": "bytes", "Last-Modified": lastModified },
    });
  }
  const start = Number(range[1]);
  const end = range[2] ? Math.min(Number(range[2]), file.size - 1) : file.size - 1;
  if (start > end) {
    return new Response(null, {
      status: 416,
      headers: { "Content-Range": `bytes */${file.size}` },
    });
  }
  return new Response(file.slice(start, end + 1), {
    status: 206,
    headers: {
      "Accept-Ranges": "bytes",
      "Last-Modified": lastModified,
      "Content-Range": `bytes ${start}-${end}/${file.size}`,
      "Content-Length": String(end - start + 1),
    },
  });
}

const server = Bun.serve({
  port: Number(values.port),
  hostname: "0.0.0.0",
  async fetch(req) {
    const { pathname } = new URL(req.url);
    if (pathname === "/") {
      return new Response(Bun.file(`${root}demo/index.html`));
    }
    if (pathname === "/zllm.wasm") {
      return new Response(Bun.file(wasm), { headers: { "Content-Type": "application/wasm" } });
    }
    if (pathname.startsWith("/models/")) {
      const name = decodeURIComponent(pathname.slice("/models/".length));
      if (name.includes("/") || name.startsWith(".")) {
        return new Response("bad model name", { status: 400 });
      }
      return serveFile(req, `${modelDir}/${name}`);
    }
    const asset = assets.get(pathname);
    if (asset !== undefined) {
      return new Response(asset);
    }
    return new Response("not found", { status: 404 });
  },
});
console.log(`zllm demo on http://localhost:${server.port}/ (models from ${modelDir})`);
