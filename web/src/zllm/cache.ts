// The weight cache: model files in the origin private file system (OPFS),
// under zllm/<key>, where the key hashes the source's identity (its URL and
// the server's ETag or Last-Modified) and its size. A changed file gets a
// new validator and so a new key: an entry is never stale. URLs served
// without a validator are not cached.
//
// A first load reads from the network and writes every range it reads into
// a writable stream; the entry is committed (and marked complete) only
// after the load succeeds. Later loads read the entry from disk. Loading
// reads the same ranges every time, so the entry holds exactly them.

import type { ModelSource } from "./source";

const DIR = "zllm";
/**
 * Written last, holding the source's URL: an entry without it is
 * incomplete and ignored.
 */
const COMPLETE = ".complete";

/** A source being cached: commit once the load succeeds, else discard. */
export interface Filling {
  source: ModelSource;
  commit(): Promise<void>;
  discard(): Promise<void>;
}

export type Cached = { hit: ModelSource } | { fill: Filling } | { none: ModelSource };

async function cacheDir(): Promise<FileSystemDirectoryHandle | null> {
  if (!("storage" in navigator) || typeof navigator.storage.getDirectory !== "function") {
    return null;
  }
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(DIR, { create: true });
}

/**
 * The cached entry for `source`, or a filling one. Sources without an
 * identity, or without room, pass through.
 */
export async function openCached(source: ModelSource): Promise<Cached> {
  if (source.identity === undefined) {
    return { none: source };
  }
  const dir = await cacheDir();
  if (dir === null) {
    return { none: source };
  }
  const key = await cacheKey(source);
  try {
    const entry = await dir.getDirectoryHandle(key);
    await entry.getFileHandle(COMPLETE);
    const file = await (await entry.getFileHandle("model.gguf")).getFile();
    if (file.size === source.size) {
      return { hit: blobSource(file, source.identity) };
    }
  } catch {
    // No complete entry.
  }

  const { usage = 0, quota = 0 } = await navigator.storage.estimate();
  if (quota - usage < source.size) {
    return { none: source };
  }
  await navigator.storage.persist();
  // This key's partial entry, and older versions of the same URL.
  const url = source.identity.split("\n")[0] ?? "";
  const stale: string[] = [];
  for await (const old of dir.values()) {
    if (old.kind === "directory" && (old.name === key || (await entryUrl(old)) === url)) {
      stale.push(old.name);
    }
  }
  for (const name of stale) {
    await dir.removeEntry(name, { recursive: true });
  }
  const entry = await dir.getDirectoryHandle(key, { create: true });
  const handle = await entry.getFileHandle("model.gguf", { create: true });
  const stream = await handle.createWritable();
  await stream.truncate(source.size);
  // Writes are chained: the stream takes one at a time.
  let writes: Promise<void> = Promise.resolve();
  let failed: unknown = null;
  return {
    fill: {
      source: {
        size: source.size,
        identity: source.identity,
        async read(offset, length) {
          const bytes = await source.read(offset, length);
          writes = writes.then(() =>
            stream.write({ type: "write", position: offset, data: bytes }).catch((e: unknown) => {
              failed ??= e;
            }),
          );
          return bytes;
        },
      },
      async commit() {
        await writes;
        if (failed !== null) {
          await stream.abort();
          await dir.removeEntry(key, { recursive: true });
          return;
        }
        await stream.close();
        const done = await (await entry.getFileHandle(COMPLETE, { create: true })).createWritable();
        await done.write(url);
        await done.close();
      },
      async discard() {
        await writes;
        await stream.abort();
        await dir.removeEntry(key, { recursive: true });
      },
    },
  };
}

/** The URL a complete entry holds, or null. */
async function entryUrl(entry: FileSystemDirectoryHandle): Promise<string | null> {
  try {
    return await (await (await entry.getFileHandle(COMPLETE)).getFile()).text();
  } catch {
    return null;
  }
}

function blobSource(file: Blob, identity: string): ModelSource {
  return {
    size: file.size,
    identity,
    async read(offset, length) {
      return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
    },
  };
}

async function cacheKey(source: ModelSource): Promise<string> {
  const id = new TextEncoder().encode(`${source.identity}\n${source.size}`);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", id));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

export interface CacheUsage {
  /** Complete cached models, and their bytes. */
  entries: number;
  bytes: number;
}

/** What the weight cache holds. */
export async function cacheUsage(): Promise<CacheUsage> {
  const dir = await cacheDir();
  const usage: CacheUsage = { entries: 0, bytes: 0 };
  if (dir === null) {
    return usage;
  }
  for await (const entry of dir.values()) {
    if (entry.kind !== "directory") {
      continue;
    }
    try {
      await entry.getFileHandle(COMPLETE);
      usage.bytes += (await (await entry.getFileHandle("model.gguf")).getFile()).size;
      usage.entries += 1;
    } catch {
      // Incomplete: not counted.
    }
  }
  return usage;
}

/** Deletes every cached model. */
export async function clearCache(): Promise<void> {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(DIR, { recursive: true }).catch(() => undefined);
}
