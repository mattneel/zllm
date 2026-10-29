// Where the model file comes from. The core only ever sees the GGUF header;
// weights go from the source straight to GPU buffers (`write_file`).

export interface ModelSource {
  /** File size in bytes. */
  readonly size: number;
  /**
   * What names this file for the weight cache (the URL and the server's
   * ETag or Last-Modified); sources without one are not cached.
   */
  readonly identity?: string;
  /** Bytes [offset, offset + length) of the file. */
  read(offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
}

/** A local file, e.g. from `<input type=file>` or a FileSystemFileHandle. */
export function fileSource(file: Blob): ModelSource {
  return {
    size: file.size,
    async read(offset, length) {
      return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
    },
  };
}

/** A URL served with HTTP range requests. */
export async function urlSource(url: string, init: RequestInit = {}): Promise<ModelSource> {
  // Past the HTTP cache: the probe must see the current validator, and the
  // weight cache (cache.ts) keeps the bytes instead.
  const head = await fetch(url, { cache: "no-store", ...init, headers: { Range: "bytes=0-0" } });
  if (head.status !== 206) {
    throw new Error(`${url}: range requests unsupported (status ${head.status})`);
  }
  const total = head.headers.get("Content-Range")?.split("/")[1];
  const size = total === undefined ? Number.NaN : Number(total);
  if (!Number.isSafeInteger(size)) {
    throw new Error(`${url}: no file size in Content-Range`);
  }
  await head.body?.cancel();
  // Without a validator a changed file could not be told apart: no identity.
  const validator = head.headers.get("ETag") ?? head.headers.get("Last-Modified");
  return {
    size,
    ...(validator === null
      ? {}
      : { identity: `${new URL(url, location.href).href}\n${validator}` }),
    async read(offset, length) {
      const res = await fetch(url, {
        cache: "no-store",
        ...init,
        headers: { Range: `bytes=${offset}-${offset + length - 1}` },
      });
      if (res.status !== 206) {
        throw new Error(`${url}: range read failed (status ${res.status})`);
      }
      const bytes = new Uint8Array(await res.arrayBuffer());
      if (bytes.length !== length) {
        throw new Error(`${url}: short read at ${offset} (${bytes.length} of ${length} bytes)`);
      }
      return bytes;
    },
  };
}
