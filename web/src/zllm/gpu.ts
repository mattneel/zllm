// The browser GPU host: executes the core's command batches
// (src/gpu/cmd.zig) on a WebGPU device, as src/gpu/host.zig does on Dawn.
// Commands between two submits go into one command encoder, consecutive
// dispatches into one compute pass. Reads complete into wasm memory once
// their submit has finished.

import { Kind, Op } from "./abi";
import type { ModelSource } from "./source";

/** Largest piece of a `write_file` range read at once. */
const READ_PIECE = 32 << 20;
/** Range reads in flight at once. */
const READ_CONCURRENCY = 6;
/** WebGPU flag values (GPUBufferUsage.MAP_READ | COPY_DST, GPUMapMode.READ). */
const STAGING_USAGE = 0x0001 | 0x0008;
const MAP_READ = 0x0001;

export interface LoadProgress {
  /** Weight bytes uploaded so far, and in total once known. */
  loaded: number;
  total: number;
}

interface PendingRead {
  staging: GPUBuffer;
  dst: number;
  size: number;
}

/** Device limits the core plans buffers against. */
export interface PlanLimits {
  max_binding: number;
  offset_align: number;
}

/** Requests a device with the adapter's largest buffer limits. */
export async function requestDevice(): Promise<{ device: GPUDevice; adapter: string }> {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU is not available in this browser");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) {
    throw new Error("no WebGPU adapter");
  }
  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  const info = adapter.info;
  const name = info.description || [info.vendor, info.architecture].filter((s) => s).join(" ");
  return { device, adapter: name || "WebGPU adapter" };
}

export function planLimits(device: GPUDevice): PlanLimits {
  return {
    max_binding: device.limits.maxStorageBufferBindingSize,
    offset_align: device.limits.minStorageBufferOffsetAlignment,
  };
}

/** The GPU device was lost (a driver reset, a hang, `destroy()`): the request in flight fails with it. */
export class DeviceLostError extends Error {
  override name = "DeviceLostError";
}

export class Executor {
  readonly #device: GPUDevice;
  readonly #memory: WebAssembly.Memory;
  readonly #source: ModelSource;
  readonly #decoder = new TextDecoder();
  readonly #buffers: (GPUBuffer | undefined)[] = [];
  readonly #shaders: (GPUShaderModule | undefined)[] = [];
  readonly #pipelines: (Promise<GPUComputePipeline> | undefined)[] = [];
  readonly #ready: (GPUComputePipeline | undefined)[] = [];
  readonly #bindGroups: (GPUBindGroup | undefined)[] = [];
  #encoder: GPUCommandEncoder | null = null;
  #pass: GPUComputePassEncoder | null = null;
  #reads: PendingRead[] = [];
  #uploads: Promise<void>[] = [];
  #error: GPUError | null = null;
  #lost: string | null = null;

  /** Weight upload progress, across every batch. */
  readonly progress: LoadProgress = { loaded: 0, total: 0 };
  onProgress: ((p: LoadProgress) => void) | null = null;

  constructor(device: GPUDevice, memory: WebAssembly.Memory, source: ModelSource) {
    this.#device = device;
    this.#memory = memory;
    this.#source = source;
    device.addEventListener("uncapturederror", (ev) => {
      this.#error ??= ev.error;
    });
    void device.lost.then((info) => {
      this.#lost = info.message || info.reason;
    });
  }

  /**
   * Encodes and submits one batch: `words` u32 words at byte address `ptr`.
   * Resolves once it is on the queue, so the next batch can follow while it
   * runs; `done` resolves when its submits have finished and their reads
   * are in wasm memory, with the GPU milliseconds the engine budgets: from
   * each submit, or the previous completion if later, to its completion.
   */
  async execute(ptr: number, words: number): Promise<{ done: Promise<number> }> {
    // A copy: the wasm memory may grow (detaching views) while we await.
    const batch = new Uint32Array(this.#memory.buffer, ptr, words).slice();
    const submits: Promise<number>[] = [];
    let at = 0;
    while (at < batch.length) {
      const header = word(batch, at);
      const n = header >>> 8;
      if (n === 0 || at + n > batch.length) {
        throw new Error(`malformed command at word ${at}`);
      }
      const a = batch.subarray(at + 1, at + n);
      at += n;
      const op = header & 0xff;
      if (op === Op.submit) {
        submits.push((await this.#submit()).done);
      } else {
        await this.#run(op, a);
      }
      this.#check();
    }
    const done = Promise.all(submits).then((times) => {
      this.#check();
      return times.reduce((sum, t) => sum + t, 0);
    });
    return { done };
  }

  /** Destroys every buffer the core created. */
  destroyAll(): void {
    for (const b of this.#buffers) {
      b?.destroy();
    }
    this.#buffers.length = 0;
  }

  /** Runs one command other than `submit`. */
  async #run(op: number, a: Uint32Array): Promise<void> {
    const d = this.#device;
    switch (op) {
      case Op.createBuffer:
        put(this.#buffers, word(a, 0), d.createBuffer({ size: u64(a, 1), usage: word(a, 3) }));
        return;
      case Op.writeBuffer: {
        const len = u64(a, 5);
        d.queue.writeBuffer(
          this.#buffer(word(a, 0)),
          u64(a, 1),
          this.#memory.buffer,
          u64(a, 3),
          len,
        );
        return;
      }
      case Op.writeFile:
        this.#writeFile(this.#buffer(word(a, 0)), u64(a, 1), u64(a, 3), u64(a, 5));
        return;
      case Op.createShader: {
        const bytes = new Uint8Array(this.#memory.buffer, u64(a, 1), word(a, 3));
        put(this.#shaders, word(a, 0), d.createShaderModule({ code: this.#decoder.decode(bytes) }));
        return;
      }
      case Op.createPipeline: {
        const id = word(a, 0);
        const module = get(this.#shaders, word(a, 1), "shader");
        const pending = d.createComputePipelineAsync({
          layout: "auto",
          compute: { module, entryPoint: "main" },
        });
        put(this.#pipelines, id, pending);
        void pending.then((p) => {
          this.#ready[id] = p;
        });
        return;
      }
      case Op.createBindGroup: {
        const pipeline = await get(this.#pipelines, word(a, 1), "pipeline");
        const count = word(a, 3);
        const entries: GPUBindGroupEntry[] = [];
        for (let i = 0; i < count; i++) {
          const e = a.subarray(4 + i * 6, 10 + i * 6);
          entries.push({
            binding: word(e, 0),
            resource: { buffer: this.#buffer(word(e, 1)), offset: u64(e, 2), size: u64(e, 4) },
          });
        }
        const layout = pipeline.getBindGroupLayout(word(a, 2));
        put(this.#bindGroups, word(a, 0), d.createBindGroup({ layout, entries }));
        return;
      }
      case Op.dispatch: {
        const pipeline = await this.#pipeline(word(a, 0));
        const pass = this.#computePass();
        pass.setPipeline(pipeline);
        const count = word(a, 4);
        for (let i = 0; i < count; i++) {
          pass.setBindGroup(i, get(this.#bindGroups, word(a, 5 + i), "bind group"));
        }
        pass.dispatchWorkgroups(word(a, 1), word(a, 2), word(a, 3));
        return;
      }
      case Op.copyBuffer:
        this.#copyEncoder().copyBufferToBuffer(
          this.#buffer(word(a, 0)),
          u64(a, 1),
          this.#buffer(word(a, 3)),
          u64(a, 4),
          u64(a, 6),
        );
        return;
      case Op.clearBuffer:
        this.#copyEncoder().clearBuffer(this.#buffer(word(a, 0)), u64(a, 1), u64(a, 3));
        return;
      case Op.readBuffer: {
        const size = u64(a, 4);
        const staging = d.createBuffer({
          size,
          usage: STAGING_USAGE,
        });
        this.#copyEncoder().copyBufferToBuffer(
          this.#buffer(word(a, 1)),
          u64(a, 2),
          staging,
          0,
          size,
        );
        this.#reads.push({ staging, dst: u64(a, 6), size });
        return;
      }
      case Op.destroy:
        this.#destroy(word(a, 0), word(a, 1));
        return;
      default:
        throw new Error(`unknown command ${op}`);
    }
  }

  /** Queues a file range into `buffer`, read in pieces with bounded concurrency. */
  #writeFile(buffer: GPUBuffer, offset: number, fileOffset: number, size: number): void {
    this.progress.total += size;
    for (let done = 0; done < size; done += READ_PIECE) {
      const len = Math.min(READ_PIECE, size - done);
      this.#uploads.push(
        this.#throttled(async () => {
          const bytes = await this.#source.read(fileOffset + done, len);
          this.#device.queue.writeBuffer(buffer, offset + done, bytes);
          this.progress.loaded += len;
          this.onProgress?.(this.progress);
        }),
      );
    }
  }

  #inFlight = 0;
  #waiting: (() => void)[] = [];

  async #throttled(task: () => Promise<void>): Promise<void> {
    while (this.#inFlight >= READ_CONCURRENCY) {
      const { promise, resolve } = Promise.withResolvers<void>();
      this.#waiting.push(resolve);
      await promise;
    }
    this.#inFlight += 1;
    try {
      await task();
    } finally {
      this.#inFlight -= 1;
      this.#waiting.shift()?.();
    }
  }

  /** When the latest submit finished. */
  #lastDone = 0;

  /**
   * Submits the open encoder. Resolves once submitted; `done` resolves at
   * its completion (reads copied into wasm memory) with its GPU time.
   */
  async #submit(): Promise<{ done: Promise<number> }> {
    // File writes are queue writes: they must land before this submit.
    const uploads = this.#uploads;
    this.#uploads = [];
    await Promise.all(uploads);
    this.#endPass();
    const t0 = performance.now();
    if (this.#encoder !== null) {
      this.#device.queue.submit([this.#encoder.finish()]);
      this.#encoder = null;
    }
    const reads = this.#reads;
    this.#reads = [];
    const finished =
      reads.length === 0
        ? this.#device.queue.onSubmittedWorkDone()
        : Promise.all(reads.map((r) => r.staging.mapAsync(MAP_READ, 0, r.size)));
    const done = finished
      .catch((e: unknown) => {
        // Nothing unmaps or destroys a staging buffer before its map resolves,
        // so an aborted map means the device was lost.
        if (this.#lost !== null || (e instanceof DOMException && e.name === "AbortError")) {
          throw new DeviceLostError(
            `WebGPU device lost: ${this.#lost ?? (e instanceof Error ? e.message : String(e))}`,
          );
        }
        throw e;
      })
      .then(() => {
        const t1 = performance.now();
        const elapsed = t1 - Math.max(t0, this.#lastDone);
        this.#lastDone = t1;
        for (const r of reads) {
          const src = new Uint8Array(r.staging.getMappedRange(0, r.size));
          new Uint8Array(this.#memory.buffer, r.dst, r.size).set(src);
          r.staging.unmap();
          r.staging.destroy();
        }
        return elapsed;
      });
    return { done };
  }

  #destroy(kind: number, id: number): void {
    switch (kind) {
      case Kind.buffer:
        get(this.#buffers, id, "buffer").destroy();
        this.#buffers[id] = undefined;
        return;
      case Kind.shader:
        this.#shaders[id] = undefined;
        return;
      case Kind.pipeline:
        this.#pipelines[id] = undefined;
        this.#ready[id] = undefined;
        return;
      case Kind.bindGroup:
        this.#bindGroups[id] = undefined;
        return;
      default:
        throw new Error(`unknown object kind ${kind}`);
    }
  }

  #buffer(id: number): GPUBuffer {
    return get(this.#buffers, id, "buffer");
  }

  /** The compiled pipeline; awaits only the first use of one still compiling. */
  async #pipeline(id: number): Promise<GPUComputePipeline> {
    return this.#ready[id] ?? (await get(this.#pipelines, id, "pipeline"));
  }

  #encoderOrNew(): GPUCommandEncoder {
    this.#encoder ??= this.#device.createCommandEncoder();
    return this.#encoder;
  }

  #computePass(): GPUComputePassEncoder {
    this.#pass ??= this.#encoderOrNew().beginComputePass();
    return this.#pass;
  }

  /** The encoder, with any open compute pass ended so copies can follow. */
  #copyEncoder(): GPUCommandEncoder {
    this.#endPass();
    return this.#encoderOrNew();
  }

  #endPass(): void {
    this.#pass?.end();
    this.#pass = null;
  }

  get device(): GPUDevice {
    return this.#device;
  }

  /** Whether the device is lost. */
  get lost(): boolean {
    return this.#lost !== null;
  }

  #check(): void {
    if (this.#lost !== null) {
      throw new DeviceLostError(`WebGPU device lost: ${this.#lost}`);
    }
    if (this.#error !== null) {
      const e = this.#error;
      this.#error = null;
      throw new Error(`WebGPU error: ${e.message}`);
    }
  }
}

function word(a: Uint32Array, i: number): number {
  const w = a[i];
  if (w === undefined) {
    throw new Error("truncated command");
  }
  return w;
}

/** A u64 argument (two words, low first). Exact up to 2^53. */
function u64(a: Uint32Array, i: number): number {
  return word(a, i) + word(a, i + 1) * 2 ** 32;
}

function put<T>(list: (T | undefined)[], id: number, obj: T): void {
  if (list[id] !== undefined) {
    throw new Error(`object id ${id} reused`);
  }
  list[id] = obj;
}

function get<T>(list: (T | undefined)[], id: number, what: string): T {
  const obj = list[id];
  if (obj === undefined) {
    throw new Error(`unknown ${what} ${id}`);
  }
  return obj;
}
