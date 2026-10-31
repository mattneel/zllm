import { expect, test } from "bun:test";
import { Op } from "../zllm/abi";
import { Executor } from "../zllm/gpu";

function command(op: number, ...args: number[]): number[] {
  return [((args.length + 1) << 8) | op, ...args];
}

function fixture() {
  const calls: string[] = [];
  const memory = new WebAssembly.Memory({ initial: 1 });
  const compiled = Promise.withResolvers<GPUComputePipeline>();
  const pipeline = {
    getBindGroupLayout: () => ({}) as GPUBindGroupLayout,
  } as unknown as GPUComputePipeline;
  const pass = {
    setPipeline: () => calls.push("pipeline"),
    setBindGroup: () => calls.push("bind"),
    dispatchWorkgroups: () => calls.push("dispatch"),
    end: () => calls.push("end"),
  } as unknown as GPUComputePassEncoder;
  const device = {
    addEventListener: () => {},
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    createShaderModule: () => ({}) as GPUShaderModule,
    createComputePipelineAsync: () => compiled.promise,
    createBindGroup: () => {
      calls.push("createBindGroup");
      return {} as GPUBindGroup;
    },
    createCommandEncoder: () => ({
      beginComputePass: () => pass,
      finish: () => ({}) as GPUCommandBuffer,
    }),
    queue: {
      submit: () => calls.push("submit"),
      onSubmittedWorkDone: () => Promise.resolve(),
    },
  } as unknown as GPUDevice;
  const exec = new Executor(device, memory, {
    size: 0,
    read: async () => {
      throw new Error("this command stream does not read model weights");
    },
  });
  function run(words: number[]) {
    new Uint32Array(memory.buffer, 1024, words.length).set(words);
    return exec.execute(1024, words.length);
  }
  return { calls, compiled, pipeline, run };
}

const kernel = command(Op.createKernel, 0, 0, 0, 0, 0, 0, 0);
const bind = command(Op.createBindGroup, 0, 0, 0, 0);
const dispatch = command(Op.dispatch, 0, 1, 1, 1, 1, 0);
const submit = command(Op.submit, 1);

test("ready pipelines replay consecutive dispatches without microtask yields", async () => {
  const f = fixture();
  await f.run(kernel);
  f.compiled.resolve(f.pipeline);
  await f.compiled.promise;
  await f.run(bind);
  f.calls.length = 0;
  const replay = f.run([...dispatch, ...dispatch, ...submit]);
  expect(f.calls).toEqual(["pipeline", "bind", "dispatch", "pipeline", "bind", "dispatch"]);
  await (await replay).done;
  expect(f.calls.slice(-2)).toEqual(["end", "submit"]);
});

test("a compiling pipeline suspends dependent commands and preserves their order", async () => {
  const f = fixture();
  await f.run(kernel);
  const replay = f.run([...bind, ...dispatch, ...submit]);
  expect(f.calls).toEqual([]);
  f.compiled.resolve(f.pipeline);
  await (await replay).done;
  expect(f.calls).toEqual(["createBindGroup", "pipeline", "bind", "dispatch", "end", "submit"]);
});

test("pipeline compilation failure rejects replay before dispatch or submit", async () => {
  const f = fixture();
  await f.run(kernel);
  const replay = f.run([...bind, ...dispatch, ...submit]);
  f.compiled.reject(new Error("shader compilation failed"));
  await expect(replay).rejects.toThrow("shader compilation failed");
  expect(f.calls).toEqual([]);
});
