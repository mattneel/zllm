// attn_split's sparse prelude (Qwen3.8-Flash-Next): the cells
// qsa_select chose. A query row seeing at most `width` cells attends to all;
// otherwise to its tail (from its last complete block on) and the cells
// sel takes from each block's start.

struct Qsa {
    heads: u32,
    dim: u32,
    ratio: u32,
    width: u32,
    stride: u32,
    n_rot: u32,
    eps: f32,
    row: u32,
}

@group(0) @binding(6) var<storage, read> sel: array<u32>;
@group(0) @binding(7) var<uniform> qsa: Qsa;

fn included(row: u32, key: u32) -> bool {
    let visible = step.pos0 + row + 1u;
    if (visible <= qsa.width) {
        return true;
    }
    let tail = visible / qsa.ratio * qsa.ratio;
    if (key >= tail) {
        return true;
    }
    return (key % qsa.ratio) < sel[row * qsa.stride + key / qsa.ratio];
}
