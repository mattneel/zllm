// mask[tokens[i]] = 1 for i < n_prompt: the prompt-token set that the
// repetition penalty reads. Shares the sampler's per-request parameters.

struct Params {
    vocab: u32,
    n_bias: u32,
    n_stop: u32,
    greedy: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    repetition: f32,
    presence: f32,
    frequency: f32,
    n_prompt: u32,
    stops: array<vec4u, 4>,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> mask: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) n: vec3u) {
    let i = id.x + id.y * n.x * 256u;
    if (i < p.n_prompt) {
        mask[tokens[i]] = 1u;
    }
}
