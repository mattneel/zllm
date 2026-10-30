// lo[t][j] = silu(down[t · stride + j] / hc), j < rank: the hyper-connection
// mixer's low-rank activation, between its down and up projections. The
// down projection's rows are `stride` wide: the stream weights' (inject)
// columns follow its rank columns, from the same multiply.

struct Params {
    count: u32,
    hc: u32,
    rank: u32,
    stride: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> down: array<f32>;
@group(0) @binding(2) var<storage, read_write> lo: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= p.count) {
        return;
    }
    let t = i / p.rank;
    let v = down[t * p.stride + i % p.rank] / f32(p.hc);
    lo[i] = v / (1.0 + exp(-v));
}
