// Appends the step's K and V rows to the cache: row t of k and v (width
// values each) lands at cache position pos0 + t, for t < m. The cache stores
// f16 pairs (pack2x16float) when f16 != 0, else f32 bits. One thread per
// pair of values; width is even.

struct Params {
    width: u32,
    f16: u32,
    _pad0: u32,
    _pad1: u32,
}

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> kc: array<u32>;
@group(0) @binding(4) var<storage, read_write> vc: array<u32>;
@group(0) @binding(5) var<uniform> step: Step;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) n: vec3u) {
    let i = id.x + id.y * n.x * 256u;
    let half = p.width / 2u;
    if (i >= step.m * half) {
        return;
    }
    let t = i / half;
    let e = i % half;
    let src = t * p.width + 2u * e;
    let pair = (step.pos0 + t) * half + e;
    if (p.f16 != 0u) {
        kc[pair] = pack2x16float(vec2f(k[src], k[src + 1u]));
        vc[pair] = pack2x16float(vec2f(v[src], v[src + 1u]));
    } else {
        kc[2u * pair] = bitcast<u32>(k[src]);
        kc[2u * pair + 1u] = bitcast<u32>(k[src + 1u]);
        vc[2u * pair] = bitcast<u32>(v[src]);
        vc[2u * pair + 1u] = bitcast<u32>(v[src + 1u]);
    }
}
