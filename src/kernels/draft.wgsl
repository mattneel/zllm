// The speculative draft: the first index of the maximum of one row of
// logits (the MTP block's), written to out[p.slot], where the next step
// embeds it. One workgroup: each thread scans a strided share of the row,
// then a tree reduction keeps the larger value, and the lower index on ties.

struct Params {
    vocab: u32,
    // Where the row starts in `logits`, in floats.
    row: u32,
    slot: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

const WG: u32 = 256u;

var<workgroup> best_v: array<f32, WG>;
var<workgroup> best_i: array<u32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid3: vec3u) {
    let lid = lid3.x;
    // `at` = none: nothing seen yet (v is then unused).
    var v = 0.0;
    var at = 0xffffffffu;
    for (var i = lid; i < p.vocab; i += WG) {
        let x = logits[p.row + i];
        // Strided indices rise, so a strict > keeps the first maximum.
        if (at == 0xffffffffu || x > v) {
            v = x;
            at = i;
        }
    }
    best_v[lid] = v;
    best_i[lid] = at;
    workgroupBarrier();
    for (var half = WG / 2u; half > 0u; half /= 2u) {
        if (lid < half) {
            let ov = best_v[lid + half];
            let oi = best_i[lid + half];
            let none = best_i[lid] == 0xffffffffu;
            if (oi != 0xffffffffu && (none || ov > best_v[lid] || (ov == best_v[lid] && oi < best_i[lid]))) {
                best_v[lid] = ov;
                best_i[lid] = oi;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        out[p.slot] = best_i[0];
    }
}
