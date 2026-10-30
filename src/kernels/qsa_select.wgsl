// The sparse attention's indexer, part 4: which cells step row t attends
// to, as sel[t · stride + b] = cells taken from block b's start (0,
// `ratio`, or fewer for the last). A row seeing at most `width` cells takes
// all of them (the attention needs no sel). Otherwise it takes its tail (the
// L cells after its last complete block) and then whole blocks in order of
// score, highest first, ties to the lower block, until `width` cells: the
// top K = ⌈(width − L) / ratio⌉ blocks, the K-th cut to what is left.
//
// Scores are sums of non-negative terms, so their bits order as their values.
// The K-th largest score T is found bit by bit (32 counting passes), then the
// blocks scoring above T are taken, and of those scoring T the lowest-index
// K − count(above T), the last of which is the K-th. One workgroup per row.

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

struct Step {
    m: u32,
    pos0: u32,
    u: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Qsa;
@group(0) @binding(1) var<storage, read> scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> sel: array<u32>;
@group(0) @binding(3) var<uniform> step: Step;

const WG: u32 = 256u;

var<workgroup> red: array<u32, WG>;

// Blocks of the row with score bits ≥ at, summed over the workgroup.
fn count_at_least(base: u32, nb: u32, at: u32, lid: u32) -> u32 {
    var c = 0u;
    for (var b = lid; b < nb; b += WG) {
        c += select(0u, 1u, bitcast<u32>(scores[base + b]) >= at);
    }
    red[lid] = c;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
        if (lid < s) {
            red[lid] += red[lid + s];
        }
        workgroupBarrier();
    }
    return workgroupUniformLoad(&red[0]);
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid3: vec3u) {
    let lid = lid3.x;
    let t = wg.y;
    let r = p.ratio;
    let visible = step.pos0 + t + 1u;
    if (visible <= p.width) {
        return;
    }
    let nb = visible / r;
    let cells = p.width - (visible - nb * r);
    let k = (cells + r - 1u) / r;
    let last = cells - (k - 1u) * r;
    let base = t * p.stride;

    var at = 0u;
    for (var bit = 31i; bit >= 0i; bit--) {
        let candidate = at | (1u << u32(bit));
        if (count_at_least(base, nb, candidate, lid) >= k) {
            at = candidate;
        }
    }
    let above = count_at_least(base, nb, at + 1u, lid);
    let ties = k - above;

    // Tie ranks by block index: each thread takes a contiguous chunk, after
    // the ties in the chunks before it (an exclusive scan).
    let chunk = (nb + WG - 1u) / WG;
    let lo = min(lid * chunk, nb);
    let hi = min(lo + chunk, nb);
    var mine = 0u;
    for (var b = lo; b < hi; b++) {
        mine += select(0u, 1u, bitcast<u32>(scores[base + b]) == at);
    }
    red[lid] = mine;
    workgroupBarrier();
    for (var s = 1u; s < WG; s *= 2u) {
        var add = 0u;
        if (lid >= s) {
            add = red[lid - s];
        }
        workgroupBarrier();
        red[lid] += add;
        workgroupBarrier();
    }
    var rank = red[lid] - mine;
    for (var b = lo; b < hi; b++) {
        let bits = bitcast<u32>(scores[base + b]);
        var take = 0u;
        if (bits > at) {
            take = r;
        } else if (bits == at) {
            if (rank < ties) {
                take = select(r, last, rank + 1u == ties);
            }
            rank += 1u;
        }
        sel[base + b] = take;
    }
}
