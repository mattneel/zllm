"""Compare `zllm gguf-index` with llama.cpp's gguf-dump on the pinned GGUF.

gguf-dump (`gguf_dump.py --json --json-array` from the pinned checkout) is the
reference for every metadata key in order, its type and its full value, and
for every tensor's name, type, shape and order. Array values are compared by
re-encoding gguf-dump's decoded elements and hashing them against the SHA-256
zllm computes over the raw encoded array.

gguf-dump does not report the data layout, so the data section offset,
alignment, and each tensor's absolute offset and byte size are compared with
the same pinned gguf-py's GGUFReader.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys

import numpy as np

from common import Failures, fixture_paths, import_pinned_gguf, parse_args

# gguf-dump type names -> zllm type names, and struct formats for scalars.
TYPES = {
    "UINT8": ("u8", "<B"),
    "INT8": ("i8", "<b"),
    "UINT16": ("u16", "<H"),
    "INT16": ("i16", "<h"),
    "UINT32": ("u32", "<I"),
    "INT32": ("i32", "<i"),
    "FLOAT32": ("f32", "<f"),
    "BOOL": ("bool", "<?"),
    "STRING": ("string", None),
    "ARRAY": ("array", None),
    "UINT64": ("u64", "<Q"),
    "INT64": ("i64", "<q"),
    "FLOAT64": ("f64", "<d"),
}


def encode_array(elem_type: str, values: list) -> bytes:
    """Encode decoded array elements back to GGUF bytes."""
    if elem_type == "STRING":
        out = bytearray()
        for s in values:
            b = s.encode("utf-8")
            out += struct.pack("<Q", len(b)) + b
        return bytes(out)
    fmt = TYPES[elem_type][1]
    return b"".join(struct.pack(fmt, v) for v in values)


def run_gguf_dump(paths) -> dict:
    script = paths.gguf_py / "gguf" / "scripts" / "gguf_dump.py"
    env = {**os.environ, "PYTHONPATH": str(paths.gguf_py)}
    out = subprocess.run(
        [sys.executable, str(script), "--json", "--json-array", str(paths.gguf)],
        check=True,
        capture_output=True,
        env=env,
    ).stdout
    return json.loads(out)


def main() -> int:
    args = parse_args(__doc__)
    paths = fixture_paths(args.fixtures)
    gguf = import_pinned_gguf(paths)

    zllm = json.loads(subprocess.run([args.zllm, "gguf-index", str(paths.gguf)], check=True, capture_output=True).stdout)
    dump = run_gguf_dump(paths)
    reader = gguf.GGUFReader(paths.gguf)
    fail = Failures()

    # gguf-dump reports the three header fields as pseudo-metadata.
    dump_meta = {k: v for k, v in dump["metadata"].items() if not k.startswith("GGUF.")}
    fail.expect_eq("version", zllm["version"], dump["metadata"]["GGUF.version"]["value"])
    fail.expect_eq("tensor count", len(zllm["tensors"]), dump["metadata"]["GGUF.tensor_count"]["value"])
    fail.expect_eq("metadata count", len(zllm["metadata"]), dump["metadata"]["GGUF.kv_count"]["value"])
    fail.expect_eq("metadata keys in order", list(zllm["metadata"]), list(dump_meta))

    n_array_elems = 0
    for key, ref in dump_meta.items():
        got = zllm["metadata"].get(key)
        if got is None:
            continue
        fail.expect_eq(f"{key}.type", got["type"], TYPES[ref["type"]][0])
        if ref["type"] == "ARRAY":
            (elem_type,) = ref["array_types"]
            arr = got["value"]
            n_array_elems += len(ref["value"])
            fail.expect_eq(f"{key}.elem_type", arr["elem_type"], TYPES[elem_type][0])
            fail.expect_eq(f"{key}.len", arr["len"], len(ref["value"]))
            want = hashlib.sha256(encode_array(elem_type, ref["value"])).hexdigest()
            fail.expect_eq(f"{key}.sha256", arr["sha256"], want)
        elif ref["type"] == "FLOAT32":
            fail.expect(f"{key}.value", np.float32(got["value"]) == np.float32(ref["value"]), f"{got['value']} != {ref['value']}")
        else:
            fail.expect_eq(f"{key}.value", got["value"], ref["value"])

    dump_tensors = sorted(dump["tensors"].items(), key=lambda kv: kv[1]["index"])
    fail.expect_eq("tensor names in order", [t["name"] for t in zllm["tensors"]], [name for name, _ in dump_tensors])
    for z, (name, ref) in zip(zllm["tensors"], dump_tensors, strict=False):
        fail.expect_eq(f"{name}.type", z["type"], ref["type"].lower())
        fail.expect_eq(f"{name}.shape", z["shape"], ref["shape"])

    # Data layout, which gguf-dump does not print.
    fail.expect_eq("alignment", zllm["alignment"], reader.alignment)
    fail.expect_eq("data_offset", zllm["data_offset"], reader.data_offset)
    for z, r in zip(zllm["tensors"], reader.tensors, strict=False):
        fail.expect_eq(f"{r.name}.offset", z["offset"], int(r.data_offset))
        fail.expect_eq(f"{r.name}.bytes", z["bytes"], int(r.n_bytes))

    print(
        f"gguf: {paths.gguf.name}: {len(dump_meta)} metadata keys ({n_array_elems} array elements), "
        f"{len(dump_tensors)} tensors vs gguf-dump; data layout vs GGUFReader"
    )
    return fail.report()


if __name__ == "__main__":
    sys.exit(main())
