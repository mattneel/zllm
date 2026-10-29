#!/usr/bin/env bash
# Run zllm on the Windows D3D12 host from WSL: build the Windows executable,
# copy it next to the staged Dawn and DXC DLLs (scripts/fixtures.sh dawn),
# and run it there with the given arguments.
#
#   scripts/d3d12.sh gpu-info
#   scripts/d3d12.sh gpu-check
#
# Paths in the arguments must be Windows paths (see `wslpath -w`).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="${ZLLM_FIXTURES:-$root/.fixtures}"
stage_file="$dir/dawn/win_stage"
if [[ ! -f "$stage_file" ]]; then
	echo "no D3D12 host staged; run scripts/fixtures.sh dawn under WSL" >&2
	exit 1
fi
stage="$(wslpath -u "$(cat "$stage_file")")"

(cd "$root" && zig build -Dtarget=x86_64-windows-gnu -Doptimize="${ZLLM_OPTIMIZE:-ReleaseSafe}" -p .zig-out-windows)
cp "$root/.zig-out-windows/bin/zllm.exe" "$stage/zllm.exe"
cd "$stage"
exec ./zllm.exe "$@"
