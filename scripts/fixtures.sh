#!/usr/bin/env bash
# Fetch and verify the pinned M0 fixtures into .fixtures/ (gitignored).
#
#   scripts/fixtures.sh          everything below
#   scripts/fixtures.sh vocab    GGUF header (one range read) + Hugging Face tokenizer files
#   scripts/fixtures.sh model    the full 0.8B BF16 GGUF (1.5 GB)
#   scripts/fixtures.sh llama    llama.cpp at the pinned commit, built for CPU
#   scripts/fixtures.sh dawn     Dawn (webgpu.h) as libwebgpu_dawn.so; under WSL with
#                                Visual Studio, also webgpu_dawn.dll staged with DXC
#
# Every download is checked against a SHA-256 pinned here; llama.cpp is checked
# out by commit. docs/src/validation.md mirrors these pins.
# CMAKE and CXX override the build tools.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="${ZLLM_FIXTURES:-$root/.fixtures}"

gguf_repo="ggml-org/Qwen3.5-0.8B-GGUF"
gguf_rev="807dfd9a6fadbf703207aac3bfb0b2b3040d077c"
gguf_file="Qwen3.5-0.8B-BF16.gguf"
gguf_sha256="9a7bed4041b7975e0f71fa34670d1e9025213bc92905ac0db75d36c4fa3fa623"

hf_repo="Qwen/Qwen3.5-0.8B"
hf_rev="2fc06364715b967f1860aea9cf38778875588b17"
hf_files=(
	"tokenizer.json 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
	"tokenizer_config.json 49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
	"config.json b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
	"chat_template.jinja 273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
)

llama_url="https://github.com/ggml-org/llama.cpp.git"
llama_rev="e09a78a25dec3f42175ad59ecc540a54bed0b9b2"
# The GGUF header (metadata + tensor index, everything before tensor data):
# all the tokenizer needs, fetched with one HTTP range read.
header_file="Qwen3.5-0.8B-BF16.header.gguf"
header_bytes=10962464
header_sha256="4c21351af5833ba7aec5063df9400a087ca60884f4052e440ba1b80b265fd4a5"

# Dawn prebuilt releases; third_party/dawn/include holds the matching webgpu.h.
dawn_tag="v20261027.184652"
dawn_rev="106704a9ac3fc21b1580ea85aea302ac17d416f1"
dawn_linux_sha256="24fb115dc55b1e5acc169e3d2558eeda5f1b0d149af4a46f10602b900103cbd3"
dawn_windows_sha256="2b082c284f078b9f91a91a6b2cf805fa42c18f60b8dcaab70476a5e5e24b74ee"
# Windows directory for the D3D12 test host: Windows will not load DLLs from
# a \\wsl.localhost path, so the executable and its DLLs live on a local drive.
win_stage="${ZLLM_WIN_STAGE:-}"
cmake="${CMAKE:-cmake}"
cxx="${CXX:-c++}"

log() { printf '[fixtures] %s\n' "$*" >&2; }

verify() {
	local path="$1" want="$2" got
	got="$(sha256sum "$path" | cut -d' ' -f1)"
	if [[ "$got" != "$want" ]]; then
		log "SHA-256 mismatch for $path"
		log "  want $want"
		log "  got  $got"
		return 1
	fi
	log "ok $(basename "$path") $got"
}

fetch() {
	local url="$1" path="$2" want="$3"
	if [[ -f "$path" ]] && verify "$path" "$want"; then
		return 0
	fi
	log "downloading $url"
	mkdir -p "$(dirname "$path")"
	curl --fail --location --retry 3 --continue-at - --output "$path.part" "$url"
	mv "$path.part" "$path"
	verify "$path" "$want"
}

# Small fixtures: tokenizer parity (CI runs this mode).
vocab() {
	local path="$dir/models/$header_file"
	if [[ ! -f "$path" ]] || ! verify "$path" "$header_sha256"; then
		log "range-reading GGUF header ($header_bytes bytes)"
		mkdir -p "$dir/models"
		curl --fail --location --retry 3 --range "0-$((header_bytes - 1))" --output "$path.part" \
			"https://huggingface.co/$gguf_repo/resolve/$gguf_rev/$gguf_file"
		mv "$path.part" "$path"
		verify "$path" "$header_sha256"
	fi
	local entry name sha
	for entry in "${hf_files[@]}"; do
		read -r name sha <<<"$entry"
		fetch "https://huggingface.co/$hf_repo/resolve/$hf_rev/$name" \
			"$dir/hf/Qwen3.5-0.8B/$name" "$sha"
	done
}

# The full model: GGUF index and forward-pass parity.
model() {
	fetch "https://huggingface.co/$gguf_repo/resolve/$gguf_rev/$gguf_file" \
		"$dir/models/$gguf_file" "$gguf_sha256"
}

llama() {
	local src="$dir/llama.cpp"
	if [[ ! -d "$src/.git" ]]; then
		log "cloning llama.cpp"
		git init --quiet "$src"
		git -C "$src" remote add origin "$llama_url"
	fi
	if [[ "$(git -C "$src" rev-parse HEAD 2>/dev/null || true)" != "$llama_rev" ]]; then
		git -C "$src" fetch --depth 1 origin "$llama_rev"
		git -C "$src" checkout --quiet --detach FETCH_HEAD
	fi
	if [[ -n "$(git -C "$src" status --porcelain)" ]]; then
		log "llama.cpp checkout is modified; refusing to build a non-pinned tree"
		return 1
	fi
	log "llama.cpp at $(git -C "$src" rev-parse HEAD)"
	"$cmake" -S "$src" -B "$src/build" -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DGGML_NATIVE=OFF \
		-DLLAMA_CURL=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_SERVER=OFF
	"$cmake" --build "$src/build" --target llama ggml

	log "building llama-dump"
	"$cxx" -std=c++17 -O2 "$root/tools/llama-dump/dump.cpp" \
		-I "$src/include" -I "$src/ggml/include" \
		-L "$src/build/bin" -Wl,-rpath,"$src/build/bin" -lllama -lggml -lggml-base \
		-o "$src/build/bin/llama-dump"
}

dawn() {
	local base="https://github.com/google/dawn/releases/download/$dawn_tag/Dawn-$dawn_rev"
	local linux="$dir/dawn/Dawn-$dawn_rev-ubuntu-latest-Release"
	fetch "$base-ubuntu-latest-Release.tar.gz" "$dir/dawn/linux.tar.gz" "$dawn_linux_sha256"
	if [[ ! -d "$linux" ]]; then tar -xzf "$dir/dawn/linux.tar.gz" -C "$dir/dawn"; fi
	# The release ships a static archive; wrap it as a shared library so zllm
	# loads the C API at run time and never links Dawn's C++.
	log "building libwebgpu_dawn.so"
	"$cxx" -shared -o "$dir/dawn/libwebgpu_dawn.so" \
		-Wl,--whole-archive "$linux/lib64/libwebgpu_dawn.a" -Wl,--no-whole-archive -lpthread -ldl

	if ! command -v cmd.exe >/dev/null; then
		log "no Windows interop; skipping the D3D12 host"
		return 0
	fi
	local vs
	vs="$(find "/mnt/c/Program Files/Microsoft Visual Studio" "/mnt/c/Program Files (x86)/Microsoft Visual Studio" \
		-maxdepth 6 -name vcvars64.bat 2>/dev/null | sort -r | head -n1 || true)"
	local dxc
	dxc="$(find "/mnt/c/Program Files (x86)/Windows Kits/10/bin" -maxdepth 3 -path '*/x64/dxcompiler.dll' 2>/dev/null | sort -V | tail -n1 || true)"
	if [[ -z "$vs" || -z "$dxc" ]]; then
		log "Visual Studio (vcvars64.bat) or Windows SDK DXC not found; skipping the D3D12 host"
		return 0
	fi

	local windows="$dir/dawn/Dawn-$dawn_rev-windows-latest-Release"
	fetch "$base-windows-latest-Release.tar.gz" "$dir/dawn/windows.tar.gz" "$dawn_windows_sha256"
	if [[ ! -d "$windows" ]]; then tar -xzf "$dir/dawn/windows.tar.gz" -C "$dir/dawn"; fi

	# Export exactly the C API declared in the vendored header.
	{
		echo "LIBRARY webgpu_dawn"
		echo "EXPORTS"
		grep -oE 'WGPU_EXPORT [^(]*\bwgpu[A-Za-z0-9]+\(' "$root/third_party/dawn/include/dawn/webgpu.h" |
			grep -oE 'wgpu[A-Za-z0-9]+' | sort -u | sed 's/^/    /'
	} >"$windows/webgpu_dawn.def"
	# Not "link.bat": in its own directory that would shadow link.exe.
	cat >"$windows/build_dll.bat" <<-EOF
		@echo off
		call "$(wslpath -w "$vs")" >nul
		if errorlevel 1 exit /b 1
		pushd "$(wslpath -w "$windows")" || exit /b 1
		link.exe /nologo /DLL /MACHINE:X64 /OUT:webgpu_dawn.dll /DEF:webgpu_dawn.def lib\\webgpu_dawn.lib d3d12.lib dxgi.lib dxguid.lib d3dcompiler.lib user32.lib kernel32.lib ole32.lib onecore.lib || exit /b 1
	EOF
	log "linking webgpu_dawn.dll with $(basename "$(dirname "$(dirname "$(dirname "$(dirname "$vs")")")")")"
	(cd /mnt/c && cmd.exe /c "$(wslpath -w "$windows/build_dll.bat")") | tr -d '\r'
	[[ -f "$windows/webgpu_dawn.dll" ]]

	if [[ -z "$win_stage" ]]; then
		win_stage="$(cd /mnt/c && cmd.exe /c 'echo %LOCALAPPDATA%' | tr -d '\r')\\zllm\\gpu"
	fi
	local stage
	stage="$(wslpath -u "$win_stage")"
	mkdir -p "$stage"
	cp "$windows/webgpu_dawn.dll" "$dxc" "$(dirname "$dxc")/dxil.dll" "$stage/"
	printf '%s\n' "$win_stage" >"$dir/dawn/win_stage"
	log "D3D12 host staged in $win_stage"
}

case "${1:-all}" in
vocab) vocab ;;
model) model ;;
llama) llama ;;
dawn) dawn ;;
all)
	vocab
	model
	llama
	dawn
	;;
*)
	log "usage: $0 [all|vocab|model|llama|dawn]"
	exit 2
	;;
esac
