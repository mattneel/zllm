#!/usr/bin/env bash
# Fetch and verify the pinned M0 fixtures into .fixtures/ (gitignored).
#
#   scripts/fixtures.sh          everything below
#   scripts/fixtures.sh vocab    GGUF header (one range read) + Hugging Face tokenizer files
#   scripts/fixtures.sh model    the full 0.8B BF16 GGUF (1.5 GB)
#   scripts/fixtures.sh llama    llama.cpp at the pinned commit, built for CPU
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

case "${1:-all}" in
vocab) vocab ;;
model) model ;;
llama) llama ;;
all)
	vocab
	model
	llama
	;;
*)
	log "usage: $0 [all|vocab|model|llama]"
	exit 2
	;;
esac
