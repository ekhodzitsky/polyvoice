#!/usr/bin/env bash
# Download production INT8 ONNX models into models/ (and silero for --legacy).
# Prefer: cargo run --features cli -- download-models --profile balanced
# This script hashes every blob (SHA-256 from src/models/manifest.toml).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"
mkdir -p "$MODEL_DIR" "$MODEL_DIR/int8"

# Same release assets as src/models/manifest.toml (models-int8-v2).
POWERSET_INT8_URL="https://github.com/ekhodzitsky/polyvoice/releases/download/models-int8-v2/powerset_int8.onnx"
POWERSET_INT8_SHA256="175896d26f639933cd86906d2dd3e6796eddb23c1f719925a3949052da76183b"
RESNET34_INT8_URL="https://github.com/ekhodzitsky/polyvoice/releases/download/models-int8-v2/resnet34_int8.onnx"
RESNET34_INT8_SHA256="24b58559fefb2af624a5d371c43ebae891a9a8ca363b2f9e7c31fd8e440a36b3"
# Commit-pinned (same blob as src/models/manifest.toml silero_vad entry).
SILERO_URL="https://github.com/snakers4/silero-vad/raw/bfdc0193023f121ea5b3cc7b176dbed570a68a59/src/silero_vad/data/silero_vad.onnx"
SILERO_SHA256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

sha256_of() {
    local f="$1"
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$f" | awk '{print $1}'
    else
        sha256sum "$f" | awk '{print $1}'
    fi
}

download() {
    local dest="$1" url="$2" expected="$3"
    local name got
    name="$(basename "$dest")"
    echo "Downloading $name..."
    if [ -f "$dest" ]; then
        got="$(sha256_of "$dest")"
        if [ "$got" = "$expected" ]; then
            echo "  Already present and hash-ok."
            return 0
        fi
        echo "  Hash mismatch (got $got), re-downloading."
        rm -f "$dest"
    fi
    curl -fsSL --retry 5 --retry-delay 2 -o "$dest" "$url"
    got="$(sha256_of "$dest")"
    if [ "$got" != "$expected" ]; then
        echo "  FAIL: SHA-256 mismatch for $name" >&2
        echo "    expected $expected" >&2
        echo "    got      $got" >&2
        rm -f "$dest"
        exit 1
    fi
    echo "  Done: $(du -h "$dest" | cut -f1) (hash-ok)"
}

# Production pair (profile balanced/mobile/fast).
download "$MODEL_DIR/powerset_int8.onnx" "$POWERSET_INT8_URL" "$POWERSET_INT8_SHA256"
download "$MODEL_DIR/resnet34_int8.onnx" "$RESNET34_INT8_URL" "$RESNET34_INT8_SHA256"
# Also place under models/int8 for quant tooling that expects that layout.
download "$MODEL_DIR/int8/powerset_int8.onnx" "$POWERSET_INT8_URL" "$POWERSET_INT8_SHA256"
download "$MODEL_DIR/int8/resnet34_int8.onnx" "$RESNET34_INT8_URL" "$RESNET34_INT8_SHA256"

# Silero VAD (legacy / BYO path only — not part of pipeline v2).
download "$MODEL_DIR/silero_vad.onnx" "$SILERO_URL" "$SILERO_SHA256"

echo ""
echo "Production INT8 models in $MODEL_DIR/ (~8.4 MB pair + silero)."
ls -lh "$MODEL_DIR"/*.onnx 2>/dev/null || true
