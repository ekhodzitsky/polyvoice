#!/usr/bin/env bash
# Download production INT8 ONNX models into models/ (and silero for --legacy).
# Prefer: cargo run --features cli -- download-models --profile balanced
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"
mkdir -p "$MODEL_DIR" "$MODEL_DIR/int8"

# Same release assets as src/models/manifest.toml (models-int8-v2).
POWERSET_INT8_URL="https://github.com/ekhodzitsky/polyvoice/releases/download/models-int8-v2/powerset_int8.onnx"
RESNET34_INT8_URL="https://github.com/ekhodzitsky/polyvoice/releases/download/models-int8-v2/resnet34_int8.onnx"
# Commit-pinned (same blob as src/models/manifest.toml silero_vad entry).
SILERO_URL="https://github.com/snakers4/silero-vad/raw/bfdc0193023f121ea5b3cc7b176dbed570a68a59/src/silero_vad/data/silero_vad.onnx"

download() {
    local dest="$1" url="$2"
    local name
    name="$(basename "$dest")"
    echo "Downloading $name..."
    if [ -f "$dest" ]; then
        echo "  Already exists, skipping."
    else
        curl -L --progress-bar -o "$dest" "$url"
        echo "  Done: $(du -h "$dest" | cut -f1)"
    fi
}

# Production pair (profile balanced/mobile/fast).
download "$MODEL_DIR/powerset_int8.onnx" "$POWERSET_INT8_URL"
download "$MODEL_DIR/resnet34_int8.onnx" "$RESNET34_INT8_URL"
# Also place under models/int8 for quant tooling that expects that layout.
download "$MODEL_DIR/int8/powerset_int8.onnx" "$POWERSET_INT8_URL"
download "$MODEL_DIR/int8/resnet34_int8.onnx" "$RESNET34_INT8_URL"

# Silero VAD (legacy / BYO path only — not part of pipeline v2).
download "$MODEL_DIR/silero_vad.onnx" "$SILERO_URL"

echo ""
echo "Production INT8 models in $MODEL_DIR/ (~8.4 MB pair + silero)."
ls -lh "$MODEL_DIR"/*.onnx 2>/dev/null || true
