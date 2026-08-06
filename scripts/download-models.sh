#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"
mkdir -p "$MODEL_DIR"

WESPEAKER_URL="https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/main/voxceleb_resnet34.onnx?download=true"
# Commit-pinned (same blob as src/models/manifest.toml silero_vad entry).
SILERO_URL="https://github.com/snakers4/silero-vad/raw/bfdc0193023f121ea5b3cc7b176dbed570a68a59/src/silero_vad/data/silero_vad.onnx"
POWERSET_URL="https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx"
CAM_PP_URL="https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus/resolve/main/voxceleb_CAM%2B%2B.onnx?download=true"

download() {
    local name="$1" url="$2"
    echo "Downloading $name..."
    if [ -f "$MODEL_DIR/$name" ]; then
        echo "  Already exists, skipping."
    else
        curl -L --progress-bar -o "$MODEL_DIR/$name" "$url"
        echo "  Done: $(du -h "$MODEL_DIR/$name" | cut -f1)"
    fi
}

download "wespeaker_resnet34.onnx" "$WESPEAKER_URL"

# Silero VAD v6-generation weights (URL tracks upstream master; pin verified by
# ModelRegistry SHA-256). Prefer a release-asset mirror once published — see
# scripts/mirror-silero-vad.md.
download "silero_vad.onnx" "$SILERO_URL"

# Powerset segmentation (pyannote/segmentation-3.0 sherpa-onnx export).
download "powerset_fp32.onnx" "$POWERSET_URL"

# CAM++ embedder (used by parity and runtime tests).
download "cam_pp_fp32.onnx" "$CAM_PP_URL"

echo ""
echo "Models downloaded to $MODEL_DIR/"
ls -lh "$MODEL_DIR/"*.onnx
