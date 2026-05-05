#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models"
mkdir -p "$MODEL_DIR"

WESPEAKER_URL="https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34/resolve/main/voxceleb_resnet34.onnx?download=true"
SILERO_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

echo "Downloading WeSpeaker ResNet34 (VoxCeleb)..."
if [ -f "$MODEL_DIR/wespeaker_resnet34.onnx" ]; then
    echo "  Already exists, skipping."
else
    curl -L --progress-bar -o "$MODEL_DIR/wespeaker_resnet34.onnx" "$WESPEAKER_URL"
    echo "  Done: $(du -h "$MODEL_DIR/wespeaker_resnet34.onnx" | cut -f1)"
fi

echo "Downloading Silero VAD v5..."
if [ -f "$MODEL_DIR/silero_vad.onnx" ]; then
    echo "  Already exists, skipping."
else
    curl -L --progress-bar -o "$MODEL_DIR/silero_vad.onnx" "$SILERO_URL"
    echo "  Done: $(du -h "$MODEL_DIR/silero_vad.onnx" | cut -f1)"
fi

echo ""
echo "Models downloaded to $MODEL_DIR/"
ls -lh "$MODEL_DIR/"*.onnx
