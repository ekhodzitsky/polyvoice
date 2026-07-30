#!/usr/bin/env bash
set -euo pipefail

# Orchestrate INT8 static quantization of the three v1.0 models.
#
# Reads FP32 models from models/, calibrates with VoxConverse-dev random
# 500-sample (seed 42), writes models/int8/<name>_int8.onnx for each.
#
# Idempotent: if INT8 file exists and is at least 1 byte smaller than FP32,
# the script skips that model.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"
INT8_DIR="${MODELS_DIR}/int8"
CALIB_DIR="${ROOT_DIR}/data/voxconverse-dev/audio"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv-m5/bin/python}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"

mkdir -p "$INT8_DIR"

if [ ! -d "$CALIB_DIR" ]; then
    echo "ERROR: calibration audio missing at $CALIB_DIR"
    echo "Run scripts/download-voxconverse-dev.sh first."
    exit 1
fi

WAV_COUNT=$(find "$CALIB_DIR" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
if [ "$WAV_COUNT" -lt 50 ]; then
    echo "ERROR: only ${WAV_COUNT} WAVs in $CALIB_DIR — calibration unstable"
    exit 1
fi
echo "Calibration source: ${WAV_COUNT} WAVs in $CALIB_DIR"
echo ""

quantize_one() {
    local name="$1"
    local fp32="$2"
    local int8="$3"
    local shape="$4"
    local exclude="${5:-}"

    if [ ! -f "$fp32" ]; then
        echo "[$name] SKIP: $fp32 not present"
        return 0
    fi
    if [ -f "$int8" ]; then
        local fp32_kb int8_kb
        fp32_kb=$(stat -f%z "$fp32" 2>/dev/null || stat -c%s "$fp32")
        int8_kb=$(stat -f%z "$int8" 2>/dev/null || stat -c%s "$int8")
        if [ "$int8_kb" -lt "$fp32_kb" ]; then
            echo "[$name] CACHED: $int8 ($int8_kb bytes vs $fp32_kb)"
            return 0
        fi
    fi
    echo "[$name] Quantizing..."
    local args=(
        --fp32 "$fp32"
        --int8 "$int8"
        --calib "$CALIB_DIR"
        --input-shape "$shape"
        --num-samples "$NUM_SAMPLES"
        --seed "$SEED"
    )
    if [ -n "$exclude" ]; then
        args+=(--exclude-nodes "$exclude")
    fi
    "$PYTHON" "$SCRIPT_DIR/quantize_models.py" "${args[@]}"
}

# Powerset segmenter: 10s window @ 16 kHz = 160_000 samples
quantize_one "powerset" \
    "$MODELS_DIR/powerset_fp32.onnx" \
    "$INT8_DIR/powerset_int8.onnx" \
    "1,1,160000" \
    ""

# CAM++: WeSpeaker fbank input shape is [B, T, 80] (T frames × 80 mel bins).
# 300 frames @ 10ms hop ≈ 3 seconds.
quantize_one "cam_pp" \
    "$MODELS_DIR/cam_pp_fp32.onnx" \
    "$INT8_DIR/cam_pp_int8.onnx" \
    "1,300,80" \
    ""

# WeSpeaker ResNet34: same fbank pipeline, shape [B, T, 80].
quantize_one "resnet34" \
    "$MODELS_DIR/wespeaker_resnet34.onnx" \
    "$INT8_DIR/resnet34_int8.onnx" \
    "1,300,80" \
    ""

echo ""
echo "=== Summary ==="
ls -lh "$INT8_DIR"/*.onnx 2>/dev/null || echo "(no INT8 outputs yet)"
