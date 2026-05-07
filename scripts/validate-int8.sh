#!/usr/bin/env bash
set -uo pipefail

# Validate the three M5 INT8 artifacts against §9.4 acceptance gates.
# Exit non-zero on first failure; print per-model report path on success.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"
INT8_DIR="${MODELS_DIR}/int8"
DEV_AUDIO="${ROOT_DIR}/data/voxconverse-dev/audio"
DEV_RTTM="${ROOT_DIR}/data/voxconverse-dev/rttm"
VC_AUDIO="${ROOT_DIR}/data/voxceleb1-subset/wav"
VC_TRIALS="${ROOT_DIR}/data/voxceleb1-subset/lists/veri_test.txt"
REPORT_DIR="${ROOT_DIR}/docs/calibration"
PYTHON="${PYTHON:-${ROOT_DIR}/.venv-m5/bin/python}"

mkdir -p "$REPORT_DIR"

DATE="$(date +%Y-%m-%d)"
REPORT_AGGREGATE="${REPORT_DIR}/${DATE}-int8-validation.md"
ALL_OK=0

validate_one() {
    local kind="$1"
    local fp32="$2"
    local int8="$3"
    local extra=("${@:4}")

    if [ ! -f "$int8" ]; then
        echo "[$kind] SKIP: $int8 missing"
        return 0
    fi
    local report="${REPORT_DIR}/${DATE}-$(basename "$int8" .onnx)-validation.md"
    echo "[$kind] Validating $int8 -> $report"
    if "$PYTHON" "$SCRIPT_DIR/validate_int8.py" \
        --fp32 "$fp32" \
        --int8 "$int8" \
        --kind "$kind" \
        --report "$report" \
        "${extra[@]}"; then
        echo "[$kind] PASS"
    else
        echo "[$kind] FAIL"
        ALL_OK=1
    fi
}

validate_one "powerset" \
    "$MODELS_DIR/powerset_fp32.onnx" \
    "$INT8_DIR/powerset_int8.onnx" \
    --hold-out "$DEV_AUDIO" \
    --hold-out-rttm "$DEV_RTTM"

validate_one "embedder" \
    "$MODELS_DIR/cam_pp_fp32.onnx" \
    "$INT8_DIR/cam_pp_int8.onnx" \
    --hold-out "$DEV_AUDIO" \
    --voxceleb-audio "$VC_AUDIO" \
    --voxceleb-trials "$VC_TRIALS" \
    --embed-input-shape "1,300,80"

validate_one "embedder" \
    "$MODELS_DIR/wespeaker_resnet34.onnx" \
    "$INT8_DIR/resnet34_int8.onnx" \
    --hold-out "$DEV_AUDIO" \
    --voxceleb-audio "$VC_AUDIO" \
    --voxceleb-trials "$VC_TRIALS" \
    --embed-input-shape "1,300,80"

{
    echo "# M5 INT8 validation aggregate"
    echo ""
    echo "Date: $DATE"
    echo ""
    for r in "$REPORT_DIR"/${DATE}-*-validation.md; do
        [ -f "$r" ] && echo "- [$(basename "$r")]($r)"
    done
} > "$REPORT_AGGREGATE"

echo ""
echo "=== Aggregate ==="
cat "$REPORT_AGGREGATE"

exit "$ALL_OK"
