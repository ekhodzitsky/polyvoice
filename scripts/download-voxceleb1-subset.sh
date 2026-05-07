#!/usr/bin/env bash
set -euo pipefail

# Download a VoxCeleb1 subset for M5 INT8 EER validation.
#
# Strategy:
#   1. Trial pairs file (canonical "veri_test2.txt", small) — always download.
#   2. Audio: VoxCeleb1-test split (~40 speakers, ~1 GB, public).
#      The full 1k-speaker subset that spec §9.4 names is license-gated;
#      this fallback is documented in the spec's Risks section.
#
# Output:
#   data/voxceleb1-subset/wav/<speaker_id>/<utt_id>.wav
#   data/voxceleb1-subset/lists/veri_test.txt   (trial pairs for EER)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/../data/voxceleb1-subset}"
WAV_DIR="${DATA_DIR}/wav"
LIST_DIR="${DATA_DIR}/lists"

mkdir -p "$WAV_DIR" "$LIST_DIR"

TRIAL_URL="https://mm.kaist.ac.kr/datasets/voxceleb/meta/veri_test2.txt"
TEST_AUDIO_URL="https://mm.kaist.ac.kr/datasets/voxceleb/data/vox1_test_wav.zip"

echo "=== VoxCeleb1 Subset Download (M5 EER validation) ==="
echo "Output: ${DATA_DIR}"
echo ""

# 1. Trial list (always download — small)
TRIAL_FILE="${LIST_DIR}/veri_test.txt"
if [ -s "$TRIAL_FILE" ]; then
    LINE_COUNT=$(wc -l < "$TRIAL_FILE" | tr -d ' ')
    echo "[Trials] Already exists: ${LINE_COUNT} pairs in ${TRIAL_FILE}"
else
    echo "[Trials] Downloading trial pairs..."
    curl -fsSL "$TRIAL_URL" -o "$TRIAL_FILE"
    LINE_COUNT=$(wc -l < "$TRIAL_FILE" | tr -d ' ')
    echo "[Trials] Done: ${LINE_COUNT} pairs"
fi

# 2. Audio files
WAV_COUNT=$(find "$WAV_DIR" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
if [ "$WAV_COUNT" -gt 1000 ]; then
    echo "[Audio] Already exists: ${WAV_COUNT} files"
else
    ZIP_FILE="${DATA_DIR}/vox1_test_wav.zip"
    if [ ! -f "$ZIP_FILE" ]; then
        echo "[Audio] Downloading VoxCeleb1 test split (~1 GB, ~40 speakers, public)..."
        echo "        Source: ${TEST_AUDIO_URL}"
        curl -fL --progress-bar -o "$ZIP_FILE" "$TEST_AUDIO_URL"
    fi
    echo "[Audio] Extracting..."
    unzip -qo "$ZIP_FILE" -d "$WAV_DIR"
    rm -f "$ZIP_FILE"
    WAV_COUNT=$(find "$WAV_DIR" -name "*.wav" | wc -l | tr -d ' ')
    echo "[Audio] Done: ${WAV_COUNT} files"
fi

echo ""
echo "=== Summary ==="
echo "Trials:  $(wc -l < "$TRIAL_FILE" | tr -d ' ') pairs in ${TRIAL_FILE}"
echo "Audio:   ${WAV_COUNT} files in ${WAV_DIR}/"
echo ""
echo "EER validation will run against this subset; spec §9.4 names a 1k-speaker"
echo "subset, but VoxCeleb1-test (~40 speakers, public) is the documented fallback"
echo "when the full split is license-gated."
