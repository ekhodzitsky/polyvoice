#!/usr/bin/env bash
set -euo pipefail

# Download AMI test set (Mix-Headset) for DER benchmarking.
#
# Downloads:
#   1. Per-meeting RTTM from pyannote/AMI-diarization-setup (only_words)
#   2. Audio WAV files from the AMI corpus mirror
#
# Output: data/ami-test/{audio/*.wav, rttm/*.rttm}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/../data/ami-test}"
AUDIO_DIR="${DATA_DIR}/audio"
RTTM_DIR="${DATA_DIR}/rttm"

mkdir -p "$AUDIO_DIR" "$RTTM_DIR"

RTTM_BASE="https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test"
AMI_MIRROR="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

# AMI test set meeting IDs (standard pyannote split — 16 meetings)
TEST_MEETINGS=(
    EN2002a EN2002b EN2002c EN2002d
    ES2004a ES2004b ES2004c ES2004d
    IS1009a IS1009b IS1009c IS1009d
    TS3003a TS3003b TS3003c TS3003d
)

echo "=== AMI Test Set Download ==="
echo "Output: ${DATA_DIR}"
echo ""

# 1. Download RTTM ground truth (per-meeting files, concatenated)
RTTM_FILE="${RTTM_DIR}/MixHeadset.test.rttm"
if [ -f "$RTTM_FILE" ] && [ "$(wc -l < "$RTTM_FILE" | tr -d ' ')" -gt 10 ]; then
    echo "[RTTM] Already exists: ${RTTM_FILE}"
else
    echo "[RTTM] Downloading ground-truth annotations (${#TEST_MEETINGS[@]} meetings)..."
    > "$RTTM_FILE"
    for MEETING in "${TEST_MEETINGS[@]}"; do
        echo -n "  ${MEETING}... "
        if curl -sL "${RTTM_BASE}/${MEETING}.rttm" >> "$RTTM_FILE"; then
            echo "ok"
        else
            echo "FAILED"
        fi
    done
    LINES=$(wc -l < "$RTTM_FILE" | tr -d ' ')
    echo "[RTTM] Done: ${LINES} lines total"
fi
echo ""

# 2. Download audio files
echo "[Audio] Downloading ${#TEST_MEETINGS[@]} meetings (Mix-Headset)..."
echo "        This may take a while (~4 GB total)."
echo ""

DOWNLOADED=0
SKIPPED=0
FAILED=0

for MEETING in "${TEST_MEETINGS[@]}"; do
    WAV_FILE="${AUDIO_DIR}/${MEETING}.Mix-Headset.wav"
    if [ -f "$WAV_FILE" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo "  [skip] ${MEETING} (already exists)"
        continue
    fi

    URL="${AMI_MIRROR}/${MEETING}/audio/${MEETING}.Mix-Headset.wav"
    echo -n "  [download] ${MEETING}... "

    if curl -L --progress-bar -o "$WAV_FILE" "$URL" 2>/dev/null; then
        SIZE=$(du -h "$WAV_FILE" | cut -f1)
        echo "done (${SIZE})"
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "FAILED"
        rm -f "$WAV_FILE"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Downloaded: ${DOWNLOADED}"
echo "Skipped:    ${SKIPPED}"
echo "Failed:     ${FAILED}"
echo "RTTM:       ${RTTM_FILE}"
echo "Audio:      ${AUDIO_DIR}/"
echo ""
echo "Run benchmark:"
echo "  POLYVOICE_MODEL_DIR=models cargo run --release --features cli --bin polyvoice-bench -- ${DATA_DIR}"
