#!/usr/bin/env bash
set -euo pipefail

# Download a single AMI test meeting (EN2002a) for E2E smoke tests.
#
# Output: data/ami-test-single/{audio/EN2002a.Mix-Headset.wav, rttm/EN2002a.rttm}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/../data/ami-test-single}"
AUDIO_DIR="${DATA_DIR}/audio"
RTTM_DIR="${DATA_DIR}/rttm"

mkdir -p "$AUDIO_DIR" "$RTTM_DIR"

RTTM_URL="https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/EN2002a.rttm"
AUDIO_URL="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/EN2002a/audio/EN2002a.Mix-Headset.wav"

WAV_FILE="${AUDIO_DIR}/EN2002a.Mix-Headset.wav"
RTTM_FILE="${RTTM_DIR}/EN2002a.rttm"

echo "=== AMI Test Single (EN2002a) Download ==="
echo "Output: ${DATA_DIR}"
echo ""

# 1. Download RTTM
if [ -f "$RTTM_FILE" ] && [ "$(wc -l < "$RTTM_FILE" | tr -d ' ')" -gt 1 ]; then
    echo "[RTTM] Already exists: ${RTTM_FILE}"
else
    echo "[RTTM] Downloading..."
    curl -sL --retry 5 --retry-delay 10 --continue-at - "$RTTM_URL" -o "$RTTM_FILE"
    echo "[RTTM] Done"
fi
echo ""

# 2. Download audio
if [ -f "$WAV_FILE" ]; then
    echo "[Audio] Already exists: ${WAV_FILE}"
else
    echo "[Audio] Downloading EN2002a.Mix-Headset.wav (~68 MB)..."
    curl -L --retry 5 --retry-delay 10 --continue-at - -o "$WAV_FILE" "$AUDIO_URL"
    SIZE=$(du -h "$WAV_FILE" | cut -f1)
    echo "[Audio] Done (${SIZE})"
fi

echo ""
echo "=== Summary ==="
echo "RTTM:  ${RTTM_FILE}"
echo "Audio: ${WAV_FILE}"
