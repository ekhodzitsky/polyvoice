#!/usr/bin/env bash
set -euo pipefail

# Download VoxConverse dev set for M5 INT8 calibration + DER hold-out validation.
#
# Mirrors the shape of `download-voxconverse-test.sh`:
#   1. RTTM ground truth from github.com/joonson/voxconverse (dev/ folder)
#   2. Audio WAV files from mm.kaist.ac.kr (voxconverse_dev_wav.zip)
#
# Output: data/voxconverse-dev/{audio/*.wav, rttm/*.rttm}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/../data/voxconverse-dev}"
AUDIO_DIR="${DATA_DIR}/audio"
RTTM_DIR="${DATA_DIR}/rttm"

mkdir -p "$AUDIO_DIR" "$RTTM_DIR"

AUDIO_URL="https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_dev_wav.zip"
ARCHIVE_URL="https://github.com/joonson/voxconverse/archive/refs/heads/master.tar.gz"

echo "=== VoxConverse Dev Set Download ==="
echo "Output: ${DATA_DIR}"
echo ""

# 1. Download RTTM ground truth (extract dev/ from repo archive)
RTTM_COUNT=$(find "$RTTM_DIR" -name "*.rttm" 2>/dev/null | wc -l | tr -d ' ')
if [ "$RTTM_COUNT" -gt 100 ]; then
    echo "[RTTM] Already exists: ${RTTM_COUNT} files in ${RTTM_DIR}"
else
    echo "[RTTM] Downloading ground-truth annotations..."
    TMP_TAR=$(mktemp /tmp/voxconverse-dev-XXXXXX.tar.gz)
    curl -sL "$ARCHIVE_URL" -o "$TMP_TAR"
    tar xzf "$TMP_TAR" --strip-components=2 -C "$RTTM_DIR" "voxconverse-master/dev/"
    rm -f "$TMP_TAR"
    RTTM_COUNT=$(find "$RTTM_DIR" -name "*.rttm" | wc -l | tr -d ' ')
    echo "[RTTM] Done: ${RTTM_COUNT} files"
fi
echo ""

# 2. Download audio files
WAV_COUNT=$(find "$AUDIO_DIR" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
if [ "$WAV_COUNT" -gt 100 ]; then
    echo "[Audio] Already exists: ${WAV_COUNT} files in ${AUDIO_DIR}"
else
    ZIP_FILE="${DATA_DIR}/voxconverse_dev_wav.zip"
    if [ -f "$ZIP_FILE" ]; then
        echo "[Audio] ZIP already downloaded, extracting..."
    else
        echo "[Audio] Downloading dev audio (~5 GB)..."
        echo "        Source: ${AUDIO_URL}"
        curl -L --progress-bar -o "$ZIP_FILE" "$AUDIO_URL"
    fi

    echo "[Audio] Extracting..."
    unzip -qo "$ZIP_FILE" -d "$AUDIO_DIR"
    if [ -d "${AUDIO_DIR}/voxconverse_dev_wav" ]; then
        mv "${AUDIO_DIR}/voxconverse_dev_wav/"*.wav "$AUDIO_DIR/" 2>/dev/null || true
        rmdir "${AUDIO_DIR}/voxconverse_dev_wav" 2>/dev/null || true
    fi
    rm -f "$ZIP_FILE"
    WAV_COUNT=$(find "$AUDIO_DIR" -name "*.wav" | wc -l | tr -d ' ')
    echo "[Audio] Done: ${WAV_COUNT} files"
fi

echo ""
echo "=== Summary ==="
echo "RTTM:  ${RTTM_COUNT} files in ${RTTM_DIR}/"
echo "Audio: ${WAV_COUNT} files in ${AUDIO_DIR}/"
