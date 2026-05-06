#!/usr/bin/env bash
set -euo pipefail

# Download VoxConverse test set for DER benchmarking.
#
# Downloads:
#   1. RTTM ground truth from github.com/joonson/voxconverse
#   2. Audio WAV files from mm.kaist.ac.kr
#
# Output: data/voxconverse-test/{audio/*.wav, rttm/*.rttm}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/../data/voxconverse-test}"
AUDIO_DIR="${DATA_DIR}/audio"
RTTM_DIR="${DATA_DIR}/rttm"

mkdir -p "$AUDIO_DIR" "$RTTM_DIR"

AUDIO_URL="https://mm.kaist.ac.kr/datasets/voxconverse/data/voxconverse_test_wav.zip"
AUDIO_MD5="834558bbd9b1ffd2d4893181556ceddd"
ARCHIVE_URL="https://github.com/joonson/voxconverse/archive/refs/heads/master.tar.gz"

echo "=== VoxConverse Test Set Download ==="
echo "Output: ${DATA_DIR}"
echo ""

# 1. Download RTTM ground truth (extract test/ from repo archive)
RTTM_COUNT=$(find "$RTTM_DIR" -name "*.rttm" 2>/dev/null | wc -l | tr -d ' ')
if [ "$RTTM_COUNT" -gt 50 ]; then
    echo "[RTTM] Already exists: ${RTTM_COUNT} files in ${RTTM_DIR}"
else
    echo "[RTTM] Downloading ground-truth annotations..."
    TMP_TAR=$(mktemp /tmp/voxconverse-XXXXXX.tar.gz)
    curl -sL "$ARCHIVE_URL" -o "$TMP_TAR"
    tar xzf "$TMP_TAR" --strip-components=2 -C "$RTTM_DIR" "voxconverse-master/test/"
    rm -f "$TMP_TAR"
    RTTM_COUNT=$(find "$RTTM_DIR" -name "*.rttm" | wc -l | tr -d ' ')
    echo "[RTTM] Done: ${RTTM_COUNT} files"
fi
echo ""

# 2. Download audio files
WAV_COUNT=$(find "$AUDIO_DIR" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
if [ "$WAV_COUNT" -gt 50 ]; then
    echo "[Audio] Already exists: ${WAV_COUNT} files in ${AUDIO_DIR}"
else
    ZIP_FILE="${DATA_DIR}/voxconverse_test_wav.zip"
    if [ -f "$ZIP_FILE" ]; then
        echo "[Audio] ZIP already downloaded, extracting..."
    else
        echo "[Audio] Downloading test audio (~1.5 GB)..."
        echo "        Source: ${AUDIO_URL}"
        curl -L --progress-bar -o "$ZIP_FILE" "$AUDIO_URL"

        # Verify checksum
        if command -v md5sum &>/dev/null; then
            ACTUAL=$(md5sum "$ZIP_FILE" | cut -d' ' -f1)
        elif command -v md5 &>/dev/null; then
            ACTUAL=$(md5 -q "$ZIP_FILE")
        else
            ACTUAL=""
        fi
        if [ -n "$ACTUAL" ] && [ "$ACTUAL" != "$AUDIO_MD5" ]; then
            echo "WARNING: MD5 mismatch (expected ${AUDIO_MD5}, got ${ACTUAL})"
        fi
    fi

    echo "[Audio] Extracting..."
    unzip -qo "$ZIP_FILE" -d "$AUDIO_DIR"
    # Flatten if extracted into a subdirectory
    if [ -d "${AUDIO_DIR}/voxconverse_test_wav" ]; then
        mv "${AUDIO_DIR}/voxconverse_test_wav/"*.wav "$AUDIO_DIR/" 2>/dev/null || true
        rmdir "${AUDIO_DIR}/voxconverse_test_wav" 2>/dev/null || true
    fi
    rm -f "$ZIP_FILE"
    WAV_COUNT=$(find "$AUDIO_DIR" -name "*.wav" | wc -l | tr -d ' ')
    echo "[Audio] Done: ${WAV_COUNT} files"
fi

echo ""
echo "=== Summary ==="
echo "RTTM:  ${RTTM_COUNT} files in ${RTTM_DIR}/"
echo "Audio: ${WAV_COUNT} files in ${AUDIO_DIR}/"
echo ""
echo "Run benchmark:"
echo "  cargo run --release --features cli --bin polyvoice-bench -- ${DATA_DIR} --threshold 0.4"
