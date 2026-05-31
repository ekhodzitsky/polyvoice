#!/usr/bin/env bash
set -euo pipefail

# Download the VoxConverse-test 10-file subset for DER/perf regression tests.
#
# Mirrors a small fixed subset (audio WAV + RTTM) from a GitHub Release asset
# (release tag `test-data`) instead of the upstream ~1.5 GB dataset on
# mm.kaist.ac.kr, which is too slow to fetch within CI job timeouts.
#
# Subset (alphabetically first 10 files): aepyx aggyz aiqwk aorju auzru
#                                          bgvvt bidnq bjruf bmsyn bpzsc
#
# Output: data/voxconverse-test/{audio/*.wav, rttm/*.rttm}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/../data/voxconverse-test}"
AUDIO_DIR="${DATA_DIR}/audio"
RTTM_DIR="${DATA_DIR}/rttm"

ASSET_URL="https://github.com/ekhodzitsky/polyvoice/releases/download/test-data/voxconverse-test-10file.tar.gz"

mkdir -p "$AUDIO_DIR" "$RTTM_DIR"

echo "=== VoxConverse Test Subset Download ==="
echo "Output: ${DATA_DIR}"
echo ""

WAV_COUNT=$(find "$AUDIO_DIR" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
RTTM_COUNT=$(find "$RTTM_DIR" -name "*.rttm" 2>/dev/null | wc -l | tr -d ' ')

if [ "$WAV_COUNT" -ge 10 ] && [ "$RTTM_COUNT" -ge 10 ]; then
    echo "[Subset] Already present: ${WAV_COUNT} wav, ${RTTM_COUNT} rttm"
else
    echo "[Subset] Downloading 10-file subset (~178 MB) from release asset..."
    echo "         ${ASSET_URL}"
    TMP_TAR=$(mktemp /tmp/voxconverse-XXXXXX.tar.gz)
    curl -fsSL --connect-timeout 30 --max-time 600 --retry 5 --retry-delay 10 -o "$TMP_TAR" "$ASSET_URL"
    tar xzf "$TMP_TAR" -C "$DATA_DIR"
    rm -f "$TMP_TAR"
    # macOS-built archives can embed AppleDouble ._* sidecars; on Linux they
    # extract as real files and ._aepyx.wav (sorts before aepyx.wav) is not a
    # RIFF file, breaking read_wav in the perf/DER tests. Strip them.
    find "$DATA_DIR" -name '._*' -type f -delete
    WAV_COUNT=$(find "$AUDIO_DIR" -name "*.wav" | wc -l | tr -d ' ')
    RTTM_COUNT=$(find "$RTTM_DIR" -name "*.rttm" | wc -l | tr -d ' ')
    echo "[Subset] Done"
fi

echo ""
echo "=== Summary ==="
echo "Audio: ${WAV_COUNT} files in ${AUDIO_DIR}/"
echo "RTTM:  ${RTTM_COUNT} files in ${RTTM_DIR}/"
