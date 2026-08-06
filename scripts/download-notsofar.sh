#!/usr/bin/env bash
set -euo pipefail

# Download the NOTSOFAR-1 dev-set-1 (single-channel track) for cross-corpus
# DER evaluation.
#
# Source: huggingface.co/datasets/microsoft/NOTSOFAR, split
# benchmark-datasets/dev_set/240825.1_dev1 (GT available, CC BY 4.0 — see
# benchmarks/DATA_LICENSE). The Azure blob original requires AzCopy and is
# blocked on some networks; the HF mirror serves plain HTTPS.
#
# Per meeting we take ONE single-channel far-field device (the first `sc_*`
# device directory, sorted — deterministic) plus the GT transcription JSON,
# and convert GT to RTTM alongside the existing datasets:
#
# Output: data/notsofar-dev/{audio/*.wav, rttm/*.rttm}
#
# Usage: scripts/download-notsofar.sh [data-dir] [max-meetings]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-${SCRIPT_DIR}/../data/notsofar-dev}"
MAX_MEETINGS="${2:-0}" # 0 = all meetings in the split

SPLIT="benchmark-datasets/dev_set/240825.1_dev1/MTG"
HF_BASE="https://huggingface.co/datasets/microsoft/NOTSOFAR/resolve/main"
HF_API="https://huggingface.co/api/datasets/microsoft/NOTSOFAR"
AUDIO_DIR="${DATA_DIR}/audio"
RTTM_DIR="${DATA_DIR}/rttm"
GT_DIR="${DATA_DIR}/gt"

mkdir -p "$AUDIO_DIR" "$RTTM_DIR" "$GT_DIR"

AUTH=()
if [ -f "$HOME/.cache/huggingface/token" ]; then
    AUTH=(-H "Authorization: Bearer $(cat "$HOME/.cache/huggingface/token")")
fi

echo "=== NOTSOFAR-1 dev-set-1 (single-channel) download ==="
echo "Output: ${DATA_DIR}"
echo ""

# One API call: full file listing of the split.
LISTING=$(curl -sfL "${AUTH[@]}" "$HF_API" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for s in d['siblings']:
    print(s['rfilename'])
")

# Per meeting: first sc_* device dir (sorted) = the single-channel device.
PLAN=$(echo "$LISTING" | python3 -c "
import sys, collections
split = '${SPLIT}/'
devices = collections.defaultdict(set)
for line in sys.stdin:
    line = line.strip()
    if not line.startswith(split):
        continue
    # benchmark-datasets/dev_set/<ver>/MTG/<MTG_id>/<device>/<file>
    parts = line.split('/')
    if len(parts) < 7:
        continue
    mtg, dev = parts[4], parts[5]
    if dev.startswith('sc_'):
        devices[mtg].add(dev)
for mtg in sorted(devices):
    print(mtg, sorted(devices[mtg])[0])
")

COUNT=0
while read -r mtg device; do
    [ -z "$mtg" ] && continue
    if [ "$MAX_MEETINGS" -gt 0 ] && [ "$COUNT" -ge "$MAX_MEETINGS" ]; then
        break
    fi

    wav_dst="${AUDIO_DIR}/${mtg}.wav"
    if [ -f "$wav_dst" ]; then
        echo "[skip] ${mtg}.wav exists"
    else
        echo "[ wav] ${mtg} <- ${device}/ch0.wav"
        curl -sfL "${AUTH[@]}" "${HF_BASE}/${SPLIT}/${mtg}/${device}/ch0.wav" -o "$wav_dst"
    fi

    gt_dst="${GT_DIR}/${mtg}.json"
    if [ ! -f "$gt_dst" ]; then
        curl -sfL "${AUTH[@]}" "${HF_BASE}/${SPLIT}/${mtg}/gt_transcription.json" -o "$gt_dst"
    fi

    COUNT=$((COUNT + 1))
done <<< "$PLAN"

echo ""
echo "Downloaded ${COUNT} meetings. Converting GT to RTTM..."

python3 "$SCRIPT_DIR/notsofar-to-rttm.py" "$DATA_DIR"

echo ""
echo "=== Done: $(ls "$AUDIO_DIR" | wc -l | tr -d ' ') wav, $(ls "$RTTM_DIR" | wc -l | tr -d ' ') rttm in ${DATA_DIR} ==="
