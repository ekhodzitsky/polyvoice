#!/usr/bin/env bash
# Offline 1-D VBx knob sweep on a DEV split for the kernel CLI.
#
# Production construction is env-free (`VbxClustererConfig::default`). This
# script opts in with POLYVOICE_VBX_FROM_ENV=1 so each grid point overlays one
# knob via POLYVOICE_VBX_{FA,EMB_SCALE,AHC_THRESHOLD,MIN_EMB_SECS}.
#
# Calibrate on DEV only — never on AMI-test or VoxConverse-test. One global
# default: pick the best DEV point, then *report* it on held-out splits.
#
# Usage:
#   scripts/calibrate-vbx.sh data/voxconverse-dev 30
#   KNOB=emb_scale GRID="4.0 4.88 6.0" scripts/calibrate-vbx.sh data/voxconverse-dev 30
#
# Knobs: fa | emb_scale | ahc_threshold | min_emb_secs | loop_prob | fb
# Scoring-chain flags (AHC_RAW_L2, SOFT_REASSIGN, CLEAN_MASK, FILTER_CLEAN)
# live in scripts/measure-scoring-chain.sh — they are not 1-D numeric knobs.
set -euo pipefail

DATASET="${1:?usage: calibrate-vbx.sh <dataset-dir> [max_files]}"
MAX_FILES="${2:-30}"
KNOB="${KNOB:-fa}"
JOBS="${JOBS:-3}"
FEATURES="${FEATURES:-cli}"

case "$KNOB" in
  fa) ENV_NAME=POLYVOICE_VBX_FA; GRID="${GRID:-0.15 0.20 0.25 0.30 0.35 0.40 0.50}" ;;
  emb_scale) ENV_NAME=POLYVOICE_VBX_EMB_SCALE; GRID="${GRID:-3.5 4.0 4.5 4.88 5.5 6.0 7.0}" ;;
  ahc_threshold) ENV_NAME=POLYVOICE_VBX_AHC_THRESHOLD; GRID="${GRID:-0.40 0.45 0.50 0.55 0.60}" ;;
  min_emb_secs) ENV_NAME=POLYVOICE_VBX_MIN_EMB_SECS; GRID="${GRID:-0 0.8 1.2 1.6 2.0 2.5}" ;;
  loop_prob) ENV_NAME=POLYVOICE_VBX_LOOP_PROB; GRID="${GRID:-0.7 0.8 0.9 0.95 0.99}" ;;
  fb) ENV_NAME=POLYVOICE_VBX_FB; GRID="${GRID:-0.4 0.6 0.8 1.0 1.2}" ;;
  *) echo "unknown KNOB=$KNOB (fa|emb_scale|ahc_threshold|min_emb_secs|loop_prob|fb)" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$(mktemp -d "${TMPDIR:-/tmp}/calibrate-vbx.XXXXXX")}"
mkdir -p "$OUT"
echo "reports: $OUT  knob=$KNOB  grid=[$GRID]  max_files=$MAX_FILES  jobs=$JOBS" >&2

cargo build --release --features "$FEATURES" --bin polyvoice-bench >&2
BENCH="$ROOT/target/release/polyvoice-bench"

printf '%-14s %12s %8s %8s %8s\n' "$KNOB" "der_micro%" "miss%" "fa%" "conf%"
best_tag=""; best_der=""
for v in $GRID; do
  tag="${KNOB}=$v"
  json="$OUT/${KNOB}-$v.json"
  log="$OUT/${KNOB}-$v.log"
  env POLYVOICE_VBX_FROM_ENV=1 \
      POLYVOICE_VBX_PLDA_DIR="${POLYVOICE_VBX_PLDA_DIR:-$ROOT/fixtures/vbx-plda}" \
      "$ENV_NAME=$v" \
      "$BENCH" "$DATASET" --profile balanced --pipeline v2 --clusterer vbx \
        --collar 0 --max-files "$MAX_FILES" --jobs "$JOBS" --output "$json" \
      >"$log" 2>&1
  line="$(python3 - "$json" "$v" << 'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f'{sys.argv[2]:<14} {r["der_no_collar_micro"]:>12.2f} {r["miss"]:>8.2f} '
      f'{r["false_alarm"]:>8.2f} {r["confusion"]:>8.2f}')
PY
)"
  echo "$line"
  der="$(echo "$line" | awk '{print $2}')"
  if [ -z "$best_der" ] || python3 -c "import sys; sys.exit(0 if float('$der') < float('$best_der') else 1)"; then
    best_der="$der"; best_tag="$tag"
  fi
done
echo
echo "best: $best_tag (no-collar micro DER $best_der%)"
echo "full reports: $OUT"
