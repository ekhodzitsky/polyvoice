#!/usr/bin/env bash
# Offline scoring-chain ablations on the kernel CLI.
#
# Production construction is env-free. This script opts in with
# POLYVOICE_VBX_FROM_ENV=1 so each step overlays one cheap change:
#   r2  clean-mask fallback + filter VBx by unmasked duration
#   r3  AHC seed on L2 embeddings, Euclidean centroid linkage
#   r4  Fa=0.07, GMM (loop_prob=0)
#   r6  emb_scale=1 (no 4.88 rescale before PLDA)
#   r5  speaker count from prior pi>1e-7 + soft-centroid reassignment
#
# Calibrate on DEV only. Report held-out on VoxConverse-test / AMI-test.
#
# Usage:
#   scripts/measure-scoring-chain.sh data/voxconverse-dev 30
#   STEPS="baseline r2 r4" scripts/measure-scoring-chain.sh data/voxconverse-test
set -euo pipefail

DATASET="${1:?usage: measure-scoring-chain.sh <dataset-dir> [max_files]}"
MAX_FILES="${2:-}"
JOBS="${JOBS:-3}"
FEATURES="${FEATURES:-cli}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-$ROOT/benchmarks/results/scoring-chain-2026-09-18}"
mkdir -p "$OUT"

if [[ ! -x "$ROOT/target/release/polyvoice-bench" ]]; then
  cargo build --release --features "$FEATURES" --bin polyvoice-bench >&2
fi
BENCH="$ROOT/target/release/polyvoice-bench"

run_one() {
  local tag="$1"
  shift
  local json="$OUT/${tag}.json"
  local log="$OUT/${tag}.log"
  echo "=== $tag ===" >&2
  env POLYVOICE_VBX_PLDA_DIR="${POLYVOICE_VBX_PLDA_DIR:-$ROOT/fixtures/vbx-plda}" \
      "$@" \
      "$BENCH" "$DATASET" --profile balanced --pipeline v2 --clusterer vbx \
        --collar 0 --jobs "$JOBS" \
        ${MAX_FILES:+--max-files "$MAX_FILES"} \
        --output "$json" \
      >"$log" 2>&1 || {
        echo "$tag FAILED (see $log)" >&2
        return 1
      }
  python3 - "$json" "$tag" << 'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f'{sys.argv[2]:<18} {r["der_no_collar_micro"]:>8.2f} {r["miss"]:>7.2f} '
      f'{r["false_alarm"]:>7.2f} {r["confusion"]:>7.2f} '
      f'{r.get("rt_factor_wall", r.get("rt_factor_avg", 0)):>8.1f}x')
PY
}

printf '%-18s %8s %7s %7s %7s %9s\n' "step" "der%" "miss%" "fa%" "conf%" "rtfx"
STEPS="${STEPS:-baseline r2 r3 r4 r6 r5}"
for step in $STEPS; do
  case "$step" in
    baseline)
      run_one "${DATASET##*/}-baseline" ;;
    r2)
      run_one "${DATASET##*/}-r2" \
        POLYVOICE_VBX_FROM_ENV=1 \
        POLYVOICE_VBX_CLEAN_MASK=1 \
        POLYVOICE_VBX_FILTER_CLEAN=1 ;;
    r3)
      run_one "${DATASET##*/}-r3" \
        POLYVOICE_VBX_FROM_ENV=1 \
        POLYVOICE_VBX_AHC_RAW_L2=1 \
        POLYVOICE_VBX_AHC_THRESHOLD="${R3_THRESHOLD:-0.75}" ;;
    r3r5)
      run_one "${DATASET##*/}-r3r5" \
        POLYVOICE_VBX_FROM_ENV=1 \
        POLYVOICE_VBX_AHC_RAW_L2=1 \
        POLYVOICE_VBX_AHC_THRESHOLD="${R3_THRESHOLD:-0.75}" \
        POLYVOICE_VBX_SOFT_REASSIGN=1 ;;
    r4)
      run_one "${DATASET##*/}-r4" \
        POLYVOICE_VBX_FROM_ENV=1 \
        POLYVOICE_VBX_FA=0.07 \
        POLYVOICE_VBX_LOOP_PROB=0 ;;
    r6)
      run_one "${DATASET##*/}-r6" \
        POLYVOICE_VBX_FROM_ENV=1 \
        POLYVOICE_VBX_EMB_SCALE=1 ;;
    r5)
      run_one "${DATASET##*/}-r5" \
        POLYVOICE_VBX_FROM_ENV=1 \
        POLYVOICE_VBX_SOFT_REASSIGN=1 ;;
    *)
      echo "unknown step $step" >&2; exit 1 ;;
  esac
done
echo "reports: $OUT"
