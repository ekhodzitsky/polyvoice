#!/usr/bin/env bash
# Offline calibration helper for the AHC merge threshold, with and without
# AS-norm score normalization.
#
# Grid-searches the threshold on a dev set, reports no-collar micro DER +
# miss/FA per point, and prints the best point. Not part of the hot path —
# run once per domain and put the winning values into the domain profiles
# (src/clusterer/domain.rs).
#
# Threshold scale differs by scorer: raw cosine lives in [-1, 1] (~0.4-0.6),
# AS-norm z-scores are roughly in [0, 10]. Sweep them separately, e.g.:
#   scripts/calibrate-threshold.sh data/voxconverse-dev 30
#   ASNORM=1 GRID="2 3 4 5 6 7 8" scripts/calibrate-threshold.sh data/voxconverse-dev 30
#
# Calibrate on DEV splits only — never on test.
set -euo pipefail

DATASET="${1:?usage: calibrate-threshold.sh <dataset-dir> [max_files]}"
MAX_FILES="${2:-30}"
GRID="${GRID:-0.40 0.45 0.50 0.55 0.60}"
ASNORM="${ASNORM:-0}"
COHORT="${COHORT:-fixtures/asnorm/cohort_voxdev.npy}"

OUT="$(mktemp -d "${TMPDIR:-/tmp}/calibrate-threshold.XXXXXX")"
echo "reports: $OUT" >&2
cargo build --release --features cli --bin polyvoice-bench >&2

printf '%-12s %12s %8s %8s %8s\n' "threshold" "der_micro%" "miss%" "fa%" "conf%"
best_t=""; best_der=""
for t in $GRID; do
  flags=(--threshold "$t")
  tag="t$t"
  if [ "$ASNORM" = "1" ]; then
    flags+=(--as-norm --cohort "$COHORT")
    tag="asnorm-t$t"
  fi
  json="$OUT/$tag.json"
  ./target/release/polyvoice-bench "$DATASET" --pipeline v2 --clusterer ahc --collar 0 \
    --max-files "$MAX_FILES" --output "$json" "${flags[@]}" \
    > "$OUT/$tag.log" 2>&1
  line="$(python3 - "$json" "$tag" << 'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f'{sys.argv[2]:<12} {r["der_no_collar_micro"]:>12.2f} {r["miss"]:>8.2f} '
      f'{r["false_alarm"]:>8.2f} {r["confusion"]:>8.2f}')
PY
)"
  echo "$line"
  der="$(echo "$line" | awk '{print $2}')"
  if [ -z "$best_der" ] || python3 -c "exit(0 if $der < $best_der else 1)"; then
    best_der="$der"; best_t="$tag"
  fi
done
echo
echo "best: $best_t (no-collar micro DER $best_der%)"
echo "full reports: $OUT"
