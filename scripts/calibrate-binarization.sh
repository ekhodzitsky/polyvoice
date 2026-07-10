#!/usr/bin/env bash
# Offline calibration helper for segmentation-posterior binarization.
#
# Grid-searches onset/offset (plus optional min-duration knobs) on a dev set,
# reports no-collar micro DER + miss/FA per point, and prints the best point.
# Not part of the hot path — run it once per model/domain and put the winning
# values into PipelineConfig.binarization.
#
# Usage:
#   scripts/calibrate-binarization.sh <dataset-dir> [max_files]
#   GRID="0.5:0.5 0.6:0.4 0.7:0.3:0.1:0.1" scripts/calibrate-binarization.sh data/voxconverse-test 10
# GRID points are onset:offset[:min_on[:min_off]]; "argmax" = no binarization.
set -euo pipefail

DATASET="${1:?usage: calibrate-binarization.sh <dataset-dir> [max_files]}"
MAX_FILES="${2:-10}"
GRID="${GRID:-argmax 0.5:0.5 0.6:0.4 0.7:0.3 0.6:0.4:0.1:0.1}"

OUT="$(mktemp -d "${TMPDIR:-/tmp}/calibrate-binarization.XXXXXX")"
echo "reports: $OUT" >&2
cargo build --release --features cli --bin polyvoice-bench >&2

printf '%-22s %12s %8s %8s %8s\n' "point" "der_micro%" "miss%" "fa%" "conf%"
best_point=""; best_der=""
for point in $GRID; do
  flags=()
  if [ "$point" != "argmax" ]; then
    IFS=':' read -r onset offset min_on min_off <<< "$point"
    flags+=(--binarize-onset "$onset" --binarize-offset "$offset")
    [ -n "${min_on:-}" ] && flags+=(--binarize-min-on "$min_on")
    [ -n "${min_off:-}" ] && flags+=(--binarize-min-off "$min_off")
  fi
  json="$OUT/${point//:/‗}.json"
  ./target/release/polyvoice-bench "$DATASET" --pipeline v2 --collar 0 \
    --max-files "$MAX_FILES" --output "$json" ${flags[@]+"${flags[@]}"} \
    > "$OUT/${point//:/‗}.log" 2>&1
  line="$(python3 - "$json" "$point" << 'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f'{sys.argv[2]:<22} {r["der_no_collar_micro"]:>12.2f} {r["miss"]:>8.2f} '
      f'{r["false_alarm"]:>8.2f} {r["confusion"]:>8.2f}')
PY
)"
  echo "$line"
  der="$(echo "$line" | awk '{print $2}')"
  if [ -z "$best_der" ] || python3 -c "exit(0 if $der < $best_der else 1)"; then
    best_der="$der"; best_point="$point"
  fi
done
echo
echo "best: $best_point (no-collar micro DER $best_der%)"
