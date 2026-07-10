#!/usr/bin/env bash
# Per-backend realtime-factor (RTFx) comparison.
#
# Runs polyvoice-bench once per execution provider on the SAME dataset and
# prints a comparison table. Reports land in a temp dir (path printed) with
# the requested provider in the filename and the resolved provider inside
# the JSON (resolved_execution_provider), so runs are attributable even when
# an unwired provider falls back to CPU.
#
# Usage:
#   scripts/bench-backends.sh <dataset-dir> [pipeline] [ep ...]
#   scripts/bench-backends.sh data/voxconverse-test-3            # v2, cpu+auto
#   scripts/bench-backends.sh data/ami-test-single v2 cpu coreml
# Extra polyvoice-bench flags go through BENCH_ARGS:
#   BENCH_ARGS="--max-files 10" scripts/bench-backends.sh data/voxconverse-test
set -euo pipefail

DATASET="${1:?usage: bench-backends.sh <dataset-dir> [pipeline] [ep ...]}"
PIPELINE="${2:-v2}"
if [ "$#" -ge 2 ]; then shift 2; else shift 1; fi
EPS=("$@")
if [ "${#EPS[@]}" -eq 0 ]; then EPS=(cpu auto); fi

OUT="$(mktemp -d "${TMPDIR:-/tmp}/bench-backends.XXXXXX")"
echo "reports: $OUT" >&2

cargo build --release --features cli --bin polyvoice-bench >&2

printf '%-10s %-12s %10s %12s %12s\n' "requested" "resolved" "RTFx" "der_micro%" "runtime_s"
for ep in "${EPS[@]}"; do
  json="$OUT/$PIPELINE-$ep.json"
  # shellcheck disable=SC2086 — BENCH_ARGS is intentionally word-split.
  ./target/release/polyvoice-bench "$DATASET" \
    --pipeline "$PIPELINE" --execution-provider "$ep" --output "$json" \
    ${BENCH_ARGS:-} > "$OUT/$PIPELINE-$ep.log" 2>&1
  python3 - "$json" "$ep" << 'PY'
import json, sys
r = json.load(open(sys.argv[1]))
runtime = sum(f["runtime_secs"] for f in r["per_file"])
print(f'{sys.argv[2]:<10} {r["resolved_execution_provider"]:<12} '
      f'{r["rt_factor_avg"]:>10.2f} {r["der_collar_micro"]:>12.2f} {runtime:>12.2f}')
PY
done
