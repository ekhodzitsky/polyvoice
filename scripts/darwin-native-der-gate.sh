#!/usr/bin/env bash
# Darwin kernel full-split DER for the product CLI (`--features cli`).
#
# Same v2+VBx / INT8 / collar-0 protocol as scripts/linux-cpu-der-gate.sh,
# on Accelerate/BNNS. Requires macOS. Full splits: VoxConverse-test 232,
# AMI-test 16.
#
# Usage:
#   bash scripts/darwin-native-der-gate.sh
#   ASSERT_BASELINE=0 bash scripts/darwin-native-der-gate.sh   # measure only
#
# Default OUT is benchmarks/results/darwin-native-der-YYYY-MM-DD/.
# Does not overwrite a hand-written NOTES.md.
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "FATAL: Darwin-only (this host is $(uname -s))." >&2
  echo "Run on macOS with data/voxconverse-test and data/ami-test present." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="${DATE:-$(date +%Y-%m-%d)}"
export FEATURES="${FEATURES:-cli}"
export EP="${EP:-cpu}"
export ASSERT_BASELINE="${ASSERT_BASELINE:-0}"
export OUT="${OUT:-$ROOT/benchmarks/results/darwin-native-der-${DATE}}"
exec bash "$ROOT/scripts/linux-cpu-der-gate.sh"
