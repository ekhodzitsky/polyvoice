#!/usr/bin/env bash
# DER harness sweep (task 300) — the single sanctioned producer of DER numbers.
#
# Runs polyvoice-bench on each dataset split SEPARATELY (dev/test never mixed) on
# the EXACT shipped FP32 artifact, emitting per-split JSON reports that carry BOTH
# the 0.25 s-collar and the no-collar DER (bench computes both passes in one run:
# der_collar_{macro,micro} and der_no_collar_{macro,micro}). A `.uem` next to a
# split is honored automatically.
#
# Prereqs (downloaded on demand below):
#   - ONNX bundle:  polyvoice download-models --profile balanced
#   - data/voxconverse-dev, data/voxconverse-test, data/ami-test
#
# Usage:  scripts/run-der-sweep.sh [out-dir]   (default: /tmp/polyvoice-der-sweep)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="${1:-/tmp/polyvoice-der-sweep}"
cd "$ROOT_DIR"
mkdir -p "$OUT_DIR"

echo "=== 1/3 ensure FP32 models (balanced profile) ==="
cargo run --release --features cli --bin polyvoice -- download-models --profile balanced

echo "=== 2/3 ensure datasets ==="
bash scripts/download-voxconverse-dev.sh  || { echo "WARN: voxconverse-dev download failed"; }
bash scripts/download-voxconverse-test.sh || { echo "WARN: voxconverse-test download failed"; }
bash scripts/download-ami-test.sh         || { echo "WARN: ami-test download failed"; }

echo "=== 3/3 build bench once ==="
cargo build --release --features cli --bin polyvoice-bench

BENCH="$ROOT_DIR/target/release/polyvoice-bench"

# run_split <label> <dataset-dir>
run_split() {
  local label="$1" dir="$2"
  if [ ! -d "$dir/audio" ]; then
    echo "[SKIP] $label — $dir/audio missing"
    return 0
  fi
  local report="$OUT_DIR/${label}.json"
  local uem_args=()
  # First .uem found anywhere under the split is applied to scored regions.
  local uem
  uem="$(find "$dir" -maxdepth 2 -name '*.uem' 2>/dev/null | head -1 || true)"
  if [ -n "$uem" ]; then
    uem_args=(--uem "$uem")
    echo "[$label] using UEM: $uem"
  fi
  echo "=== sweep: $label ==="
  # --collar 0.25 ⇒ report carries BOTH der_collar_* (0.25 s) and der_no_collar_*.
  "$BENCH" "$dir" --profile balanced --collar 0.25 "${uem_args[@]}" --output "$report"
  echo "[$label] report → $report"
  if command -v jq >/dev/null 2>&1; then
    jq -r '"  collar0.25  macro=\(.der_collar_macro)%  micro=\(.der_collar_micro)%\n  no-collar   macro=\(.der_no_collar_macro)%  micro=\(.der_no_collar_micro)%\n  files=\(.files_processed)  models=\([.model_hashes[].model_id] | join(\",\"))"' "$report"
  fi
}

run_split voxconverse-dev  "$ROOT_DIR/data/voxconverse-dev"
run_split voxconverse-test "$ROOT_DIR/data/voxconverse-test"
run_split ami-test         "$ROOT_DIR/data/ami-test"

echo ""
echo "=== sweep complete — per-split reports in $OUT_DIR ==="
echo "Lock baselines into tests/der_baseline.json from these (dev for tuning, test for reporting)."
