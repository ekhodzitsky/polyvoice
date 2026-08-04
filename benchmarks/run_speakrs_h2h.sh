#!/usr/bin/env bash
# Speakrs × polyvoice head-to-head via the single der.py scorer.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/benchmarks"
export SPEAKRS_RTTM_BIN="${SPEAKRS_RTTM_BIN:-$ROOT/benchmarks/tools/speakrs-rttm/target/release/speakrs-rttm}"
MAX="${1:-10}"
OUT="results/speakrs-h2h-$(date +%F)/smoke-${MAX}file.json"
mkdir -p "$(dirname "$OUT")"
if [[ ! -x "$SPEAKRS_RTTM_BIN" ]]; then
  echo "Build speakrs-rttm first:"
  echo "  cargo build --release --manifest-path benchmarks/tools/speakrs-rttm/Cargo.toml --features coreml"
  exit 1
fi
python3 benchmark.py \
  --dataset voxconverse_test \
  --runners polyvoice,speakrs-cpu,speakrs-coreml \
  --max-files "$MAX" \
  --collar 0.25 \
  --output "$OUT"
echo "→ $OUT"
