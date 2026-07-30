#!/usr/bin/env bash
set -euo pipefail

# Operational follow-up — populate tests/der_baseline.json after the INT8 publish.
# Run after `polyvoice download-models --profile balanced` has cached the bundle
# and `data/voxconverse-test/` has been downloaded via
# `scripts/download-voxconverse-test.sh`.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET="${1:-${ROOT_DIR}/data/voxconverse-test}"

cd "$ROOT_DIR"
cargo run --release --features cli --bin polyvoice-bench -- \
    "$DATASET" --profile balanced --pipeline v2 --clusterer vbx \
    --output /tmp/der-bench-report.json

echo ""
echo "Update tests/der_baseline.json with the numbers from /tmp/der-bench-report.json"
echo "(files, der_collar_0_25_skip_overlap, der_no_collar from running with --collar 0)"
