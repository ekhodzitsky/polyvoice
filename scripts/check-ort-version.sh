#!/usr/bin/env bash
# Assert the whole workspace resolves to exactly ONE `ort` version.
#
# Core's `ort` is optional (`onnx` / `pipeline-full`). polyvoice-asr pins the same
# version for Parakeet TDT. Two `ort` versions linked at once means two
# runtimes (symbol clashes / crashes). This guard is a release/CI gate; run
# it whenever a dependency that pulls `ort` (e.g. parakeet-rs) changes.
set -euo pipefail

EXPECTED="2.0.0-rc.12"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

versions="$(cargo metadata --format-version 1 2>/dev/null \
  | python3 -c "import sys,json; print('\n'.join(sorted({p['version'] for p in json.load(sys.stdin)['packages'] if p['name']=='ort'})))")"

count="$(printf '%s\n' "$versions" | grep -c . || true)"

if [ "$count" -ne 1 ]; then
  echo "FAIL: workspace must resolve to a single 'ort' version, found ${count}:"
  printf '  %s\n' $versions
  echo "Hint: align polyvoice-asr's ort pin with the core (and check parakeet-rs)."
  exit 1
fi

if [ "$versions" != "$EXPECTED" ]; then
  echo "FAIL: ort resolved to '$versions', expected '$EXPECTED'."
  echo "Core's optional ort and polyvoice-asr (Parakeet) must both pin $EXPECTED."
  exit 1
fi

echo "OK: single ort $versions across the workspace (shared ONNX runtime)."
