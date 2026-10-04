#!/usr/bin/env bash
# Assert the workspace resolves to exactly one `ort` version.
#
# The core crate has no `ort` dependency. polyvoice-asr pins `ort` for
# Parakeet TDT. Two `ort` versions linked at once means two runtimes
# (symbol clashes / crashes). This guard is a release/CI gate; run it
# whenever a dependency that pulls `ort` (e.g. parakeet-rs) changes.
set -euo pipefail

# Parakeet companion pin. Core does not depend on ort.
EXPECTED="2.0.0-rc.12"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

metadata="$(cargo metadata --locked --format-version 1)"
versions="$(python3 -c "import sys,json; print('\n'.join(sorted({p['version'] for p in json.load(sys.stdin)['packages'] if p['name']=='ort'})))" <<< "$metadata")"

count="$(printf '%s\n' "$versions" | grep -c . || true)"

if [ "$count" -ne 1 ]; then
  echo "FAIL: workspace must resolve to a single 'ort' version, found ${count}:"
  printf '  %s\n' $versions
  echo "Hint: polyvoice-asr (Parakeet) is the crate that pins ort; check parakeet-rs. Core has no ort dependency."
  exit 1
fi

if [ "$versions" != "$EXPECTED" ]; then
  echo "FAIL: ort resolved to '$versions', expected '$EXPECTED'."
  echo "polyvoice-asr (Parakeet) must pin $EXPECTED. The core crate has no ort dependency."
  exit 1
fi

echo "OK: single ort $versions (polyvoice-asr / Parakeet). Core does not depend on ort."
