#!/usr/bin/env bash
# pipeline-local is kernels + VBx from files on disk. It must not pull the
# downloader (ureq / rustls / ring) or an ONNX runtime (ort / tract-onnx).
# A Cargo resolution failure is a failed check, not "dependency absent".
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

graph="$(cargo tree --locked -e normal --prefix none \
  --no-default-features --features pipeline-local)" || {
  echo "FAIL: cargo tree could not resolve --features pipeline-local" >&2
  exit 1
}

fail_if() {
  local pkg="$1"
  if grep -q "^${pkg} v" <<<"$graph"; then
    echo "FAIL: ${pkg} leaked into pipeline-local:"
    printf '%s\n' "$graph"
    exit 1
  fi
  echo "OK: no ${pkg} in pipeline-local"
}

fail_if ureq
fail_if rustls
fail_if ring
fail_if ort
fail_if tract-onnx

if ! grep -q '^polyvoice-kernels v' <<<"$graph"; then
  echo "FAIL: pipeline-local did not pull polyvoice-kernels"
  exit 1
fi
echo "OK: polyvoice-kernels present in pipeline-local"
echo "OK: pipeline-local dependency graph"
