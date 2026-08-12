#!/usr/bin/env bash
# Prove standalone workspaces resolve under --locked (mirrors CI job
# standalone-lockfiles). After a core crate version bump, run
# `bash scripts/bump-version.sh <ver>` (or `cargo update -p polyvoice` in each
# manifest) so these do not go stale.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MANIFESTS=(
  fuzz/Cargo.toml
  polyvoice-asr-sherpa/Cargo.toml
  python/Cargo.toml
)

echo "=== standalone lockfiles (cargo metadata --locked) ==="
for m in "${MANIFESTS[@]}"; do
  if [[ ! -f "$m" ]]; then
    echo "FAIL: missing manifest $m" >&2
    exit 1
  fi
  cargo metadata --locked --format-version 1 --manifest-path "$m" >/dev/null
  echo "OK: $m"
done
echo "OK: all standalone lockfiles resolve under --locked"
