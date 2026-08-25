#!/usr/bin/env bash
# Assert the ort-free library path never pulls `ort` into its normal dep graph.
#
# `default = []` is intentional: consumers can use Pipeline / StreamingPipeline /
# EnergyVad / clustering with a BYO embedder and no ONNX Runtime native dylib.
# This gate fails CI if a feature regression reintroduces `dep:ort` into those
# graphs. Only **normal** deps are checked (`-e normal`) so dev-deps cannot
# trigger a false positive.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Returns 0 if `ort` appears in the normal dependency graph for the given
# cargo-tree args; 1 if cargo cannot resolve package ID `ort` (clean).
ort_in_graph() {
  # Redirect stderr: when clean, cargo prints "package ID specification `ort`
  # did not match any packages" and exits non-zero. We only care whether any
  # package lines land on stdout.
  # Portable grep (CI images may not have ripgrep). Any stdout line means
  # cargo resolved an `ort` package into the normal graph.
  if cargo tree -e normal -i ort "$@" 2>/dev/null | grep -q .; then
    return 0
  fi
  return 1
}

fail_if_ort() {
  local label="$1"
  shift
  if ort_in_graph "$@"; then
    echo "FAIL: ort leaked into ${label} dependency graph:"
    cargo tree -e normal -i ort "$@" || true
    exit 1
  fi
  echo "OK: no ort in ${label}"
}

# Bare default-free core (no features).
fail_if_ort "--no-default-features" --no-default-features

# Pure-Rust feature combinations that must stay free of ort.
fail_if_ort "--no-default-features --features clusterer,vbx" \
  --no-default-features --features clusterer,vbx

fail_if_ort "--no-default-features --features clusterer,vbx,spectral,segmentation,embedder,resegmentation,attribution" \
  --no-default-features --features clusterer,vbx,spectral,segmentation,embedder,resegmentation,attribution

# Pure-Rust inference: tract must not pull the native ONNX Runtime dylib.
fail_if_ort "--no-default-features --features backend-tract" \
  --no-default-features --features backend-tract

fail_if_ort "--no-default-features --features pipeline-tract,vbx" \
  --no-default-features --features pipeline-tract,vbx

fail_if_ort "--no-default-features --features cli-tract" \
  --no-default-features --features cli-tract

fail_if_ort "--no-default-features --features embedder-native" \
  --no-default-features --features embedder-native

fail_if_ort "--no-default-features --features segmenter-native" \
  --no-default-features --features segmenter-native

fail_if_ort "--no-default-features --features pipeline-native,vbx" \
  --no-default-features --features pipeline-native,vbx

fail_if_ort "--no-default-features --features cli-native" \
  --no-default-features --features cli-native

fail_if_ort "--no-default-features --features cli" \
  --no-default-features --features cli

fail_if_ort "--no-default-features --features ffi" \
  --no-default-features --features ffi

echo "OK: ort-free library graphs stay free of ort."
