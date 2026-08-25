#!/usr/bin/env bash
# Zero-dependency aspiration gate for polyvoice.
#
# Layers checked (normal dependency graph only, `-e normal`):
#   1. No `ort` in intentional pure-Rust library combos (delegates to check-ort-free.sh)
#   2. No `earshot` without feature `vad-earshot`
#   3. No `tract` without feature `backend-tract`; `backend-tract` /
#      `pipeline-tract` must not pull `ort`
#
# This freezes *invariants* so regressions reintroduce native deps into the
# BYO / product kernel surfaces. Production CLI is kernels (`cli`); ort is `cli-ort`.
# An **opt-in** pure-Rust v2 path exists (backend-tract + powerset rewrite +
# FP32 ResNet) — see docs/strategy/zero-deps.md — but is not product default.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pkg_in_graph() {
  local pkg="$1"
  shift
  if cargo tree -e normal -i "$pkg" "$@" 2>/dev/null | grep -q .; then
    return 0
  fi
  return 1
}

fail_if_pkg() {
  local pkg="$1"
  local label="$2"
  shift 2
  if pkg_in_graph "$pkg" "$@"; then
    echo "FAIL: ${pkg} leaked into ${label}:"
    cargo tree -e normal -i "$pkg" "$@" || true
    exit 1
  fi
  echo "OK: no ${pkg} in ${label}"
}

require_pkg() {
  local pkg="$1"
  local label="$2"
  shift 2
  if ! pkg_in_graph "$pkg" "$@"; then
    echo "FAIL: expected ${pkg} in ${label} but cargo tree -i found nothing"
    exit 1
  fi
  echo "OK: ${pkg} present in ${label}"
}

echo "=== 1. ort-free library graphs ==="
bash scripts/check-ort-free.sh

echo ""
echo "=== 2. earshot stays opt-in ==="
fail_if_pkg earshot "--no-default-features" --no-default-features
fail_if_pkg earshot "default features" 
# With feature, must pull earshot (weights-in-crate pure-Rust VAD).
require_pkg earshot "--features vad-earshot" --features vad-earshot
# vad-earshot alone must not require ort.
fail_if_pkg ort "--features vad-earshot only" --features vad-earshot

echo ""
echo "=== 3. tract stays opt-in and does not pull ort ==="
fail_if_pkg tract-onnx "--no-default-features" --no-default-features
fail_if_pkg tract-onnx "default features"
require_pkg tract-onnx "--features backend-tract" --features backend-tract
fail_if_pkg ort "--features backend-tract only" --no-default-features --features backend-tract
fail_if_pkg ort "--features pipeline-tract,vbx" --no-default-features --features pipeline-tract,vbx
fail_if_pkg ort "--features cli-tract" --no-default-features --features cli-tract
require_pkg tract-onnx "--features cli-tract" --no-default-features --features cli-tract
fail_if_pkg ort "--features embedder-native" --no-default-features --features embedder-native
require_pkg polyvoice-kernels "--features embedder-native" --no-default-features --features embedder-native
fail_if_pkg ort "--features segmenter-native" --no-default-features --features segmenter-native
require_pkg polyvoice-kernels "--features segmenter-native" --no-default-features --features segmenter-native
fail_if_pkg ort "--features pipeline-native,vbx" --no-default-features --features pipeline-native,vbx
fail_if_pkg tract-onnx "--features pipeline-native,vbx" --no-default-features --features pipeline-native,vbx
fail_if_pkg ort "--features cli-native" --no-default-features --features cli-native
fail_if_pkg tract-onnx "--features cli-native" --no-default-features --features cli-native
require_pkg polyvoice-kernels "--features cli-native" --no-default-features --features cli-native
fail_if_pkg ort "--features cli" --no-default-features --features cli
fail_if_pkg tract-onnx "--features cli" --no-default-features --features cli
require_pkg polyvoice-kernels "--features cli" --no-default-features --features cli
fail_if_pkg ort "--features ffi" --no-default-features --features ffi
require_pkg ort "--features cli-ort" --no-default-features --features cli-ort

echo ""
echo "=== 4. status snapshot (informational) ==="
cat <<'EOF'
| Surface                              | Native dylib? | Notes |
|--------------------------------------|---------------|-------|
| default = [] library                 | no            | BYO embedder + EnergyVad |
| vad-earshot                          | no            | pure-Rust VAD; DER Δ vs Silero ~+2.6pp (opt-in only) |
| backend-tract (no onnx)              | no            | tract-onnx only; **no ort** |
| pipeline-tract + vbx                 | no            | v2 stack without libonnxruntime |
| cli-tract                            | no            | same CLI bins; tract; no ort |
| embedder-native                      | no            | hand-written ResNet34 kernels |
| segmenter-native                     | no            | hand-written powerset + LSTM |
| pipeline-native / cli-native         | no            | v2 + kernels; no ort/tract |
| backend-tract + ResNet FP32 / CAM++  | no tract dylib| pure-Rust ONNX; FP32 parity OK |
| backend-tract + ResNet INT8          | no            | load/run only — cosine ~0 vs ort; **unsafe** for DER |
| backend-tract + shipping powerset    | n/a           | LOAD FAIL (If / InstanceNorm) |
| backend-tract + powerset rewrite     | no            | export-powerset-tract.py; pipeline remaps; N=1; smoke DER ≈ ort |
| pipeline-native / cli / ffi          | no            | **product default** (kernels) |
| features onnx / cli-ort / pipeline-full | yes (ort)  | opt-in INT8 ONNX Runtime |
EOF

echo ""
echo "OK: zero-deps invariants hold."
echo "Product default is cli (kernels, no ort). ONNX Runtime is --features cli-ort / onnx."
echo "See docs/strategy/zero-deps.md"
