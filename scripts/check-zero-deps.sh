#!/usr/bin/env bash
# Zero-dependency aspiration gate for polyvoice.
#
# Layers checked (normal dependency graph only, `-e normal`):
#   1. No `ort` in intentional pure-Rust library combos (delegates to check-ort-free.sh)
#   2. No `earshot` without feature `vad-earshot`
#   3. No `tract` without feature `backend-tract`
#
# This freezes *invariants* so regressions reintroduce native deps into the
# BYO / optional pure-Rust surfaces. Production CLI default remains ort+INT8.
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
echo "=== 3. tract stays opt-in ==="
fail_if_pkg tract-onnx "--no-default-features" --no-default-features
fail_if_pkg tract-onnx "default features"
# backend-tract implies onnx (and thus ort) today — document, don't forbid.
# Only assert tract appears when feature is on.
require_pkg tract-onnx "--features backend-tract" --features backend-tract

echo ""
echo "=== 4. status snapshot (informational) ==="
cat <<'EOF'
| Surface                              | Native dylib? | Notes |
|--------------------------------------|---------------|-------|
| default = [] library                 | no            | BYO embedder + EnergyVad |
| vad-earshot                          | no            | pure-Rust VAD; DER Δ vs Silero ~+2.6pp (opt-in only) |
| backend-tract + ResNet FP32 / CAM++  | no tract dylib| pure-Rust ONNX; FP32 parity OK |
| backend-tract + ResNet INT8          | no            | load/run only — cosine ~0 vs ort; **unsafe** for DER |
| backend-tract + shipping powerset    | n/a           | LOAD FAIL (If / InstanceNorm) |
| backend-tract + powerset rewrite     | no            | export-powerset-tract.py; pipeline remaps; N=1; smoke DER ≈ ort |
| features onnx / cli / pipeline       | yes (ort)     | **product default** (INT8 v2+VBx) |
EOF

echo ""
echo "OK: zero-deps invariants hold."
echo "Product default remains ort+INT8; pure-Rust v2 is opt-in (rewrite + FP32 ResNet)."
echo "See docs/strategy/zero-deps.md"
