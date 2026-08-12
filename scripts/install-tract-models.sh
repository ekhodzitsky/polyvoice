#!/usr/bin/env bash
# Install pure-Rust (tract) model assets next to the user model cache so
# POLYVOICE_INFERENCE_BACKEND=tract can remap powerset and load FP32 ResNet.
#
# What this does:
#   1. Optionally builds models/powerset_fp32_tract.onnx from shipping FP32
#      (scripts/export-powerset-tract.py — needs python3 + onnx).
#   2. Copies rewrite + wespeaker_resnet34.onnx into the polyvoice models cache
#      (same directory ModelRegistry uses).
#
# Does **not** change product default (ort + INT8). Prefer the signed registry
# path when online: ModelRegistry::ensure("powerset_fp32_tract") downloads from
# the models-tract-v1 GitHub release. This script is the offline/dev shortcut.
#
# Usage:
#   bash scripts/install-tract-models.sh
#   bash scripts/install-tract-models.sh --skip-export   # use existing rewrite only
#   CACHE_DIR=/path/to/models bash scripts/install-tract-models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SKIP_EXPORT=0
for arg in "$@"; do
  case "$arg" in
    --skip-export) SKIP_EXPORT=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

# Match dirs::cache_dir() / ModelRegistry::default layout.
if [[ -n "${CACHE_DIR:-}" ]]; then
  CACHE="$CACHE_DIR"
elif [[ -d "${HOME}/Library/Caches/polyvoice/models" ]]; then
  CACHE="${HOME}/Library/Caches/polyvoice/models"
elif [[ -d "${HOME}/.cache/polyvoice/models" ]]; then
  CACHE="${HOME}/.cache/polyvoice/models"
else
  if [[ "$(uname -s)" == "Darwin" ]]; then
    CACHE="${HOME}/Library/Caches/polyvoice/models"
  else
    CACHE="${HOME}/.cache/polyvoice/models"
  fi
fi
mkdir -p "$CACHE"

REWRITE_SRC="$ROOT/models/powerset_fp32_tract.onnx"
FP32_SRC="$ROOT/models/powerset_fp32.onnx"
RESNET_SRC="$ROOT/models/wespeaker_resnet34.onnx"

if [[ "$SKIP_EXPORT" -eq 0 ]]; then
  if [[ ! -f "$FP32_SRC" ]]; then
    echo "FATAL: $FP32_SRC missing — fetch FP32 powerset or pass --skip-export with an existing rewrite" >&2
    exit 1
  fi
  if ! command -v python3 >/dev/null; then
    echo "FATAL: python3 required to export rewrite (or use --skip-export)" >&2
    exit 1
  fi
  echo "=== export powerset_fp32_tract.onnx ==="
  python3 "$ROOT/scripts/export-powerset-tract.py" --verify
fi

if [[ ! -f "$REWRITE_SRC" ]]; then
  echo "FATAL: rewrite missing at $REWRITE_SRC" >&2
  exit 1
fi

echo "=== install into cache: $CACHE ==="
cp -f "$REWRITE_SRC" "$CACHE/powerset_fp32_tract.onnx"
echo "OK: powerset_fp32_tract.onnx"

if [[ -f "$RESNET_SRC" ]]; then
  cp -f "$RESNET_SRC" "$CACHE/wespeaker_resnet34.onnx"
  echo "OK: wespeaker_resnet34.onnx (FP32; required under tract)"
elif [[ -f "$CACHE/wespeaker_resnet34.onnx" ]]; then
  echo "OK: wespeaker_resnet34.onnx already in cache"
else
  echo "WARN: wespeaker_resnet34.onnx not in models/ or cache — pipeline will ModelRegistry::ensure on first tract run" >&2
fi

echo ""
echo "Installed. Example (not product default):"
echo "  cargo build --release --features 'cli,backend-tract' --bin polyvoice-bench"
echo "  POLYVOICE_INFERENCE_BACKEND=tract \\"
echo "    ./target/release/polyvoice-bench <dataset> --profile balanced --execution-provider cpu"
echo "See docs/strategy/zero-deps.md"
