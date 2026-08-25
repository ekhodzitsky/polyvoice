#!/usr/bin/env bash
# Linux / CPU DER gate for the ort-free kernel path (`cli-native`).
#
# Same protocol as scripts/linux-cpu-der-gate.sh (v2+VBx, INT8 pair, EP=cpu,
# powerset N=8). Builds `--features cli-native` so libonnxruntime is not
# linked. On Linux, GEMM uses system OpenBLAS when pkg-config finds it
# (Docker image installs libopenblas-dev).
#
# Usage:
#   bash scripts/linux-cpu-native-der-gate.sh
#   DOCKER=1 bash scripts/linux-cpu-native-der-gate.sh
#   MAX_VOX=10 MAX_AMI=0 DOCKER=1 bash scripts/linux-cpu-native-der-gate.sh
set -euo pipefail
export FEATURES="${FEATURES:-cli-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/linux-cpu-der-gate.sh"
