#!/usr/bin/env bash
# Linux / CPU DER gate for the ort-free kernel path (`cli-native`).
#
# Same protocol as scripts/linux-cpu-der-gate.sh (v2+VBx, INT8 pair, EP=cpu,
# powerset N=8). Builds `--features cli-native` so libonnxruntime is not
# linked. Linux defaults to Rust kernels, matching release assets. To measure
# the optional LP64 OpenBLAS path, set FEATURES=cli-native,system-openblas;
# install its development files and pkg-config first. Reports record FEATURES.
#
# Usage:
#   bash scripts/linux-cpu-native-der-gate.sh
#   DOCKER=1 bash scripts/linux-cpu-native-der-gate.sh
#   MAX_VOX=10 MAX_AMI=0 DOCKER=1 bash scripts/linux-cpu-native-der-gate.sh
set -euo pipefail
export FEATURES="${FEATURES:-cli-native}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/linux-cpu-der-gate.sh"
