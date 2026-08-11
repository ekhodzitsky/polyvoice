#!/usr/bin/env bash
# Official Linux / CPU DER gate for the production INT8 path.
#
# Protocol (product truth for non-Apple deploys):
#   - profile balanced (powerset_int8 + resnet34_int8)
#   - pipeline v2 + VBx
#   - execution-provider cpu (never CoreML / auto)
#   - powerset micro-batch N=8 (product default; override POLYVOICE_POWERSET_BATCH_SIZE)
#   - collar 0.25 run (JSON also carries no-collar micro/macro)
#   - full splits when present: VoxConverse-test 232, AMI-test 16
#
# Usage:
#   bash scripts/linux-cpu-der-gate.sh
#   OUT=benchmarks/results/linux-cpu-der-manual bash scripts/linux-cpu-der-gate.sh
#   MAX_VOX=10 MAX_AMI=4 bash scripts/linux-cpu-der-gate.sh   # smoke
#   DOCKER=1 bash scripts/linux-cpu-der-gate.sh              # force Linux via Docker
#
# Writes under OUT/ (default: benchmarks/results/linux-cpu-der-<date>/):
#   host.txt, summary.json, NOTES.md,
#   voxconverse-test-*.json, ami-test-*.json, *.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATE="${DATE:-$(date +%Y-%m-%d)}"
OUT="${OUT:-benchmarks/results/linux-cpu-der-${DATE}}"
EP="${EP:-cpu}"
PROFILE="${PROFILE:-balanced}"
PIPELINE="${PIPELINE:-v2}"
CLUSTERER="${CLUSTERER:-vbx}"
# 0 = all files present in the split.
MAX_VOX="${MAX_VOX:-0}"
MAX_AMI="${MAX_AMI:-0}"
# Product default micro-batch; set explicitly so the gate is self-describing.
export POLYVOICE_POWERSET_BATCH_SIZE="${POLYVOICE_POWERSET_BATCH_SIZE:-8}"
export POLYVOICE_VBX_PLDA_DIR="${POLYVOICE_VBX_PLDA_DIR:-$ROOT/fixtures/vbx-plda}"

# Prefer CARGO_TARGET_DIR when set (Docker / cross builds) so a host-OS
# `target/release/*` binary is never executed under a foreign arch.
if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  BENCH="${BENCH:-$CARGO_TARGET_DIR/release/polyvoice-bench}"
  CLI="${CLI:-$CARGO_TARGET_DIR/release/polyvoice}"
else
  BENCH="${BENCH:-target/release/polyvoice-bench}"
  CLI="${CLI:-target/release/polyvoice}"
fi

# --- optional: re-exec inside Linux (Docker) ---------------------------------
if [[ "${DOCKER:-0}" == "1" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "FATAL: DOCKER=1 but docker not found" >&2
    exit 1
  fi
  # Seed a portable model cache the container can use (flat filenames).
  MODEL_CACHE="${MODEL_CACHE:-$ROOT/.cache/polyvoice-models}"
  mkdir -p "$MODEL_CACHE"
  for f in powerset_int8.onnx resnet34_int8.onnx powerset_int8.onnx.minisig resnet34_int8.onnx.minisig; do
    if [[ -f "models/int8/$f" && ! -f "$MODEL_CACHE/$f" ]]; then
      cp "models/int8/$f" "$MODEL_CACHE/$f"
    fi
  done
  # PLDA fixtures are small and already in-tree.
  # Ubuntu 24.04: ort download-binaries need glibc ≥ ~2.38 (`__isoc23_*`).
  # Debian bookworm (2.36) fails to link. Matches GitHub Actions ubuntu-latest.
  IMG="${DOCKER_IMAGE:-ubuntu:24.04}"
  echo "=== Docker Linux gate (image=$IMG) ==="
  # shellcheck disable=SC2086
  exec docker run --rm \
    -e DATE \
    -e OUT="/work/$OUT" \
    -e EP \
    -e PROFILE \
    -e PIPELINE \
    -e CLUSTERER \
    -e MAX_VOX \
    -e MAX_AMI \
    -e POLYVOICE_POWERSET_BATCH_SIZE \
    -e POLYVOICE_VBX_PLDA_DIR=/work/fixtures/vbx-plda \
    -e XDG_CACHE_HOME=/cache \
    -e CARGO_HOME=/cargo \
    -e CARGO_TARGET_DIR=/target \
    -e PATH="/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e DOCKER=0 \
    -e DEBIAN_FRONTEND=noninteractive \
    -v "$ROOT:/work:rw" \
    -v "$MODEL_CACHE:/cache/polyvoice/models:rw" \
    -v polyvoice-cargo-linux:/cargo \
    -v polyvoice-target-linux:/target \
    -w /work \
    "$IMG" \
    bash -c 'set -euo pipefail
             apt-get update -qq
             apt-get install -y -qq curl ca-certificates build-essential \
               pkg-config libssl-dev git python3 >/dev/null
             if ! command -v cargo >/dev/null 2>&1; then
               curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0
             fi
             # shellcheck disable=SC1091
             source /cargo/env 2>/dev/null || true
             export PATH="/cargo/bin:$PATH"
             bash scripts/linux-cpu-der-gate.sh'
fi

mkdir -p "$OUT"

# --- host fingerprint -------------------------------------------------------
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "uname: $(uname -a)"
  echo "os: $(uname -s)"
  echo "arch: $(uname -m)"
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    echo "distro: ${PRETTY_NAME:-$ID}"
  fi
  if command -v nproc >/dev/null 2>&1; then
    echo "cpus: $(nproc)"
  elif command -v sysctl >/dev/null 2>&1; then
    echo "cpus: $(sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
  fi
  echo "ep: $EP"
  echo "profile: $PROFILE"
  echo "pipeline: $PIPELINE + $CLUSTERER"
  echo "powerset_batch: $POLYVOICE_POWERSET_BATCH_SIZE"
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
} | tee "$OUT/host.txt"

# --- build ------------------------------------------------------------------
need_build=0
if [[ ! -x "$BENCH" ]]; then
  need_build=1
elif ! "$BENCH" --help >/dev/null 2>&1; then
  # Present but wrong arch / corrupt (typical when host target/ is bind-mounted).
  echo "note: $BENCH not runnable on this host — rebuilding"
  need_build=1
fi
if [[ "$need_build" == "1" ]]; then
  echo "=== build release polyvoice-bench (features=cli) ==="
  cargo build --release --features cli --bin polyvoice-bench --bin polyvoice
  if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    BENCH="$CARGO_TARGET_DIR/release/polyvoice-bench"
    CLI="$CARGO_TARGET_DIR/release/polyvoice"
  else
    BENCH="target/release/polyvoice-bench"
    CLI="target/release/polyvoice"
  fi
fi

if ! "$BENCH" --help >/dev/null 2>&1; then
  echo "FATAL: bench binary still not runnable: $BENCH" >&2
  exit 1
fi
echo "bench: $BENCH"
if "$CLI" --version >/dev/null 2>&1; then
  echo "polyvoice: $("$CLI" --version 2>/dev/null || true)" | tee -a "$OUT/host.txt"
fi

# Ensure models (download if cache cold).
if "$CLI" --version >/dev/null 2>&1; then
  echo "=== ensure INT8 models (profile=$PROFILE) ==="
  "$CLI" download-models --profile "$PROFILE" || true
fi

# --- runs -------------------------------------------------------------------
run_split() {
  local label="$1" dataset="$2" maxf="$3"
  if [[ ! -d "$dataset/audio" ]]; then
    echo "[SKIP] $label — missing $dataset/audio" | tee -a "$OUT/run.log"
    return 0
  fi
  local n
  n="$(find "$dataset/audio" -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
  local json="$OUT/${label}.json"
  local log="$OUT/${label}.log"
  local args=(--profile "$PROFILE" --pipeline "$PIPELINE" --clusterer "$CLUSTERER"
              --collar 0.25 --execution-provider "$EP" --output "$json")
  if [[ "$maxf" != "0" && -n "$maxf" ]]; then
    args+=(--max-files "$maxf")
  fi
  echo ""
  echo "=== $label (files_on_disk=$n max_files=${maxf:-all} ep=$EP batch=$POLYVOICE_POWERSET_BATCH_SIZE) ===" \
    | tee -a "$OUT/run.log"
  set +e
  "$BENCH" "$dataset" "${args[@]}" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAIL $label rc=$rc — tail $log:" | tee -a "$OUT/run.log"
    tail -40 "$log" || true
    return "$rc"
  fi
  python3 - "$json" <<'PY' | tee -a "$OUT/run.log"
import json, sys
r = json.load(open(sys.argv[1]))
print(
    f"  files={r.get('files_processed')}  EP={r.get('resolved_execution_provider')}\n"
    f"  DER0 micro={r.get('der_no_collar_micro'):.4f}  macro={r.get('der_no_collar_macro'):.4f}\n"
    f"  DER0.25 micro={r.get('der_collar_micro'):.4f}  macro={r.get('der_collar_macro'):.4f}\n"
    f"  miss={r.get('miss'):.2f} FA={r.get('false_alarm'):.2f} conf={r.get('confusion'):.2f}\n"
    f"  RTFx={r.get('rt_factor_avg'):.2f}  stages={r.get('stage_totals')}"
)
PY
}

run_split "voxconverse-test" "data/voxconverse-test" "$MAX_VOX"
run_split "ami-test" "data/ami-test" "$MAX_AMI"

# --- summary.json -----------------------------------------------------------
python3 - "$OUT" <<'PY'
import json, pathlib, sys, platform, os
out = pathlib.Path(sys.argv[1])
summary = {
    "schema": "polyvoice-linux-cpu-der-gate-v1",
    "protocol": {
        "profile": os.environ.get("PROFILE", "balanced"),
        "pipeline": "v2+vbx",
        "execution_provider": os.environ.get("EP", "cpu"),
        "powerset_batch": int(os.environ.get("POLYVOICE_POWERSET_BATCH_SIZE", "8")),
        "collar_request": 0.25,
        "note": "collar 0.25 JSON also carries no-collar micro/macro",
    },
    "host": {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    },
    "runs": {},
}
for p in sorted(out.glob("*.json")):
    if p.name == "summary.json":
        continue
    r = json.loads(p.read_text())
    summary["runs"][p.stem] = {
        "files_processed": r.get("files_processed"),
        "resolved_execution_provider": r.get("resolved_execution_provider"),
        "der_no_collar_micro": r.get("der_no_collar_micro"),
        "der_no_collar_macro": r.get("der_no_collar_macro"),
        "der_collar_micro": r.get("der_collar_micro"),
        "der_collar_macro": r.get("der_collar_macro"),
        "miss": r.get("miss"),
        "false_alarm": r.get("false_alarm"),
        "confusion": r.get("confusion"),
        "rt_factor_avg": r.get("rt_factor_avg"),
        "speaker_count": r.get("speaker_count"),
        "crate_version": r.get("crate_version"),
        "git_sha": r.get("git_sha"),
        "model_hashes": r.get("model_hashes"),
        "stage_totals": r.get("stage_totals"),
    }
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"wrote {out / 'summary.json'}")
PY

# --- NOTES.md ---------------------------------------------------------------
cat >"$OUT/NOTES.md" <<EOF
# Linux CPU DER gate

**Date:** ${DATE}
**Protocol:** INT8 balanced, pipeline v2 + VBx, EP=\`${EP}\`, powerset micro-batch N=\`${POLYVOICE_POWERSET_BATCH_SIZE}\`.
**Command:** \`bash scripts/linux-cpu-der-gate.sh\` (this tree).

## Headline (no-collar micro)

See \`summary.json\` / per-split JSON. Numbers are the **Linux / CPU product
truth** for non-Apple deploys; Mac CoreML remains a separate headline in
\`tests/der_baseline.json\`.

## Reproduce

\`\`\`bash
# host Linux (or DOCKER=1 from macOS / any Docker host)
bash scripts/download-ami-test.sh
# full VoxConverse-test (232) must already be under data/voxconverse-test/
POLYVOICE_POWERSET_BATCH_SIZE=8 bash scripts/linux-cpu-der-gate.sh
\`\`\`

Smoke (subset):

\`\`\`bash
MAX_VOX=10 MAX_AMI=4 bash scripts/linux-cpu-der-gate.sh
\`\`\`
EOF

echo ""
echo "=== Linux CPU DER gate complete → $OUT ==="
echo "Summary: $OUT/summary.json"
if command -v jq >/dev/null 2>&1 && [[ -f "$OUT/summary.json" ]]; then
  jq -r '.runs | to_entries[] | "\(.key): DER0=\(.value.der_no_collar_micro) DER0.25=\(.value.der_collar_micro) RTFx=\(.value.rt_factor_avg) files=\(.value.files_processed) EP=\(.value.resolved_execution_provider)"' "$OUT/summary.json"
fi
