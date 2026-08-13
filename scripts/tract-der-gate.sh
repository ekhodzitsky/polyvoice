#!/usr/bin/env bash
# Opt-in pure-Rust (tract) DER + RTF smoke / gate helper.
#
# Compares product ort+INT8 vs POLYVOICE_INFERENCE_BACKEND=tract
# (signed powerset_fp32_tract + FP32 ResNet) on the same dataset.
#
# Usage:
#   bash scripts/tract-der-gate.sh
#   DATASET=data/ami-test OUT=benchmarks/results/tract-der-ami bash scripts/tract-der-gate.sh
#   MAX_FILES=10 DATASET=data/voxconverse-test bash scripts/tract-der-gate.sh
#
# Not a product release gate — product remains ort+INT8 (see
# scripts/linux-cpu-der-gate.sh). Requires features cli,backend-tract and
# network/cache for models unless already installed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATE="${DATE:-$(date +%Y-%m-%d)}"
OUT="${OUT:-benchmarks/results/tract-der-${DATE}}"
DATASET="${DATASET:-data/ami-test}"
PROFILE="${PROFILE:-balanced}"
PIPELINE="${PIPELINE:-v2}"
CLUSTERER="${CLUSTERER:-vbx}"
EP="${EP:-cpu}"
MAX_FILES="${MAX_FILES:-0}"   # 0 = all files present
export POLYVOICE_VBX_PLDA_DIR="${POLYVOICE_VBX_PLDA_DIR:-$ROOT/fixtures/vbx-plda}"

if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
  BENCH="${BENCH:-$CARGO_TARGET_DIR/release/polyvoice-bench}"
else
  BENCH="${BENCH:-target/release/polyvoice-bench}"
fi

mkdir -p "$OUT"
uname -a | tee "$OUT/host.txt"
(sysctl -n machdep.cpu.brand_string 2>/dev/null || true) | tee -a "$OUT/host.txt"
echo "dataset=$DATASET profile=$PROFILE ep=$EP max_files=$MAX_FILES" | tee -a "$OUT/host.txt"

if [[ ! -d "$DATASET/audio" ]] || [[ ! -d "$DATASET/rttm" ]]; then
  echo "FATAL: dataset needs audio/ and rttm/ under $DATASET" >&2
  exit 1
fi

if [[ ! -x "$BENCH" ]]; then
  echo "Building polyvoice-bench (cli,backend-tract)..."
  cargo build --release --features "cli,backend-tract" --bin polyvoice-bench
fi

COMMON=(--profile "$PROFILE" --pipeline "$PIPELINE" --clusterer "$CLUSTERER"
        --collar 0.0 --execution-provider "$EP")
if [[ "$MAX_FILES" != "0" ]]; then
  COMMON+=(--max-files "$MAX_FILES")
fi

run_one() {
  local name="$1"
  shift
  local json="$OUT/${name}.json"
  local log="$OUT/${name}.log"
  echo ""
  echo "=== $name ==="
  set +e
  /usr/bin/time -p env "$@" "$BENCH" "$DATASET" "${COMMON[@]}" --output "$json" 2>"$log"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAIL: $name rc=$rc — see $log" >&2
    tail -30 "$log" || true
    exit "$rc"
  fi
  python3 - "$json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
st = d.get("stage_totals") or {}
print(f"  files={d.get('files_processed')} RTFx={d.get('rt_factor_avg'):.2f} "
      f"DER0_micro={d.get('der_no_collar_micro'):.2f} "
      f"seg_s={st.get('segmentation_secs', 0):.1f} emb_s={st.get('embedding_secs', 0):.1f}")
PY
}

# ort product (clear any forced backend)
run_one ort -u POLYVOICE_INFERENCE_BACKEND
# tract pure-Rust path
run_one tract POLYVOICE_INFERENCE_BACKEND=tract

python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
ort = json.load(open(out / "ort.json"))
tr = json.load(open(out / "tract.json"))
summary = {
    "dataset_note": "see host.txt",
    "ort": {
        "files": ort.get("files_processed"),
        "rt_factor_avg": ort.get("rt_factor_avg"),
        "der_no_collar_micro": ort.get("der_no_collar_micro"),
        "stage_totals": ort.get("stage_totals"),
    },
    "tract": {
        "files": tr.get("files_processed"),
        "rt_factor_avg": tr.get("rt_factor_avg"),
        "der_no_collar_micro": tr.get("der_no_collar_micro"),
        "stage_totals": tr.get("stage_totals"),
    },
    "delta_der0_pp": tr["der_no_collar_micro"] - ort["der_no_collar_micro"],
    "rtfx_ratio_tract_over_ort": tr["rt_factor_avg"] / max(ort["rt_factor_avg"], 1e-9),
}
json.dump(summary, open(out / "summary.json", "w"), indent=2)
print("=== COMPARE ===")
print(f"DER0  ort={summary['ort']['der_no_collar_micro']:.2f}  "
      f"tract={summary['tract']['der_no_collar_micro']:.2f}  "
      f"Δ={summary['delta_der0_pp']:+.2f} pp")
print(f"RTFx  ort={summary['ort']['rt_factor_avg']:.2f}  "
      f"tract={summary['tract']['rt_factor_avg']:.2f}  "
      f"ratio={summary['rtfx_ratio_tract_over_ort']:.3f}")
print(f"wrote {out / 'summary.json'}")
PY

echo ""
echo "OK: tract DER compare finished → $OUT"
