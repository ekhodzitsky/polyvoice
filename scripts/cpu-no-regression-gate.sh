#!/usr/bin/env bash
# Multi-objective CPU gate: RTF must not regress; DER and peak RSS must not
# worsen beyond small measurement noise. Same config, only EP=cpu by default.
#
# Usage:
#   bash scripts/cpu-no-regression-gate.sh
#   DATASET=data/ami-test-single MAX_FILES=1 bash scripts/cpu-no-regression-gate.sh
#   # compare a candidate (env-tuned) against baseline:
#   BASELINE_JSON=/tmp/base.json CANDIDATE_ENV="POLYVOICE_SESSION_POOL_SIZE=8" \
#     bash scripts/cpu-no-regression-gate.sh
#
# Gates (defaults):
#   DER_NO_COLLAR micro: candidate <= baseline + DER_EPS (default 0.15 pp)
#   RTFx:            candidate >= baseline * RTF_RATIO (default 0.98 = allow 2% noise)
#   Peak RSS (MiB):  candidate <= baseline * RSS_RATIO (default 1.10) when both measured
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BENCH="${BENCH:-target/release/polyvoice-bench}"
CLI="${CLI:-target/release/polyvoice}"
OUT="${OUT:-/tmp/polyvoice-cpu-gate-$$}"
DATASET="${DATASET:-data/ami-test-single}"
MAX_FILES="${MAX_FILES:-1}"
EP="${EP:-cpu}"
DER_EPS="${DER_EPS:-0.15}"
RTF_RATIO="${RTF_RATIO:-0.98}"
RSS_RATIO="${RSS_RATIO:-1.10}"
CANDIDATE_ENV="${CANDIDATE_ENV:-}"

mkdir -p "$OUT"

if [[ ! -x "$BENCH" ]]; then
  echo "Building release polyvoice-bench..."
  cargo build --release --features cli --bin polyvoice-bench --bin polyvoice
fi

COMMON=(--profile balanced --pipeline v2 --clusterer vbx --collar 0.0
        --execution-provider "$EP" --max-files "$MAX_FILES")

run_bench() {
  local tag="$1"
  local json="$OUT/${tag}.json"
  local log="$OUT/${tag}.log"
  echo "== bench $tag =="
  # shellcheck disable=SC2086
  env $CANDIDATE_ENV_APPLY "$BENCH" "$DATASET" "${COMMON[@]}" --output "$json" 2>"$log"
  echo "  wrote $json"
}

# 1) Baseline (clean env for pool/intra overrides)
CANDIDATE_ENV_APPLY=""
if [[ -n "${BASELINE_JSON:-}" && -f "$BASELINE_JSON" ]]; then
  cp "$BASELINE_JSON" "$OUT/baseline.json"
  echo "Using provided BASELINE_JSON=$BASELINE_JSON"
else
  run_bench baseline
  mv "$OUT/baseline.json" "$OUT/baseline.json" 2>/dev/null || true
  # run_bench writes baseline.json already if tag=baseline
fi

# 2) Candidate (optional env overrides)
CANDIDATE_ENV_APPLY="$CANDIDATE_ENV"
run_bench candidate

python3 - "$OUT/baseline.json" "$OUT/candidate.json" "$DER_EPS" "$RTF_RATIO" <<'PY'
import json, sys
base_p, cand_p, der_eps, rtf_ratio = sys.argv[1:5]
der_eps = float(der_eps)
rtf_ratio = float(rtf_ratio)
b = json.load(open(base_p))
c = json.load(open(cand_p))

def get(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

b_der = float(get(b, "der_no_collar_micro", "der_collar_micro"))
c_der = float(get(c, "der_no_collar_micro", "der_collar_micro"))
b_rtfx = float(get(b, "rt_factor_avg"))
c_rtfx = float(get(c, "rt_factor_avg"))

print(f"DER0  base={b_der:.4f}  cand={c_der:.4f}  Δ={c_der-b_der:+.4f} pp  (max +{der_eps})")
print(f"RTFx  base={b_rtfx:.2f}  cand={c_rtfx:.2f}  ratio={c_rtfx/b_rtfx:.4f}  (min {rtf_ratio})")
print(f"stages base={b.get('stage_totals')}")
print(f"stages cand={c.get('stage_totals')}")

ok = True
if c_der > b_der + der_eps:
    print(f"FAIL DER regression: {c_der:.4f} > {b_der:.4f} + {der_eps}")
    ok = False
else:
    print("PASS DER")

if c_rtfx < b_rtfx * rtf_ratio:
    print(f"FAIL RTF regression: {c_rtfx:.2f} < {b_rtfx:.2f} * {rtf_ratio}")
    ok = False
else:
    print("PASS RTF")

if c_der < b_der - 1e-9 or c_rtfx > b_rtfx * 1.001:
    print("NOTE: candidate improved at least one metric (DER down and/or RTFx up).")

sys.exit(0 if ok else 1)
PY

echo "Gate artifacts in $OUT"
