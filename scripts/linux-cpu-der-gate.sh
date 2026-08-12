#!/usr/bin/env bash
# Official Linux / CPU DER gate for the production INT8 path.
#
# Protocol (product truth for non-Apple deploys):
#   - profile balanced (powerset_int8 + resnet34_int8)
#   - pipeline v2 + VBx
#   - execution-provider cpu (never CoreML / auto)
#   - powerset micro-batch N=8 (product default; override POLYVOICE_POWERSET_BATCH_SIZE)
#   - collar 0.25 run (JSON also carries no-collar micro/macro)
#   - full splits when MAX_*=0: VoxConverse-test 232, AMI-test 16
#
# Hard fails when:
#   - required dataset/audio is missing
#   - files_processed != expected (baseline.files when MAX=0, else MAX_*)
#   - resolved EP is not Cpu
#   - (ASSERT_BASELINE=1, full run) DER₀ micro > baseline + tolerance
#   - model download fails
#
# Usage:
#   bash scripts/linux-cpu-der-gate.sh
#   OUT=benchmarks/results/linux-cpu-der-manual bash scripts/linux-cpu-der-gate.sh
#   MAX_VOX=10 MAX_AMI=16 bash scripts/linux-cpu-der-gate.sh   # smoke (AMI still gated)
#   DOCKER=1 bash scripts/linux-cpu-der-gate.sh
#   ASSERT_BASELINE=0 bash scripts/linux-cpu-der-gate.sh       # measure only
#
# Writes under OUT/:
#   host.txt, summary.json, NOTES.auto.md, gate-result.json,
#   voxconverse-test.json, ami-test.json, *.log
# Does NOT overwrite a hand-written NOTES.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATE="${DATE:-$(date +%Y-%m-%d)}"
OUT="${OUT:-benchmarks/results/linux-cpu-der-${DATE}}"
EP="${EP:-cpu}"
PROFILE="${PROFILE:-balanced}"
PIPELINE="${PIPELINE:-v2}"
CLUSTERER="${CLUSTERER:-vbx}"
# 0 = all files present in the split (must match baseline.files for full gate).
MAX_VOX="${MAX_VOX:-0}"
MAX_AMI="${MAX_AMI:-0}"
ASSERT_BASELINE="${ASSERT_BASELINE:-1}"
BASELINE_JSON="${BASELINE_JSON:-$ROOT/tests/der_baseline.json}"
export POLYVOICE_POWERSET_BATCH_SIZE="${POLYVOICE_POWERSET_BATCH_SIZE:-8}"
export POLYVOICE_VBX_PLDA_DIR="${POLYVOICE_VBX_PLDA_DIR:-$ROOT/fixtures/vbx-plda}"

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
  MODEL_CACHE="${MODEL_CACHE:-$ROOT/.cache/polyvoice-models}"
  mkdir -p "$MODEL_CACHE"
  for f in powerset_int8.onnx resnet34_int8.onnx powerset_int8.onnx.minisig resnet34_int8.onnx.minisig; do
    if [[ -f "models/int8/$f" && ! -f "$MODEL_CACHE/$f" ]]; then
      cp "models/int8/$f" "$MODEL_CACHE/$f"
    fi
  done
  # Ubuntu 24.04: ort download-binaries need glibc ≥ ~2.38.
  IMG="${DOCKER_IMAGE:-ubuntu:24.04}"
  echo "=== Docker Linux gate (image=$IMG) ==="
  exec docker run --rm \
    -e DATE \
    -e OUT="/work/$OUT" \
    -e EP \
    -e PROFILE \
    -e PIPELINE \
    -e CLUSTERER \
    -e MAX_VOX \
    -e MAX_AMI \
    -e ASSERT_BASELINE \
    -e BASELINE_JSON=/work/tests/der_baseline.json \
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
GATE_FAIL=0

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
  echo "assert_baseline: $ASSERT_BASELINE"
  echo "max_vox: $MAX_VOX"
  echo "max_ami: $MAX_AMI"
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
} | tee "$OUT/host.txt"

# --- build ------------------------------------------------------------------
need_build=0
if [[ ! -x "$BENCH" ]]; then
  need_build=1
elif ! "$BENCH" --help >/dev/null 2>&1; then
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
if ! "$CLI" --version >/dev/null 2>&1; then
  echo "FATAL: polyvoice CLI not runnable: $CLI" >&2
  exit 1
fi
echo "polyvoice: $("$CLI" --version 2>/dev/null || true)" | tee -a "$OUT/host.txt"

echo "=== ensure INT8 models (profile=$PROFILE) ==="
"$CLI" download-models --profile "$PROFILE"

# --- runs -------------------------------------------------------------------
# run_split label dataset maxf baseline_key
# Writes json path to RUN_JSONS list via side file.
: >"$OUT/_runs.list"

run_split() {
  local label="$1" dataset="$2" maxf="$3" baseline_key="$4"
  if [[ ! -d "$dataset/audio" ]]; then
    echo "FATAL: required dataset missing: $dataset/audio" | tee -a "$OUT/run.log" >&2
    GATE_FAIL=1
    return 1
  fi
  local n
  n="$(find "$dataset/audio" -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$n" -eq 0 ]]; then
    echo "FATAL: no wav files under $dataset/audio" | tee -a "$OUT/run.log" >&2
    GATE_FAIL=1
    return 1
  fi

  local expected
  if [[ "$maxf" == "0" || -z "$maxf" ]]; then
    expected="$n"
  else
    expected="$maxf"
    if [[ "$n" -lt "$maxf" ]]; then
      echo "FATAL: $label needs max_files=$maxf but only $n wavs on disk" | tee -a "$OUT/run.log" >&2
      GATE_FAIL=1
      return 1
    fi
  fi

  local json="$OUT/${label}.json"
  local log="$OUT/${label}.log"
  local args=(--profile "$PROFILE" --pipeline "$PIPELINE" --clusterer "$CLUSTERER"
              --collar 0.25 --execution-provider "$EP" --output "$json")
  if [[ "$maxf" != "0" && -n "$maxf" ]]; then
    args+=(--max-files "$maxf")
  fi

  echo ""
  echo "=== $label (files_on_disk=$n expected=$expected max_files=${maxf:-all} ep=$EP batch=$POLYVOICE_POWERSET_BATCH_SIZE baseline_key=$baseline_key) ===" \
    | tee -a "$OUT/run.log"

  set +e
  "$BENCH" "$dataset" "${args[@]}" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAIL $label bench rc=$rc — tail $log:" | tee -a "$OUT/run.log" >&2
    tail -40 "$log" || true
    GATE_FAIL=1
    return 1
  fi
  if [[ ! -f "$json" ]]; then
    echo "FATAL: $label produced no JSON at $json" | tee -a "$OUT/run.log" >&2
    GATE_FAIL=1
    return 1
  fi

  python3 - "$json" "$expected" "$label" <<'PY' | tee -a "$OUT/run.log"
import json, sys
path, expected, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
r = json.load(open(path))
files = int(r.get("files_processed") or 0)
ep = r.get("resolved_execution_provider")
print(
    f"  files={files} (expected {expected})  EP={ep}\n"
    f"  DER0 micro={r.get('der_no_collar_micro'):.4f}  macro={r.get('der_no_collar_macro'):.4f}\n"
    f"  DER0.25 micro={r.get('der_collar_micro'):.4f}  macro={r.get('der_collar_macro'):.4f}\n"
    f"  miss/FA/conf @collar-request={r.get('miss'):.2f}/{r.get('false_alarm'):.2f}/{r.get('confusion'):.2f}\n"
    f"  RTFx={r.get('rt_factor_avg'):.2f}  stages={r.get('stage_totals')}"
)
errors = []
if files != expected:
    errors.append(f"files_processed={files} != expected={expected}")
if str(ep) != "Cpu":
    errors.append(f"resolved_execution_provider={ep!r} (want Cpu)")
if errors:
    print("  HARD FAIL: " + "; ".join(errors), file=sys.stderr)
    sys.exit(2)
PY
  local py_rc=$?
  if [[ $py_rc -ne 0 ]]; then
    GATE_FAIL=1
    return 1
  fi

  echo "${label}|${json}|${baseline_key}|${expected}|${maxf}" >>"$OUT/_runs.list"
}

run_split "voxconverse-test" "data/voxconverse-test" "$MAX_VOX" "voxconverse_test_linux_cpu" || true
run_split "ami-test" "data/ami-test" "$MAX_AMI" "ami_test_linux_cpu" || true

if [[ ! -s "$OUT/_runs.list" ]]; then
  echo "FATAL: no successful split runs" >&2
  exit 1
fi

# --- summary.json + baseline assert -----------------------------------------
python3 - "$OUT" "$BASELINE_JSON" "$ASSERT_BASELINE" "$POLYVOICE_POWERSET_BATCH_SIZE" "$PROFILE" "$EP" "$GATE_FAIL" <<'PY'
import json, pathlib, sys, platform, os

out = pathlib.Path(sys.argv[1])
baseline_path = pathlib.Path(sys.argv[2])
assert_baseline = sys.argv[3] == "1"
batch = int(sys.argv[4])
profile = sys.argv[5]
ep_req = sys.argv[6]
prior_fail = sys.argv[7] == "1"

baseline = {}
if baseline_path.is_file():
    baseline = json.loads(baseline_path.read_text())
else:
    print(f"WARN: baseline missing at {baseline_path}", file=sys.stderr)

summary = {
    "schema": "polyvoice-linux-cpu-der-gate-v1",
    "protocol": {
        "profile": profile,
        "pipeline": "v2+vbx",
        "execution_provider": ep_req,
        "powerset_batch": batch,
        "collar_request": 0.25,
        "note": "collar 0.25 JSON also carries no-collar micro/macro; miss/FA/conf are for the requested collar",
    },
    "host": {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    },
    "assert_baseline": assert_baseline,
    "runs": {},
    "gate": {"ok": True, "checks": []},
}

runs_list = (out / "_runs.list").read_text().strip().splitlines()
fail = False

for line in runs_list:
    label, json_path, bkey, expected_s, maxf_s = line.split("|")
    expected = int(expected_s)
    maxf = int(maxf_s) if maxf_s not in ("", "0") else 0
    r = json.loads(pathlib.Path(json_path).read_text())
    entry = {
        "files_processed": r.get("files_processed"),
        "resolved_execution_provider": r.get("resolved_execution_provider"),
        "der_no_collar_micro": r.get("der_no_collar_micro"),
        "der_no_collar_macro": r.get("der_no_collar_macro"),
        "der_collar_micro": r.get("der_collar_micro"),
        "der_collar_macro": r.get("der_collar_macro"),
        "miss": r.get("miss"),
        "false_alarm": r.get("false_alarm"),
        "confusion": r.get("confusion"),
        "miss_fa_conf_collar_secs": r.get("collar_secs"),
        "rt_factor_avg": r.get("rt_factor_avg"),
        "speaker_count": r.get("speaker_count"),
        "crate_version": r.get("crate_version"),
        "git_sha": r.get("git_sha"),
        "model_hashes": r.get("model_hashes"),
        "stage_totals": r.get("stage_totals"),
        "baseline_key": bkey,
        "max_files": maxf if maxf else None,
        "powerset_batch": batch,
    }
    summary["runs"][label] = entry

    def check(ok, msg):
        summary["gate"]["checks"].append({"split": label, "ok": ok, "msg": msg})
        print(("PASS" if ok else "FAIL") + f" [{label}] {msg}")
        return ok

    ep = r.get("resolved_execution_provider")
    if not check(ep == "Cpu", f"EP resolved={ep!r} want Cpu"):
        fail = True

    files = int(r.get("files_processed") or 0)
    if not check(files == expected, f"files_processed={files} want {expected}"):
        fail = True

    bl = baseline.get(bkey) or {}
    bl_files = bl.get("files")
    bl_der = bl.get("der_no_collar_micro")
    if bl_der is None:
        bl_der = bl.get("der_no_collar")
    tol = float(bl.get("tolerance") or 1.0)
    full_match = bl_files is not None and files == int(bl_files)

    if assert_baseline and full_match:
        if bl_der is None:
            if not check(False, f"baseline {bkey} missing der_no_collar*"):
                fail = True
        else:
            measured = float(r["der_no_collar_micro"])
            limit = float(bl_der) + tol
            ok = measured <= limit + 1e-9
            if not check(
                ok,
                f"DER0 micro={measured:.4f} <= baseline {float(bl_der):.4f} + tol {tol:.2f} (= {limit:.4f})",
            ):
                fail = True
    elif assert_baseline and not full_match:
        summary["gate"]["checks"].append(
            {
                "split": label,
                "ok": True,
                "msg": f"smoke: files={files} (baseline full={bl_files}); DER assert skipped",
            }
        )
        print(f"SKIP DER assert [{label}] smoke files={files} baseline_full={bl_files}")

if prior_fail:
    summary["gate"]["checks"].append(
        {
            "split": "*",
            "ok": False,
            "msg": "one or more splits failed before assert (see run.log)",
        }
    )
    fail = True
summary["gate"]["ok"] = not fail
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "gate-result.json").write_text(
    json.dumps(summary["gate"], indent=2) + "\n"
)
print(f"wrote {out / 'summary.json'}")
print(f"wrote {out / 'gate-result.json'} ok={summary['gate']['ok']}")
sys.exit(1 if fail else 0)
PY
assert_rc=$?
if [[ $assert_rc -ne 0 ]]; then
  GATE_FAIL=1
fi

# --- NOTES.auto.md (never clobber hand-written NOTES.md) --------------------
cat >"$OUT/NOTES.auto.md" <<EOF
# Linux CPU DER gate (auto)

**Date:** ${DATE}
**Protocol:** INT8 balanced, pipeline v2 + VBx, EP=\`${EP}\`, powerset micro-batch N=\`${POLYVOICE_POWERSET_BATCH_SIZE}\`.
**Assert baseline:** ${ASSERT_BASELINE} (\`${BASELINE_JSON}\`)
**Command:** \`bash scripts/linux-cpu-der-gate.sh\`

## Headline (no-collar micro)

See \`summary.json\` / per-split JSON. **miss/FA/conf** in reports are for the
**requested collar (0.25 s)**, not collar 0.

## Reproduce

\`\`\`bash
DOCKER=1 bash scripts/linux-cpu-der-gate.sh
# smoke:
MAX_VOX=10 MAX_AMI=16 bash scripts/linux-cpu-der-gate.sh
\`\`\`

Hand-written context (if present): \`NOTES.md\`.
EOF

rm -f "$OUT/_runs.list"

echo ""
if [[ "$GATE_FAIL" -ne 0 ]]; then
  echo "=== Linux CPU DER gate FAILED → $OUT ===" >&2
  if [[ -f "$OUT/gate-result.json" ]]; then
    cat "$OUT/gate-result.json" >&2 || true
  fi
  exit 1
fi

echo "=== Linux CPU DER gate PASSED → $OUT ==="
if command -v jq >/dev/null 2>&1 && [[ -f "$OUT/summary.json" ]]; then
  jq -r '.runs | to_entries[] | "\(.key): DER0=\(.value.der_no_collar_micro) DER0.25=\(.value.der_collar_micro) RTFx=\(.value.rt_factor_avg) files=\(.value.files_processed) EP=\(.value.resolved_execution_provider)"' "$OUT/summary.json"
fi
