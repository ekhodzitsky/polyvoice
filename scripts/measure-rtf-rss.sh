#!/usr/bin/env bash
# Measure batch RTF (polyvoice-bench) and peak RSS (/usr/bin/time) for the
# production INT8 default (v2 + VBx + balanced profile).
#
# Usage:
#   bash scripts/measure-rtf-rss.sh
#   MAX_VOX=10 MAX_AMI=16 bash scripts/measure-rtf-rss.sh
#
# Writes under benchmarks/results/int8-rtf-rss-<date>/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATE="${DATE:-$(date +%Y-%m-%d)}"
OUT="${OUT:-benchmarks/results/int8-rtf-rss-${DATE}}"
mkdir -p "$OUT"

BENCH="${BENCH:-target/release/polyvoice-bench}"
CLI="${CLI:-target/release/polyvoice}"
MAX_VOX="${MAX_VOX:-10}"
MAX_AMI="${MAX_AMI:-16}"

if [[ ! -x "$BENCH" ]]; then
  echo "Building release binaries..."
  cargo build --release --features cli --bin polyvoice-bench --bin polyvoice
fi

COMMON=(--profile balanced --pipeline v2 --clusterer vbx --collar 0.0)

echo "== Host =="
uname -a | tee "$OUT/host.txt"
sysctl -n machdep.cpu.brand_string 2>/dev/null | tee -a "$OUT/host.txt" || true
echo "polyvoice: $($CLI --version 2>/dev/null || true)" | tee -a "$OUT/host.txt"

run_bench() {
  local name="$1" dataset="$2" maxf="$3"
  local json="$OUT/${name}.json"
  echo ""
  echo "== Bench: $name (max-files=$maxf) =="
  set +e
  "$BENCH" "$dataset" "${COMMON[@]}" --max-files "$maxf" --output "$json" 2>"$OUT/${name}.log"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "bench failed rc=$rc — see $OUT/${name}.log"
    tail -20 "$OUT/${name}.log" || true
    return "$rc"
  fi
  python3 - "$json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
# polyvoice-bench: rt_factor_avg = audio_secs / wall_secs = RTFx (higher = faster)
rtfx = d.get("rt_factor_avg")
rtf = (1.0 / float(rtfx)) if rtfx else None
files = d.get("files_processed")
audio = d.get("total_audio_secs")
print(f"  files={files} audio_s={audio}")
print(f"  RTFx={rtfx}  RTF={rtf}")
print(f"  DER0 micro={d.get('der_no_collar_micro')}  DER0.25 micro={d.get('der_collar_micro')}")
print(f"  profile={d.get('profile')} clusterer={d.get('clusterer')} ep={d.get('execution_provider')}")
PY
}

run_rss() {
  local name="$1" wav="$2"
  local log="$OUT/${name}-rss.txt"
  echo ""
  echo "== RSS: $name ($wav) =="
  if [[ ! -f "$wav" ]]; then
    echo "  missing wav, skip"
    return 0
  fi
  # macOS: /usr/bin/time -l ; Linux: /usr/bin/time -v
  set +e
  if /usr/bin/time -l true 2>/dev/null; then
    /usr/bin/time -l "$CLI" diarize "$wav" --profile balanced --quiet \
      --output "$OUT/${name}.rttm" >"$OUT/${name}-stdout.txt" 2>"$log"
  else
    /usr/bin/time -v "$CLI" diarize "$wav" --profile balanced --quiet \
      --output "$OUT/${name}.rttm" >"$OUT/${name}-stdout.txt" 2>"$log"
  fi
  set -e
  # Parse peak RSS
  python3 - "$log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
# macOS: "12345678  maximum resident set size" (bytes)
m = re.search(r"(\d+)\s+maximum resident set size", text)
if m:
    b = int(m.group(1))
    print(f"  peak RSS = {b/1024/1024:.1f} MiB  ({b} bytes)")
else:
    # Linux: Maximum resident set size (kbytes): N
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if m:
        kb = int(m.group(1))
        print(f"  peak RSS = {kb/1024:.1f} MiB  ({kb} KiB)")
    else:
        print("  (could not parse peak RSS — see log)")
        print(text[-800:])
PY
}

# --- RTF suite ---
if [[ -d data/ami-test-single ]]; then
  run_bench "ami-test-single" data/ami-test-single 1 || true
fi
if [[ -d data/ami-test ]]; then
  run_bench "ami-test-${MAX_AMI}" data/ami-test "$MAX_AMI" || true
fi
if [[ -d data/voxconverse-test ]]; then
  run_bench "voxconverse-test-${MAX_VOX}" data/voxconverse-test "$MAX_VOX" || true
fi

# --- RSS on representative files ---
if [[ -f data/ami-test-single/audio/EN2002a.Mix-Headset.wav ]]; then
  run_rss "ami-EN2002a" data/ami-test-single/audio/EN2002a.Mix-Headset.wav
fi
# longest-ish vox file if present
VOX_WAV=$(ls data/voxconverse-test/audio/*.wav 2>/dev/null | head -1 || true)
if [[ -n "${VOX_WAV:-}" ]]; then
  run_rss "vox-sample" "$VOX_WAV"
fi

# --- summary ---
python3 - "$OUT" <<'PY'
import json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for p in sorted(out.glob("*.json")):
    if p.name == "summary.json":
        continue
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    # rt_factor_avg is RTFx (audio/wall)
    rtfx = d.get("rt_factor_avg")
    if rtfx is None:
        continue
    rtfx = float(rtfx)
    per = d.get("per_file") or []
    audio = sum(float(x.get("audio_duration_secs") or 0) for x in per)
    wall = sum(float(x.get("runtime_secs") or 0) for x in per)
    rows.append({
        "run": p.stem,
        "files": d.get("files_processed"),
        "audio_secs": audio or None,
        "wall_secs": wall or None,
        "rtfx": rtfx,
        "rtf": 1.0 / max(rtfx, 1e-12),
        "der0_micro": d.get("der_no_collar_micro"),
        "der025_micro": d.get("der_collar_micro"),
        "profile": d.get("profile", "balanced"),
        "resolved_execution_provider": d.get("resolved_execution_provider"),
        "model_ids": [h.get("model_id") for h in (d.get("model_hashes") or [])],
        "stage_totals": d.get("stage_totals"),
        "crate_version": d.get("crate_version"),
    })
rss = []
for p in sorted(out.glob("*-rss.txt")):
    text = p.read_text(errors="replace")
    m = re.search(r"(\d+)\s+maximum resident set size", text)
    if m:
        rss.append({"run": p.stem.replace("-rss",""), "peak_rss_mib": int(m.group(1))/1024/1024})
    else:
        m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        if m:
            rss.append({"run": p.stem.replace("-rss",""), "peak_rss_mib": int(m.group(1))/1024})

summary = {"rtf_runs": rows, "rss_runs": rss}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("\n== SUMMARY ==")
print(json.dumps(summary, indent=2))
print(f"\nWrote {out}/summary.json")
PY

echo "Done → $OUT"
