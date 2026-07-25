#!/usr/bin/env bash
# Fast full-DER remainder: CoreML + 4-way file shards for Vox v2+VBx, then AMI.
# Legacy Vox 232 already done (CPU). Gate still uses no-collar micro DER.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
OUT="benchmarks/results/full-der-2026-07-25"
LOG="$OUT/fast-run.log"
BIN="./target/release/polyvoice-bench"
export POLYVOICE_VBX_PLDA_DIR="$ROOT/data/vbx-plda"
N_SHARDS=4
# CoreML on Apple Silicon — accuracy gate (DER) is EP-stable; was forcing cpu and
# serial which made the suite multi-day wall time.
EP=coreml

exec > >(tee -a "$LOG") 2>&1
echo "==== FAST SUITE START $(date -u +%Y-%m-%dT%H:%M:%SZ) ep=$EP shards=$N_SHARDS ===="

make_shard() {
  local src_audio="$1" src_rttm="$2" ids_file="$3" dest="$4"
  rm -rf "$dest"
  mkdir -p "$dest/audio" "$dest/rttm"
  while read -r id; do
    [[ -z "$id" ]] && continue
    ln -sf "$src_audio/${id}.wav" "$dest/audio/${id}.wav"
    # rttm may be id.rttm
    if [[ -f "$src_rttm/${id}.rttm" ]]; then
      ln -sf "$src_rttm/${id}.rttm" "$dest/rttm/${id}.rttm"
    else
      # some layouts nest differently
      ln -sf "$src_rttm/${id}.rttm" "$dest/rttm/${id}.rttm" 2>/dev/null || true
    fi
  done <"$ids_file"
  echo "shard $dest files=$(ls "$dest/audio" | wc -l | tr -d ' ')"
}

# --- 1) V2+VBx Vox 232, parallel CoreML shards ---
echo "==== START v2-vbx-vox-232-fast $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
SHARD_ROOT="$OUT/shards/vox-v2"
rm -rf "$SHARD_ROOT"
mkdir -p "$SHARD_ROOT"
IDS="$SHARD_ROOT/all.ids"
ls data/voxconverse-test/audio/*.wav | xargs -n1 basename | sed 's/\.wav//' | sort >"$IDS"
# macOS BSD split has no -n l/N; use fixed lines (232/4=58).
N_FILES=$(wc -l <"$IDS" | tr -d ' ')
LINES_PER=$(( (N_FILES + N_SHARDS - 1) / N_SHARDS ))
split -l "$LINES_PER" -d -a 1 "$IDS" "$SHARD_ROOT/ids."

pids=()
for i in $(seq 0 $((N_SHARDS - 1))); do
  make_shard \
    "$ROOT/data/voxconverse-test/audio" \
    "$ROOT/data/voxconverse-test/rttm" \
    "$SHARD_ROOT/ids.$i" \
    "$SHARD_ROOT/ds$i"
  (
    echo "==== shard$i START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
    "$BIN" "$SHARD_ROOT/ds$i" \
      --pipeline v2 --clusterer vbx --profile balanced --collar 0.25 \
      --execution-provider "$EP" \
      --output "$OUT/shard-v2-vox-$i.json"
    echo "==== shard$i OK $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
  ) >"$OUT/shard-v2-vox-$i.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "==== FAIL shard pid=$pid ===="
    fail=1
  fi
done
# surface shard logs
for i in $(seq 0 $((N_SHARDS - 1))); do
  echo "----- shard $i tail -----"
  tail -5 "$OUT/shard-v2-vox-$i.log" || true
done
if [[ "$fail" -ne 0 ]]; then
  echo "==== FAIL v2-vbx-vox shards ===="
  exit 1
fi

python3 "$OUT/merge_shard_reports.py" \
  "$OUT/v2-vbx-voxconverse-test-232.json" \
  "$OUT"/shard-v2-vox-*.json
echo "==== OK v2-vbx-vox-232-fast $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

# --- 2) Legacy AMI 16 (CoreML for speed; legacy default is cpu but EP override ok for RT) ---
echo "==== START legacy-ami-16 $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
"$BIN" data/ami-test --pipeline legacy --profile balanced --collar 0.25 \
  --min-cluster-size 2 --execution-provider "$EP" \
  --output "$OUT/legacy-ami-test-16.json"
echo "==== OK legacy-ami-16 $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

# --- 3) V2+VBx AMI 16 ---
echo "==== START v2-vbx-ami-16 $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
"$BIN" data/ami-test --pipeline v2 --clusterer vbx --profile balanced --collar 0.25 \
  --execution-provider "$EP" \
  --output "$OUT/v2-vbx-ami-test-16.json"
echo "==== OK v2-vbx-ami-16 $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

echo "==== ALL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
bash "$OUT/write_verdict.sh"
echo "==== VERDICT WRITTEN ===="
