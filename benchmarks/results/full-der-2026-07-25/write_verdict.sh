#!/usr/bin/env bash
# Generate VERDICT.md from the four full-DER JSON result files.
# Gate: v2+VBx no-collar micro DER ≤ legacy on VoxConverse-test AND AMI-test.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/VERDICT.md"

need=(
  legacy-voxconverse-test-232.json
  v2-vbx-voxconverse-test-232.json
  legacy-ami-test-16.json
  v2-vbx-ami-test-16.json
)
for f in "${need[@]}"; do
  if [[ ! -f "$DIR/$f" ]]; then
    echo "missing $f — run not finished" >&2
    exit 1
  fi
done

extract() {
  local file="$1"
  jaq -r --arg k "$2" '.[$k]' "$file"
}

fmt_pct() {
  # print as XX.XXX%
  awk -v v="$1" 'BEGIN { printf "%.3f", v }'
}

LV_N=$(extract "$DIR/legacy-voxconverse-test-232.json" der_no_collar_micro)
VV_N=$(extract "$DIR/v2-vbx-voxconverse-test-232.json" der_no_collar_micro)
LV_C=$(extract "$DIR/legacy-voxconverse-test-232.json" der_collar_micro)
VV_C=$(extract "$DIR/v2-vbx-voxconverse-test-232.json" der_collar_micro)
LA_N=$(extract "$DIR/legacy-ami-test-16.json" der_no_collar_micro)
VA_N=$(extract "$DIR/v2-vbx-ami-test-16.json" der_no_collar_micro)
LA_C=$(extract "$DIR/legacy-ami-test-16.json" der_collar_micro)
VA_C=$(extract "$DIR/v2-vbx-ami-test-16.json" der_collar_micro)

LV_MISS=$(extract "$DIR/legacy-voxconverse-test-232.json" miss)
VV_MISS=$(extract "$DIR/v2-vbx-voxconverse-test-232.json" miss)
LV_FA=$(extract "$DIR/legacy-voxconverse-test-232.json" false_alarm)
VV_FA=$(extract "$DIR/v2-vbx-voxconverse-test-232.json" false_alarm)
LV_CONF=$(extract "$DIR/legacy-voxconverse-test-232.json" confusion)
VV_CONF=$(extract "$DIR/v2-vbx-voxconverse-test-232.json" confusion)
LA_MISS=$(extract "$DIR/legacy-ami-test-16.json" miss)
VA_MISS=$(extract "$DIR/v2-vbx-ami-test-16.json" miss)
LA_FA=$(extract "$DIR/legacy-ami-test-16.json" false_alarm)
VA_FA=$(extract "$DIR/v2-vbx-ami-test-16.json" false_alarm)
LA_CONF=$(extract "$DIR/legacy-ami-test-16.json" confusion)
VA_CONF=$(extract "$DIR/v2-vbx-ami-test-16.json" confusion)

LV_RT=$(extract "$DIR/legacy-voxconverse-test-232.json" rt_factor_avg)
VV_RT=$(extract "$DIR/v2-vbx-voxconverse-test-232.json" rt_factor_avg)
LA_RT=$(extract "$DIR/legacy-ami-test-16.json" rt_factor_avg)
VA_RT=$(extract "$DIR/v2-vbx-ami-test-16.json" rt_factor_avg)

GIT=$(extract "$DIR/legacy-voxconverse-test-232.json" git_sha)
HOST=$(jaq -r '"\(.host_os) \(.host_arch) cpus=\(.host_cpus)"' \
  "$DIR/legacy-voxconverse-test-232.json")

# Gate: v2 ≤ legacy on no-collar micro for both datasets
vox_ok=$(awk -v a="$VV_N" -v b="$LV_N" 'BEGIN { print (a <= b) ? 1 : 0 }')
ami_ok=$(awk -v a="$VA_N" -v b="$LA_N" 'BEGIN { print (a <= b) ? 1 : 0 }')

if [[ "$vox_ok" -eq 1 && "$ami_ok" -eq 1 ]]; then
  DECISION="GO — promote pipeline v2 + VBx to default"
  FLIP="yes"
else
  DECISION="NO-GO — keep legacy as default; document gaps below"
  FLIP="no"
fi

vox_delta=$(awk -v a="$VV_N" -v b="$LV_N" 'BEGIN { printf "%+.3f", a - b }')
ami_delta=$(awk -v a="$VA_N" -v b="$LA_N" 'BEGIN { printf "%+.3f", a - b }')

cat > "$OUT" <<EOF
# Full DER gate verdict — legacy vs v2+VBx

**Date:** 2026-07-25  
**Git:** \`$GIT\`  
**Host:** $HOST  
**Protocol:** collar 0.25 scored + no-collar; overlap scored; Hungarian; profile balanced; EP CPU; PLDA \`data/vbx-plda\`  
**Legacy knobs:** min_cluster_size=2  
**V2 knobs:** \`--pipeline v2 --clusterer vbx\`

## Decision

**$DECISION**

Gate rule (hard): flip only if v2+VBx **no-collar micro DER ≤ legacy** on both VoxConverse-test and AMI-test.

| Dataset | legacy no-collar micro | v2+VBx no-collar micro | Δ (v2 − legacy) | Pass |
|---------|------------------------:|------------------------:|----------------:|:----:|
| VoxConverse-test (232) | $(fmt_pct "$LV_N")% | $(fmt_pct "$VV_N")% | ${vox_delta} pp | $([[ "$vox_ok" -eq 1 ]] && echo PASS || echo FAIL) |
| AMI-test (16) | $(fmt_pct "$LA_N")% | $(fmt_pct "$VA_N")% | ${ami_delta} pp | $([[ "$ami_ok" -eq 1 ]] && echo PASS || echo FAIL) |

**Flip default?** \`$FLIP\`

## Full metrics

### VoxConverse-test (232 files)

| Metric | legacy | v2+VBx |
|--------|-------:|-------:|
| DER no-collar micro | $(fmt_pct "$LV_N")% | $(fmt_pct "$VV_N")% |
| DER collar 0.25 micro | $(fmt_pct "$LV_C")% | $(fmt_pct "$VV_C")% |
| Miss | $(fmt_pct "$LV_MISS")% | $(fmt_pct "$VV_MISS")% |
| FA | $(fmt_pct "$LV_FA")% | $(fmt_pct "$VV_FA")% |
| Confusion | $(fmt_pct "$LV_CONF")% | $(fmt_pct "$VV_CONF")% |
| RT factor (avg) | $(fmt_pct "$LV_RT")× | $(fmt_pct "$VV_RT")× |

### AMI-test (16 files)

| Metric | legacy | v2+VBx |
|--------|-------:|-------:|
| DER no-collar micro | $(fmt_pct "$LA_N")% | $(fmt_pct "$VA_N")% |
| DER collar 0.25 micro | $(fmt_pct "$LA_C")% | $(fmt_pct "$VA_C")% |
| Miss | $(fmt_pct "$LA_MISS")% | $(fmt_pct "$VA_MISS")% |
| FA | $(fmt_pct "$LA_FA")% | $(fmt_pct "$VA_FA")% |
| Confusion | $(fmt_pct "$LA_CONF")% | $(fmt_pct "$VA_CONF")% |
| RT factor (avg) | $(fmt_pct "$LA_RT")× | $(fmt_pct "$VA_RT")× |

## Artifacts

- \`legacy-voxconverse-test-232.json\`
- \`v2-vbx-voxconverse-test-232.json\`
- \`legacy-ami-test-16.json\`
- \`v2-vbx-ami-test-16.json\`
- \`full-run.log\`
- \`smoke-legacy.json\` / \`smoke-v2-vbx.json\` (1-file smoke)

## Next steps

$(if [[ "$FLIP" == "yes" ]]; then
  cat <<'NEXT'
1. Flip CLI/Python defaults to pipeline v2 + VBx (separate PR).
2. Refresh `tests/der_baseline.json` from these aggregates.
3. Update `docs/BENCHMARKS.md` and production-readiness notes.
NEXT
else
  cat <<'NEXT'
1. Do **not** flip defaults.
2. Investigate the larger gap (usually confusion / over-cluster) on the failing set(s).
3. Keep dual path until a later gate passes.
NEXT
fi)
EOF

echo "Wrote $OUT"
echo "DECISION=$DECISION"
echo "vox: legacy=$(fmt_pct "$LV_N")% v2=$(fmt_pct "$VV_N")% delta=${vox_delta}pp pass=$vox_ok"
echo "ami: legacy=$(fmt_pct "$LA_N")% v2=$(fmt_pct "$VA_N")% delta=${ami_delta}pp pass=$ami_ok"
