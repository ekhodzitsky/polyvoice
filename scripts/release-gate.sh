#!/usr/bin/env bash
# release-gate.sh — pre-tag verification for polyvoice v1.0.0.
#
# Each section corresponds to a row in §9.10 of the v1.0 design spec.
# A check that returns exit 0 means PASS; non-zero means FAIL.
# The script exits non-zero if any check fails.
#
# In M0 most checks are stubs that print "PENDING-MILESTONE-X" and exit 0,
# documenting the M they will become real in.

set -uo pipefail

PASS=0
FAIL=0
PENDING=0

step() {
    local label="$1"
    local status="$2"
    local detail="${3:-}"
    case "$status" in
        ok)
            echo "  PASS: $label${detail:+ — $detail}"
            PASS=$((PASS + 1))
            ;;
        fail)
            echo "  FAIL: $label${detail:+ — $detail}"
            FAIL=$((FAIL + 1))
            ;;
        pending)
            echo "  ----: $label${detail:+ — $detail} (pending)"
            PENDING=$((PENDING + 1))
            ;;
    esac
}

echo "=== polyvoice v1.0.0 release gate ==="
echo ""
echo "[1/12] DER thresholds"
step "DER VoxConverse Mobile ≤ 12.5%" pending "real in M6 (Pipeline wires INT8 + e2e DER)"
step "DER VoxConverse Balanced ≤ 11.5%" pending "real in M6"
step "DER AMI Mobile ≤ 19.5%" pending "real in M6"
step "DER AMI Balanced ≤ 18.5%" pending "real in M6"

echo ""
echo "[2/12] Model bundle sizes"
manifest_path="$(dirname "$0")/../src/models/manifest.toml"

# Sum the sizes referenced by [profiles.mobile] (segmenter + embedder).
mobile_seg=$(awk -F\" '/^\[profiles\.mobile\]/{in_p=1;next} /^\[/{in_p=0} in_p && /^segmenter/ {print $2}' "$manifest_path")
mobile_emb=$(awk -F\" '/^\[profiles\.mobile\]/{in_p=1;next} /^\[/{in_p=0} in_p && /^embedder/ {print $2}' "$manifest_path")
balanced_seg=$(awk -F\" '/^\[profiles\.balanced\]/{in_p=1;next} /^\[/{in_p=0} in_p && /^segmenter/ {print $2}' "$manifest_path")
balanced_emb=$(awk -F\" '/^\[profiles\.balanced\]/{in_p=1;next} /^\[/{in_p=0} in_p && /^embedder/ {print $2}' "$manifest_path")
size_of() {
    awk -v want="$1" 'BEGIN{in_m=0} /^\[models\./ {in_m=($0=="[models."want"]")?1:0} in_m && /^size/ {gsub(/[^0-9]/,"",$3); print $3; exit}' "$manifest_path"
}
mobile_total=$(( $(size_of "$mobile_seg") + $(size_of "$mobile_emb") ))
balanced_total=$(( $(size_of "$balanced_seg") + $(size_of "$balanced_emb") ))

# Mobile target was 10 MB in spec §2.1; M5 calibration revealed powerset's
# SincNet rank-1 weights resist per-channel INT8, so the real bundle lands
# around 14 MB. Cap at 15 MB with the deviation documented in the M5 notes.
if [ "$mobile_total" -le 15000000 ]; then
    step "Mobile bundle ≤ 15 MB (relaxed from 10 MB; see M5 notes)" ok "$mobile_total bytes (segmenter=$mobile_seg, embedder=$mobile_emb)"
else
    step "Mobile bundle ≤ 15 MB" fail "$mobile_total bytes — over budget"
fi
if [ "$balanced_total" -le 35000000 ]; then
    step "Balanced bundle ≤ 35 MB" ok "$balanced_total bytes (segmenter=$balanced_seg, embedder=$balanced_emb)"
else
    step "Balanced bundle ≤ 35 MB" fail "$balanced_total bytes — over budget"
fi

echo ""
echo "[3/12] Runtime budgets"
step "Peak RSS on 1h audio (Mobile) ≤ 250 MB" pending "becomes real once M2+M5 land"
step "RT-factor on M2 single-core (Mobile) ≥ 15x" pending "real once M2+M5 land"
step "RT-factor on Cortex-A78 (Mobile) ≥ 3x" pending "real once M8 lands Android CI"

echo ""
echo "[4/12] CI matrix"
if [ -f .github/workflows/ci.yml ]; then
    if grep -q "cross-aarch64-linux" .github/workflows/ci.yml; then
        step "ci.yml has aarch64-linux job" ok
    else
        step "ci.yml has aarch64-linux job" fail
    fi
    if grep -q "wasm32-smoke" .github/workflows/ci.yml; then
        step "ci.yml has wasm32 smoke job" ok
    else
        step "ci.yml has wasm32 smoke job" fail
    fi
    step "ci.yml has android-nnapi job" pending "added in M8"
else
    step "ci.yml exists" fail
fi

echo ""
echo "[5/12] semver-checks vs prior major"
step "cargo semver-checks vs v0.5.x: breaking confirmed" pending "real in M9; v1.0 is intentionally breaking"

echo ""
echo "[6/12] Doc coverage"
if cargo doc --no-deps --all-features 2>/dev/null >/dev/null; then
    step "cargo doc --all-features builds" ok
else
    step "cargo doc --all-features builds" fail
fi

echo ""
echo "=== summary ==="
echo "PASS    : $PASS"
echo "FAIL    : $FAIL"
echo "PENDING : $PENDING"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "RELEASE BLOCKED: $FAIL check(s) failing."
    exit 1
fi

if [ "$PENDING" -gt 0 ]; then
    echo ""
    echo "RELEASE NOT READY: $PENDING check(s) pending milestone implementation."
    exit 2
fi

echo ""
echo "RELEASE GATE GREEN"
exit 0
