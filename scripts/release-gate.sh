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
step "DER VoxConverse Mobile ≤ 12.5%" pending "becomes real in M5 (INT8 calibration)"
step "DER VoxConverse Balanced ≤ 11.5%" pending "becomes real in M4 (resegmenter)"
step "DER AMI Mobile ≤ 19.5%" pending "becomes real in M5"
step "DER AMI Balanced ≤ 18.5%" pending "becomes real in M4"

echo ""
echo "[2/12] Model bundle sizes"
step "Mobile bundle ≤ 10 MB" pending "real INT8 weights ship in M5"
step "Balanced bundle ≤ 35 MB" pending "real INT8 weights ship in M5"

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
