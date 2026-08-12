#!/usr/bin/env bash
# Release gate: must pass before `cargo publish`.
# Fails fast on any check so broken code never reaches crates.io.
set -euo pipefail

cd "$(dirname "$0")/.."

# The DER regression steps below must actually run on real audio. Require the
# data to be present (and make the in-test guards hard-fail on absence) so a
# partial cache/download miss can never silently green-light a release.
export POLYVOICE_REQUIRE_DATA=1

# Default CLI path is v2 + VBx (0.11+). Point at the checked-in PLDA fixtures
# unless the caller already set POLYVOICE_VBX_PLDA_DIR (e.g. local override).
export POLYVOICE_VBX_PLDA_DIR="${POLYVOICE_VBX_PLDA_DIR:-$PWD/fixtures/vbx-plda}"
if [ ! -f "${POLYVOICE_VBX_PLDA_DIR}/plda_transform.npy" ]; then
  echo "FATAL: VBx PLDA fixtures missing at ${POLYVOICE_VBX_PLDA_DIR}" >&2
  echo "Expected the checked-in set under fixtures/vbx-plda/ (see README there)." >&2
  exit 1
fi

echo "=== 1. Format check ==="
cargo fmt --all -- --check

echo "=== 1b. Standalone lockfiles (fuzz / sherpa / python) ==="
# Catches path-dep version drift after a core bump without re-running
# scripts/bump-version.sh. Same gate as CI job standalone-lockfiles.
bash scripts/check-standalone-lockfiles.sh

echo "=== 1c. Zero-deps invariants (ort/earshot/tract stay opt-in) ==="
bash scripts/check-zero-deps.sh

echo "=== 2. Clippy (all-features + product front-door features) ==="
cargo clippy --all-targets --all-features -- -D warnings
# CI also gates onnx,ffi,cli without every optional EP; catch that shape too.
cargo clippy --all-targets --features onnx,ffi,cli -- -D warnings

echo "=== 2b. Supply-chain audit (advisories, licenses, bans, sources) ==="
# Block publish on any advisory/license/source/ban violation — run early so it
# fails fast, before the long DER regression steps. Reads the committed Cargo.lock.
cargo audit
cargo deny check

echo "=== 3. Doc ==="
cargo doc --no-deps --all-features

echo "=== 4. Unit + integration tests (fast) ==="
cargo nextest run --profile ci --all-features

echo "=== 4b. Required DER test data present ==="
require_file() {
  if [ ! -f "$1" ]; then
    echo "FATAL: required DER test data missing: $1" >&2
    exit 1
  fi
}
require_file "tests/data/e2e-smoke/audio/fuzfh.wav"
require_file "data/voxconverse-test/audio/aepyx.wav"
# AMI single uses either the .Mix-Headset or the bare filename.
if [ ! -f "data/ami-test-single/audio/EN2002a.Mix-Headset.wav" ] \
   && [ ! -f "data/ami-test-single/audio/EN2002a.wav" ]; then
  echo "FATAL: required DER test data missing: data/ami-test-single/audio/EN2002a(.Mix-Headset).wav" >&2
  exit 1
fi

echo "=== 5. DER regression — legacy e2e_smoke ==="
cargo nextest run --profile ci --run-ignored only --test der_regression_test --features "onnx,download" der_regression_e2e_smoke --nocapture

echo "=== 6. DER regression — legacy VoxConverse 10-file ==="
cargo nextest run --profile ci --run-ignored only --test der_regression_test --features "onnx,download" der_regression_voxconverse_10_file_subset --nocapture

echo "=== 7. DER regression — legacy AMI single ==="
cargo nextest run --profile ci --run-ignored only --test der_regression_test --features "onnx,download" der_regression_ami_test_single --nocapture

echo "=== 8. DER regression — pipeline v2 e2e_smoke (library API) ==="
cargo nextest run --profile ci --run-ignored only --test pipeline_v2_integration --features "onnx,segmentation,embedder,clusterer,resegmentation,download" --nocapture

echo "=== 9. DER regression — CLI pipeline v2 e2e_smoke ==="
cargo nextest run --profile ci --run-ignored only --test cli_der_regression_test --features "cli,download" cli_der_regression_v2_e2e_smoke --nocapture

echo "=== 10. DER regression — CLI pipeline v2 AMI single ==="
cargo nextest run --profile ci --run-ignored only --test cli_der_regression_test --features "cli,download" cli_der_regression_v2_ami_single --nocapture

echo ""
echo "=== ALL CHECKS PASSED ==="
echo "Safe to publish: cargo publish"
