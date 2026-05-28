#!/usr/bin/env bash
# Release gate: must pass before `cargo publish`.
# Fails fast on any check so broken code never reaches crates.io.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== 1. Format check ==="
cargo fmt -- --check

echo "=== 2. Clippy ==="
cargo clippy --all-targets --all-features -- -D warnings

echo "=== 3. Doc ==="
cargo doc --no-deps --all-features

echo "=== 4. Unit + integration tests (fast) ==="
cargo nextest run --profile ci --all-features

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
