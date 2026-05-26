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
cargo test --all-features

echo "=== 5. DER regression — legacy e2e_smoke ==="
cargo test --test der_regression_test --features "onnx,download" der_regression_e2e_smoke -- --ignored --nocapture

echo "=== 6. DER regression — legacy VoxConverse 10-file ==="
cargo test --test der_regression_test --features "onnx,download" der_regression_voxconverse_10_file_subset -- --ignored --nocapture

echo "=== 7. DER regression — legacy AMI single ==="
cargo test --test der_regression_test --features "onnx,download" der_regression_ami_test_single -- --ignored --nocapture

echo "=== 8. DER regression — pipeline v2 e2e_smoke ==="
cargo test --test pipeline_v2_integration --features "onnx,segmentation,embedder,clusterer,resegmentation,download" -- --ignored --nocapture

echo ""
echo "=== ALL CHECKS PASSED ==="
echo "Safe to publish: cargo publish"
