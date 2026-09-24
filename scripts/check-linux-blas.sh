#!/usr/bin/env bash
# Linux backend contract. Requires LP64 OpenBLAS development files, pkg-config,
# readelf and ldd. Checks the actual CLI artifact as well as kernel arithmetic.
set -euo pipefail
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# No pkg-config executable is needed for the default Rust kernel build.
PKG_CONFIG=/bin/false cargo check --locked -p polyvoice-kernels

# An explicit backend must fail, rather than silently change implementation.
if PKG_CONFIG=/bin/false cargo check --locked -p polyvoice-kernels \
    --features system-openblas >"$work/missing.log" 2>&1; then
  echo 'FAIL: system-openblas succeeded without pkg-config' >&2
  exit 1
fi
if ! grep -q 'system-openblas requires LP64 OpenBLAS' "$work/missing.log"; then
  cat "$work/missing.log" >&2
  exit 1
fi

# OpenBLAS is installed for BOTH builds: its presence must not affect default.
pkg-config --exists openblas
for backend in rust openblas; do
  features=cli
  kernel_features=()
  if [[ "$backend" == openblas ]]; then
    features+=,system-openblas
    kernel_features=(--features system-openblas)
  fi
  cargo test --locked --release -p polyvoice-kernels "${kernel_features[@]}"
  cargo build --locked --release --no-default-features --features "$features" --bin polyvoice
  binary="${CARGO_TARGET_DIR:-target}/release/polyvoice"
  readelf -d "$binary" >"$work/dynamic.txt"
  ldd "$binary" >"$work/ldd.txt"
  cat "$work/dynamic.txt" "$work/ldd.txt" >"$work/linkage.txt"
  if [[ "$backend" == rust ]]; then
    if grep -Eiq 'blas|lapack' "$work/linkage.txt"; then
      cat "$work/linkage.txt" >&2
      echo 'FAIL: default artifact links BLAS' >&2
      exit 1
    fi
  else
    grep -q 'libopenblas' "$work/dynamic.txt"
    if grep -q 'not found' "$work/ldd.txt"; then
      cat "$work/ldd.txt" >&2
      exit 1
    fi
  fi
  "$binary" --version
  echo "OK: $backend kernel tests and CLI linkage"
done
