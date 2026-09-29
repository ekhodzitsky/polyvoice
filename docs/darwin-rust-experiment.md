# Experimental Apple Rust kernels

`experimental-darwin-rust` selects the existing Rust GEMM, quantized convolution
and LSTM implementations on Apple instead of Accelerate/BNNS. It is an
**unqualified experiment**, not a new product default. The feature does not
activate a pipeline by itself and has no backend effect on Linux or Windows.
Cargo features are additive: any consumer enabling it selects this backend for
Apple kernel users in the same resolved build. `--all-features` includes it.

```bash
cargo build --locked --release --no-default-features \
  --features pipeline-local,experimental-darwin-rust --example local_native
./target/release/examples/local_native /path/to/models /path/to/plda meeting.wav
```

Use `pipeline-local` for the dependency claim: adding `cli` or downloads also
adds TLS/crypto dependencies, including native crypto compilation. The selected
local-model graph still has external Rust crates. This change adds none: it
reuses `rten-gemm`, `rten-tensor`, in-crate SIMD/INT8 kernels and the existing
weight loader. Existing kernels/dependencies contain Rust inline assembly;
this is not an assembly-free implementation. The `cc` build dependency and its
Rust helper crates remain in the graph, but the experimental Apple branch does
not invoke them to compile C sources. The standard platform runtime, Rust
linker driver and OS APIs remain necessary.

## Build and linkage evidence

The backend decision is emitted once by the kernel build script as
`apple_accelerate`. Default Apple builds keep the C shims and framework link;
the experimental feature removes those modules/links and enables Rust paths.
Apple INT8 convolution then uses the existing SDOT path by default. BNNS
profiling counters remain available and return zero for the Rust backend.

From Linux, the Apple ARM64 type/compile check is reproducible without a C
compiler or Apple SDK:

```bash
rustup target add aarch64-apple-darwin
CC=false CXX=false cargo check --locked --target aarch64-apple-darwin \
  --no-default-features --features pipeline-local,experimental-darwin-rust \
  --example local_native
```

Without the experimental feature, the same command fails while trying to
compile `bnns_conv.c`. Cross-checking does not prove final Mach-O linkage or
runtime behavior. The `darwin-rust-kernels` CI job additionally runs kernel
numerical tests and the following audit on a real Darwin ARM64 runner:

```bash
bash scripts/check-darwin-rust.sh
```

It uses a fresh build directory, disables C/C++ source compilers, builds a
local-model release executable, checks `otool -L` and undefined symbols for
Accelerate/BNNS/BLAS, then runs real audio using hash-verified local models.
Artifacts retain source revision, model/binary hashes, host, command, turns,
linkage and dependency graph. Shared CI proves build/linkage/smoke behavior;
it cannot certify the isolated M1 Pro speed or memory floors.

Linux numerical tests and local inference exercise reused code, not Apple
performance. A same-host M1 Pro comparison is now available below; it does
not qualify the experimental backend for promotion.

## Measured M1 Pro comparison

The [retained reports](../benchmarks/results/darwin-rust-comparison-2026-09-29/README.md)
compare both backends at clean revision
`0ed55f75df9fe2ba504c5e4c5ab1877de4e90f39`, using Rust 1.98.0 on a
10-core Apple M1 Pro with 16 GiB RAM, Darwin 25.1.0. Models, all six PLDA
files and the Vox-3 inputs were hash-verified. Separate release builds used
the same compiler and protocol. One warm-up per backend preceded three
paired measurements with alternating order; every result is retained.

| Metric | Product Accelerate/BNNS | Experimental Rust | Locked limit |
|---|---:|---:|---:|
| DER micro, all runs | 7.10884% | 7.00680% | ≤7.11% |
| DER macro, all runs | 7.38894% | 7.32948% | ≤7.39% |
| RTFx, runs 1 / 2 / 3 | 161.108 / 154.523 / 163.693 | 115.756 / 118.864 / 115.722 | ≥117 |
| Peak RSS, runs 1 / 2 / 3, MiB | 444.328 / 440.188 / 439.672 | 448.750 / 443.766 / 450.516 | ≤556 |
| INT8 pair bytes | 8,414,314 | 8,414,314 | ≤8,414,314 |

The product passed every floor in every run. Rust missed the speed floor in
two of three measured runs and in its warm-up (112.682×). Its lower DER on
these three files does not establish a full-corpus quality improvement.
The host was on AC power with low-power mode off and no reported thermal
warnings, but residual background agent/system activity remained after
foreground workloads were closed. These short runs are sensitive to that
noise; the result establishes an observed floor failure, not an intrinsic
maximum for the Rust kernels or an isolated-host release certification.

Stage timings point to embedding as the next profiling target: median
embedding time was 0.638 s for Rust versus 0.337 s for the product;
segmentation was 0.267 s versus 0.307 s. These are coarse pipeline timings,
not proof of a particular kernel bottleneck. Keep the backend experimental
and all limits unchanged. The smallest next step is to profile the Rust
embedding path and repeat the comparison on a quiescent host before proposing
any optimization or default change. No dependencies were added.

## Required comparison before any promotion

On the isolated Apple M1 Pro, use one clean source revision, identical cached
INT8 models and all six registry PLDA files, and the committed Vox-3 data.
Clear tuning overrides and ensure no other workload shares the host. Record
`git rev-parse HEAD`, `git status --porcelain`, `rustc -Vv`, `uname -a` and
`sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize`. Hash both ONNX files
and all PLDA files; confirm their registry hashes before measurement.

Build each variant into a separate directory to avoid accidentally measuring
the previous feature selection:

```bash
# On the isolated M1 Pro, after model/PLDA bootstrap into the default cache.
out="$PWD/bench-results/darwin-rust-comparison"
mkdir -p "$out"
for backend in product rust; do
  features=cli
  if [ "$backend" = rust ]; then
    features=cli,experimental-darwin-rust
  fi
  CARGO_TARGET_DIR="$out/$backend-target" cargo build --locked --release \
    --no-default-features --features "$features" --bin polyvoice-bench
  POLYVOICE_VBX_PLDA_DIR="$HOME/Library/Caches/polyvoice/models" \
    /usr/bin/time -l "$out/$backend-target/release/polyvoice-bench" \
    tests/data/native-vox3 --profile balanced --pipeline v2 --clusterer vbx \
    --execution-provider cpu --collar 0 --jobs 1 \
    --output "$out/$backend.json" > "$out/$backend.log" 2>&1
done
```

This uses the existing benchmark for both inference backends. The benchmark
CLI is download-enabled; its whole dependency graph is not the local-model
pure-Rust artifact audited above. Retain both JSON reports, full commands,
feature sets, build/binary/model/PLDA hashes and host metadata. Assert exact
`euqef`, `fuzfh`, `msbyq` coverage, zero skipped files, collar zero, overlap
scored and CPU resolution. Compare **each individual run** against every
unchanged [scoreboard floor](../tests/native_scoreboard.json):

| Characteristic | Required limit |
|---|---:|
| DER micro | ≤7.11% |
| DER macro | ≤7.39% |
| `rt_factor_avg` | ≥117× |
| Actual INT8 pair bytes | ≤8,414,314 |
| Peak process RSS (`time -l` bytes / 1,048,576) | ≤556 MiB |

A speed win cannot compensate for worse accuracy or memory, and a smaller
source dependency surface cannot compensate for any regression. Keep failures
as experimental evidence; do not tune the test set, weaken limits, mix metrics
from different runs or silently switch the product backend. This experiment
is not accepted by the [product release evidence gate](release-quality.md).
The measured comparison above does not meet the promotion requirement.
Any future candidate needs a fresh, quiescent same-host comparison that
passes every floor; performance qualification and promotion remain open.
