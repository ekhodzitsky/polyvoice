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
performance. No same-host isolated Apple M1 Pro comparison is available yet.

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
The next decision needs the isolated same-host comparison; until then
performance feasibility and promotion remain open.
