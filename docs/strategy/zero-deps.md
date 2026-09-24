# Dependency reduction toward pure Rust

**Updated: 2026-09-24.** Native batch diarization is the product default;
ONNX Runtime is absent from the core crate. Dependency reduction continues,
but the product is not literally zero-dependency or entirely pure Rust.

## Terms and the 1.0 boundary

| Claim | Meaning | Current status |
|-------|---------|----------------|
| No ONNX Runtime | No `ort` / `ort-sys` or `libonnxruntime` in the selected product graph | Core diarization meets this; companion ASR is separate |
| Pure Rust implementation | Selected implementation and transitive dependencies do not compile/link C/C++ or assembly components | BYO algorithm core qualifies; full native/download-enabled product does not |
| No native build/runtime dependencies | No extra native toolchain components or linked native libraries beyond the documented platform runtime | Not a blanket product claim: Darwin C shims/Accelerate, opt-in Linux OpenBLAS and downloader crypto matter |
| Zero external Rust crates | No third-party Cargo dependencies | Not achieved, including with empty default features |

Pure Rust does not mean a static executable, absence of libc, or freedom from
OS/ABI requirements. Distribution claims must name the feature set, target
and linked libraries. GNU Linux artifacts still have libc requirements.

For 1.0, enforce the no-ORT product graph, explicit reproducible backend
selection, and honest dependency disclosure. Literal zero crates and
replacement of Darwin Accelerate/BNNS are longer-term work; they do not block
the [batch release contract](../../PRODUCTION-READINESS.md).

## Current feature and platform matrix

| Surface | Dependency behavior |
|---------|---------------------|
| `default = []` | BYO embedder, Energy VAD and algorithm core. Rust dependencies remain (`ndarray`, `realfft`, `serde`, etc.); no model engine/downloads |
| `clusterer`, `vbx`, `vad-earshot` | Optional Rust algorithms; local VBx assets do not require downloading |
| `pipeline-local` | Native v2 + VBx with local assets; no `ureq`, `rustls` or `ring` in its graph. Still includes Rust crates and platform kernel dependencies |
| `pipeline-native,vbx` | Native product plus downloader/model registry |
| `cli`, `ffi`, Python | Native v2 + VBx, download-enabled; no ORT or tract. Python adds its interpreter/ABI |
| `mcp` | Same native engine and downloads, plus experimental protocol/server dependencies |
| `backend-tract` | Opt-in Rust ONNX engine; no ORT. This does not certify every surrounding feature as pure Rust |
| `pipeline-tract`, `cli-tract` | Experimental tract pipeline plus downloader/TLS; not a fully pure-Rust dependency graph |
| `polyvoice-asr` | Independent Parakeet companion; still uses ONNX Runtime |

Native kernels have platform-specific implementations:

- **Darwin:** C shims compiled by `cc`, linked to system Accelerate/BNNS.
  No ONNX Runtime, but not a pure-Rust inference stack.
- **Linux:** `rten-gemm` and in-crate kernels, with no system BLAS by default.
  `system-openblas` explicitly enables LP64 OpenBLAS via `pkg-config`; missing
  prerequisites fail the build.
- **Windows:** Rust kernel path; platform/runtime linking and download crypto
  still need to be considered for the complete artifact.

The kernel crate uses `cc` for Darwin shims; `pkg-config` is an optional build
dependency enabled by `system-openblas`.
Download-enabled features add `ureq` → `rustls` → `ring`, which includes native
crypto code/toolchain requirements. Removing HTTP/TLS via `pipeline-local`
does not remove Darwin's C shims or system frameworks.

## Ordered work

1. Preserve the native product's no-ORT/no-tract graph and the downloader-free
   local graph. Do not reintroduce a generic ML runtime to reduce one kernel.
2. Keep Linux backend selection explicit; test Rust kernels and opt-in
   OpenBLAS independently, including actual artifact linkage.
3. Reduce avoidable dependencies using existing code or the standard library;
   account for build dependencies and transitive costs as well as direct crates.
4. Evaluate a fully Rust Darwin kernel path against the same accuracy, speed,
   model-size and RSS floors. A slower or larger path is not a product replacement.
5. Evaluate download crypto/toolchain alternatives separately. Keep model
   signature verification and trust-boundary validation intact.

Implement only the operations used by powerset and ResNet34; do not clone a
general ONNX executor. No silent quality regression to claim fewer dependencies.
The [locked scoreboard](../../tests/native_scoreboard.json) remains binding.

## Checks and their limits

```bash
# Existing dependency invariants (normal Cargo edges):
bash scripts/check-zero-deps.sh

# Inspect normal AND build edges for the exact selected target:
cargo tree --locked --no-default-features --features pipeline-local -e normal,build
cargo tree --locked --no-default-features --features cli -e normal,build

# Downloader-free consumer (requires local models and PLDA assets):
cargo run --no-default-features --features pipeline-local --example local_native -- \
  models/int8 fixtures/vbx-plda audio.wav

# Product CLI:
cargo run --release --features cli --bin polyvoice -- diarize meeting.wav
```

The dependency script checks package presence/absence; it does not prove
zero crates, absence of native code, or absence of system dylibs. Its
informational summary is not a linker audit. Inspect target-specific build
scripts and the actual release artifact's imports as well as Cargo graphs.

## Experimental tract evidence

Tract is not the product default or a prerequisite for native 1.0. Its
powerset path needs a rewritten graph and uses FP32 ResNet because the
measured INT8 embedder path collapsed speakers. Historical results and model
preparation are retained in:

- [Powerset export notes](../../benchmarks/results/powerset-tract-export-2026-08-12/NOTES.md)
- [Tract runtime/accuracy notes](../../benchmarks/results/powerset-tract-rtf-der-2026-08-12/NOTES.md)
- [AMI evaluation](../../benchmarks/results/tract-der-ami-2026-08-13/NOTES.md)
- [Backend verdict](../../benchmarks/results/tract-backend-verdict.md)
- [Earshot measurements](../../benchmarks/results/earshot-vad-notes.md)
- [Library mode and feature inventory](../library-mode.md)

These reports describe measured revisions, not current release certification.

## Linux backend selection

`cli`, `cli-native`, `ffi` and `pipeline-local` default to Rust Linux kernels,
even when OpenBLAS is installed. Release CLI assets, Python wheels and the
scheduled Linux quality gate use this default. Darwin keeps Accelerate/BNNS.

To opt into Linux system OpenBLAS:

```bash
# Debian/Ubuntu prerequisites (LP64, not openblas64 / ILP64):
sudo apt-get install libopenblas-dev pkg-config
cargo build --release --features cli,system-openblas
# Direct kernel consumers: polyvoice-kernels feature system-openblas.
```

The feature forwards to enabled native kernels; it does not enable a pipeline
by itself. It has no backend effect on non-Linux targets. Cargo features are
additive: `--all-features` includes this opt-in and requires OpenBLAS on Linux.
For a custom install use `PKG_CONFIG_PATH`; for cross-compilation configure
pkg-config for the target libraries. Generic BLAS is not supported because
the adapter uses `openblas_set_num_threads` and 32-bit CBLAS integer arguments.
The optional backend retains Rust GEMM/INT8 routing where that already wins;
it enables OpenBLAS for the existing CBLAS branches, not every multiplication.
Floating-point differences can change clustering and DER; qualify the selected
backend on your audio rather than assuming bit-identical results.

CI runs `scripts/check-linux-blas.sh` with OpenBLAS installed for both builds:
kernel numerical tests, unavailable-pkg-config failure, and `readelf`/`ldd`
inspection of actual CLI artifacts. Release Rust artifacts must not import
BLAS/OpenBLAS. Benchmark reports must record `FEATURES`; use
`FEATURES=cli-native,system-openblas` only for explicitly labeled OpenBLAS
comparisons. Historical auto-detected builds need their original linkage
record before assigning them a backend label.

[Linux backend comparison](../../benchmarks/results/linux-blas-selection-2026-09-24/NOTES.md)
records fixed-subset quality, artifact hashes and the limits of the measured
throughput/RSS evidence.
