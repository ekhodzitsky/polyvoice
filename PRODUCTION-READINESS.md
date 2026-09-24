# Production readiness and the 1.0 contract

**Development version: 0.22.0 (unreleased). Updated: 2026-09-24.**
**1.0 status: NOT GO.** The contract below defines the intended release;
it does not certify the current revision or start the RC window.

## Supported product scope

The 1.0 product is **batch speaker diarization on CPU**, using powerset
segmentation, ResNet34 INT8 embeddings and VBx clustering (v2). The default
production model pair is `powerset_int8` + `resnet34_int8`. The result is
speaker turns, not speaker identity or transcription.

| Surface | Intended 1.0 contract |
|---------|-----------------------|
| Rust | Crate-root `Pipeline`, configuration, errors and result types; `pipeline-native,vbx` with model downloads, or `pipeline-local` with caller-provided local assets |
| CLI | `cli` uses the native v2 + VBx pipeline; documented flags, exit behavior and JSON output |
| Python | Installed `polyvoice` wheel, `Pipeline` and result API, using the same native engine |
| C FFI | Published header, ABI v3, lifecycle, status codes and input validation |
| JSON | Versioned [result schema](schema/diarization-result-v1.json) |

`--clusterer ahc` changes clustering within v2. It does not select the former
Silero pipeline. The product CLI rejects `--legacy`.

Existing BYO Rust API commitments remain: `LegacyPipeline`,
`StreamingPipeline`, `Embedder`, `EnergyVad` and local VBx loading stay within
the [advertised API policy](docs/semver.md). Their caller-supplied models and
streaming latency do not inherit the native batch accuracy/performance
claim. Native powerset streaming is not a 1.0 requirement.

| Adjacent surface | Release scope |
|------------------|---------------|
| MCP | Opt-in experimental protocol/tool surface. Uses native diarization, but its tool API is outside the 1.0 stability promise |
| tract and ONNX-file adapters | Experimental inference alternatives; not required for native product qualification |
| Companion ASR / Parakeet | Independent product and dependency lifecycle; its ONNX Runtime dependency does not block diarization 1.0 |
| Hosted services | Authentication, tenant isolation, scheduling, quotas and SLA are deployment responsibilities. Library 1.0 does not certify an unattended multi-tenant service |

## Dependency contract

**No ONNX Runtime** is already the core diarization contract. It does not
mean no C/C++ code, no system libraries, or zero Rust crates.

- Empty default features provide the BYO core with normal Rust dependencies.
- Native Darwin inference compiles C shims and links Accelerate/BNNS.
- Native Linux uses Rust kernels and currently auto-detects installed BLAS.
  Making that choice explicit and reproducible remains a release requirement.
- Download-enabled builds pull TLS dependencies, including `ring` native
  code. `pipeline-local` omits the downloader and its TLS graph.
- Python also depends on the Python runtime/ABI.

Literal zero external crates and fully pure-Rust Darwin inference are
long-term goals, not prerequisites for 1.0. Accurate dependency disclosure,
locked release inputs, and reproducible backend selection are prerequisites.
See the [dependency strategy](docs/strategy/zero-deps.md) for the exact terms
and feature/platform matrix.

## Platforms and required release evidence

The intended native target set follows the current CLI release matrix:

| Platform | Rust target | Required 1.0 evidence |
|----------|-------------|-----------------------|
| Linux x86_64 | `x86_64-unknown-linux-gnu` | Packaged Rust consumer, CLI and C consumer run on target; native model smoke and platform quality gate |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | Same checks on ARM64 hardware; cross-compilation alone is insufficient |
| macOS ARM64 | `aarch64-apple-darwin` | Same checks plus the locked Darwin scoreboard |
| Windows x86_64 | `x86_64-pc-windows-msvc` | Same checks with the shipped DLL/header and executable |

This table is an acceptance requirement, not a claim that every check is
already automated. Release metadata must record supported OS baselines,
CPU requirements, system libraries and compiler versions. macOS Intel and
other targets are not in this release matrix. wasm32 checks cover the
algorithm/BYO subset, not complete native diarization.

For each target, test an external Rust consumer against the packaged crate
with empty defaults, `pipeline-native,vbx`, and `pipeline-local`. Run a
consumer against the actual release CLI and C library/header, including
invalid input, errors, lifecycle and JSON compatibility. The local-assets
consumer must run without HTTP/TLS dependencies or network access. Source
tree tests alone are insufficient evidence for packaged artifacts.

Python qualification is per **wheel platform, interpreter and ABI**. The
current tag release workflow builds on Linux x86_64, macOS ARM64 and Windows
x86_64 using CPython 3.12; the separate wheel workflow also includes Linux
ARM64. Before advertising any wheel combination, install that artifact in a
clean environment and test import, real diarization, results and errors.
`requires-python >=3.9` is package metadata, not proof of wheel availability
or qualification for every newer interpreter. Publish the tested matrix
with each release; do not infer it from source-only Python CI.

## Quality and resource evidence

The [benchmark protocol](docs/BENCHMARKS.md) remains the source of truth for
accuracy. Existing measurements establish baselines, not certification of a
future release candidate:

| Native CPU measurement | VoxConverse-test DER₀ (232 files) | AMI-test DER₀ (16 files) | Evidence |
|------------------------|-----------------------------------|-------------------------|----------|
| Linux x86_64, 2026-09-13 | 13.34% | 24.19% | [Report](benchmarks/results/linux-cpu-native-der-2026-09-13-vbx-ahc/) |
| Darwin ARM64, 2026-09-22 | 13.33% | 23.61% | [Report](benchmarks/results/darwin-native-der-2026-09-22/) |

Release evidence must cover full VoxConverse-test and AMI-test plus at least
one additional licensed corpus or documented fixed subset. Publish file
lists, collar/overlap policy, aggregate metrics and predeclared regression
thresholds. A small smoke test does not replace full-split evaluation.
Each supported platform needs a native quality gate; full-split reference
runs remain required on Linux x86_64 and Darwin ARM64.

The Darwin Vox-3 protocol (euqef / fuzfh / msbyq, collar 0, balanced v2 + VBx,
INT8 pair) retains **all** floors from
[`tests/native_scoreboard.json`](tests/native_scoreboard.json):

| Characteristic | Release limit |
|----------------|---------------|
| DER₀ micro | ≤ 7.11% |
| DER₀ macro | ≤ 7.39% |
| Real-time factor | ≥ 117× |
| On-disk INT8 pair | ≤ 8,414,314 bytes |
| Peak process RSS | ≤ 556 MiB |

A speed gain cannot excuse a memory or accuracy regression. Timing and RSS
must use the reference host/protocol; these numbers are not portable speed
promises for arbitrary hardware. Record Linux performance separately on its
reference host, including jobs and wall/per-file timing.

Every qualifying report must identify the candidate revision, artifact and
model hashes, features, dataset/protocol, host and toolchain. A green report
from another revision is historical evidence until the release gate checks
its applicability; changed model or inference code requires fresh runs.

## 1.0 GO checklist

All boxes require linked evidence before the release is declared GO.

- [ ] **API boundary finalized.** Advertised Rust features, CLI, Python,
      C ABI and JSON contracts have compatibility checks. Publicly reachable
      implementation details are resolved before the final freeze.
- [ ] **Dependency/build contract enforced.** No ONNX Runtime in core product
      graphs; downloader-free local mode; explicit Linux BLAS selection;
      locked release inputs and disclosed native/system dependencies.
- [ ] **Quality and resource gates enforced.** Revision-bound full-split
      Vox/AMI, an additional corpus, platform quality checks, and all five
      Darwin scoreboard limits pass without lowering floors.
- [ ] **Packaged consumers pass.** Actual Rust packages, CLI assets, Python
      wheels and C artifacts pass the target/surface matrix above; supported
      OS, CPU and Python combinations are published.
- [ ] **Repository release checks pass.** Formatting, clippy, tests,
      compatibility, dependency/security checks and documentation checks pass
      for the release candidate; unresolved failures are not waived by this document.
- [ ] **RC stability window completed.** At least two published candidates
      and 14 consecutive days under the [RC policy](docs/semver.md#release-candidate-window),
      with consumer evidence and no unresolved release-blocking regressions.
- [ ] **Readiness reviewed for the exact release.** Link the evidence and
      remaining limitations here, then explicitly change the release verdict.

The open checklist is the release decision. Experimental tract performance,
Parakeet runtime upgrades and removal of every external Rust crate are
tracked separately and must not be substituted for these batch-product gates.
