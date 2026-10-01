# Documentation index

Stable release: **1.0.0** ([CHANGELOG](../CHANGELOG.md)). The native CPU batch
product ships as Rust, CLI, Python and C ABI v3. All use the INT8 kernel engine;
ONNX Runtime is not a core dependency. MCP and tract are experimental.

- [Release status and evidence](../PRODUCTION-READINESS.md)
- [API stability contract](semver.md)
- [Benchmark protocols and history](BENCHMARKS.md)
- [Held-out NOTSOFAR evaluation](notsofar-eval.md)

## By audience

### CLI user
- [../README.md](../README.md) — installation, first diarization, headline DER
- [BENCHMARKS.md](BENCHMARKS.md) — full protocol, RTF, competitor orientation

### Rust library (kernels / production)
- Features: `pipeline-native` + `vbx` (same as `cli`); set `ClustererKind::Vbx` for CLI parity
- [API.md](API.md) — three layers, types, streaming presets
- [PIPELINE-ARCHITECTURE.md](PIPELINE-ARCHITECTURE.md) — who calls whom
- rustdoc: https://docs.rs/polyvoice

### Rust library (experimental tract)
- Features: `pipeline-tract` + `vbx` (opt-in; no ort)
- Same crate-root `Pipeline` as kernels; engine is tract instead of `polyvoice-kernels`

### Rust library (BYO / no ONNX)
- [library-mode.md](library-mode.md) — empty default features, surface inventory
- [../examples/byo_embedder.rs](../examples/byo_embedder.rs)

### Python
- [../python/README.md](../python/README.md) — install, API, VBx default; same INT8 kernels as the CLI

### C FFI
- [FFI.md](FFI.md) — build, lifecycle, status codes, audio caps
- [../include/polyvoice.h](../include/polyvoice.h) — ABI v3
- [../examples/ffi_usage.c](../examples/ffi_usage.c)

### Agents / experimental MCP / schema
- [../examples/agent_quickstart.md](../examples/agent_quickstart.md)
- [../schema/diarization-result-v1.json](../schema/diarization-result-v1.json)

### Security / ops
- [Experimental Apple Rust kernels](darwin-rust-experiment.md) — build/link audit and measured M1 Pro comparison; experimental path not promoted.
- [Release quality evidence](release-quality.md) — exact-revision full-corpus and Darwin resource gates.
- [../PRODUCTION-READINESS.md](../PRODUCTION-READINESS.md)
- [security/ort-native-binary-provenance.md](security/ort-native-binary-provenance.md)
- [security/audit-2026-05-08.md](security/audit-2026-05-08.md) — **historical** (May 2026)
- [vbx-plda-release.md](vbx-plda-release.md) — shipping PLDA weights

### Optional adapters
- [eres2netv2.md](eres2netv2.md) · [eres2netv2-measured.md](eres2netv2-measured.md)

### Contributors
- [release-artifacts.md](release-artifacts.md) — packaged consumers, release matrix and publication prerequisites
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — feature recipes and development checks
- [DEVELOPMENT-PROCESS.md](DEVELOPMENT-PROCESS.md) — **development process** (not runtime architecture)
- [PIPELINE-ARCHITECTURE.md](PIPELINE-ARCHITECTURE.md) — **runtime** architecture
- [GLOSSARY.md](GLOSSARY.md) · [FORMALISM.md](FORMALISM.md) · [SEVERITY.md](SEVERITY.md)
- [ort-ep-migration.md](ort-ep-migration.md)

### Strategy / competitors (not product manuals)
- [COMPETITORS.md](COMPETITORS.md)
- [strategy/zero-deps.md](strategy/zero-deps.md) — dependency definitions, platform requirements and pure-Rust roadmap
- [strategy/2026-06-20-wavlm-eend-spike.md](strategy/2026-06-20-wavlm-eend-spike.md)

### Archival
- [MIGRATING-FROM-0.5.md](MIGRATING-FROM-0.5.md) — **0.5 → 0.6 only**, not “to 1.0”

### Internal only (not shipped / not linked)
- `docs/superpowers/` is **gitignored** agent plan debris — not part of the product doc set.

## Naming note

| File | About |
|------|--------|
| `PIPELINE-ARCHITECTURE.md` | Diarization runtime: stages, consumers, config defaults |
| `DEVELOPMENT-PROCESS.md` | How we develop (spec → types → verify); stub at `PIPELINE.md` |

## Feature quick map

| Goal | Features |
|------|----------|
| Ort-free BYO | `--no-default-features` (+ `clusterer`, `vbx` optional) |
| Kernels library | `pipeline-native` + `vbx` (CLI parity) |
| Native library with local assets, no downloader | `pipeline-local` |
| Tract ONNX-file library | `pipeline-tract` + `vbx` |
| CLI / FFI / MCP | `cli` / `ffi` / `mcp` (= `pipeline-native` + `vbx`; no `ort`) |
| CLI with tract | `cli-tract` |
| WAVE ingest | always-on (`ryf`); 16 kHz WAV without extra features |
| Multi-format audio | `audio-io` (often with `cli` or `cli-tract`): other containers + resample |

Full table: [library-mode.md](library-mode.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).
