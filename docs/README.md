# Documentation index (polyvoice 0.16)

Crate version: **0.16.x** ([CHANGELOG](../CHANGELOG.md)). Canonical accuracy
numbers live in [BENCHMARKS.md](BENCHMARKS.md). Deployment GO/NO-GO:
[PRODUCTION-READINESS.md](../PRODUCTION-READINESS.md).

## By audience

### CLI user
- [../README.md](../README.md) — install, 60-second diarize, headline DER
- [BENCHMARKS.md](BENCHMARKS.md) — full protocol, RTF, competitor orientation

### Rust library (ONNX production)
- Features: `pipeline-full` + `vbx` (enable code); set `ClustererKind::Vbx` for CLI parity
- [API.md](API.md) — dual pipelines, types, streaming presets
- [PIPELINE-ARCHITECTURE.md](PIPELINE-ARCHITECTURE.md) — who calls whom
- rustdoc: https://docs.rs/polyvoice

### Rust library (BYO / no ONNX)
- [library-mode.md](library-mode.md) — empty default features, surface inventory
- [../examples/byo_embedder.rs](../examples/byo_embedder.rs)

### Python
- [../python/README.md](../python/README.md) — install, API, defaults (VBx)

### C FFI
- [FFI.md](FFI.md) — build, lifecycle, status codes, audio caps
- [../include/polyvoice.h](../include/polyvoice.h) — ABI v3
- [../examples/ffi_usage.c](../examples/ffi_usage.c)

### Agents / MCP / schema
- [../examples/agent_quickstart.md](../examples/agent_quickstart.md)
- [../schema/diarization-result-v1.json](../schema/diarization-result-v1.json)

### Security / ops
- [../PRODUCTION-READINESS.md](../PRODUCTION-READINESS.md)
- [security/ort-native-binary-provenance.md](security/ort-native-binary-provenance.md)
- [security/audit-2026-05-08.md](security/audit-2026-05-08.md) — **historical** (May 2026)
- [vbx-plda-release.md](vbx-plda-release.md) — shipping PLDA weights

### Optional adapters
- [sortformer.md](sortformer.md)
- [eres2netv2.md](eres2netv2.md) · [eres2netv2-measured.md](eres2netv2-measured.md)

### Contributors
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — feature recipes, deprecations toward 1.0
- [DEVELOPMENT-PROCESS.md](DEVELOPMENT-PROCESS.md) — **development process** (not runtime architecture)
- [PIPELINE-ARCHITECTURE.md](PIPELINE-ARCHITECTURE.md) — **runtime** architecture
- [GLOSSARY.md](GLOSSARY.md) · [FORMALISM.md](FORMALISM.md) · [SEVERITY.md](SEVERITY.md)
- [ort-ep-migration.md](ort-ep-migration.md)

### Strategy / competitors (not product manuals)
- [COMPETITORS.md](COMPETITORS.md)
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
| ONNX library | `pipeline-full` (+ set `ClustererKind::Vbx`) |
| CLI / FFI / MCP | `cli` / `ffi` / `mcp` (= `pipeline-full` + `vbx` + extras) |
| Multi-format audio | `audio-io` (often with `cli`) |

Full table: [library-mode.md](library-mode.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).
