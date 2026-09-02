# Contributing to polyvoice

Thanks for your interest. This guide matches the **0.19** tree.

## Setup

```bash
git clone https://github.com/ekhodzitsky/polyvoice.git
cd polyvoice

# Ort-free core (default features are empty)
cargo test

# Product stack (kernels, crate-root Pipeline = pipeline v2)
cargo test --features "pipeline-native,vbx"

# ONNX Runtime stack (Python / cli-ort)
cargo test --features "pipeline-full,vbx"

# Product CLI binary (no libonnxruntime)
cargo build --features cli

# Download profile models (~8.4 MB INT8 balanced; signed in release)
cargo run --features cli --bin polyvoice -- download-models --profile balanced
# or: bash scripts/download-models.sh
```


### Feature recipes

| Goal | Features |
|------|----------|
| BYO embedder / library mode | `--no-default-features` (+ optional `clusterer`, `vbx`) — see [docs/library-mode.md](docs/library-mode.md) |
| Production library (kernels) | `pipeline-native` + `vbx` (same as `cli`) |
| ONNX Runtime library | `pipeline-full` + `vbx` |
| CLI / FFI / MCP | `cli` / `ffi` / `mcp` (kernels, no ort) |
| CLI with ONNX Runtime | `cli-ort` |
| CLI with tract | `cli-tract` |
| Native ResNet34 embedder | `embedder-native` (`ResNet34Native`, no ONNX runtime) |
| Native powerset segmenter | `segmenter-native` (`PowersetNative`, N>1 LSTM) |
| WAVE ingest (always on) | `ryf` via `wav::read_wav` / `wav::load_audio` (16 kHz WAV without extra features) |
| Multi-format audio decode | `audio-io` (with `cli`, `cli-tract`, or `cli-native` for the binary): mp3/flac/ogg/m4a plus resample; WAV still uses `ryf` |

Architecture map: [docs/PIPELINE-ARCHITECTURE.md](docs/PIPELINE-ARCHITECTURE.md).
Full doc index: [docs/README.md](docs/README.md). C FFI: [docs/FFI.md](docs/FFI.md).

### Python bindings

```bash
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop --release
pytest tests/ -v
```

## Making changes

1. Fork and create a feature branch from `master`
2. Prefer tests first for behavior changes
3. `cargo fmt` and `cargo clippy --features "pipeline-full,vbx" -- -D warnings`
4. If you touch markdown links: `bash scripts/check-docs-links.sh`
5. Keep PRs focused — one feature or fix per PR
6. Update docs if you change public API
7. Do **not** put internal roadmap task numbers in source, commits, or shipped docs (see [AGENTS.md](AGENTS.md))

## Code style

- Comment only when the *why* is non-obvious
- Match existing patterns
- ONNX-dependent code stays behind `onnx` / stage feature gates
- Lib code: domain `thiserror` errors; no `unwrap`/`expect` outside tests (crate deny)

## Testing

| Command | What it tests |
|---------|---------------|
| `cargo test` | Ort-free unit + integration |
| `cargo test -p polyvoice-kernels` | Hand-written ResNet / powerset kernels |
| `cargo test --no-default-features --features cli-native --test cli_native_smoke` | Kernel-only CLI (no ort/tract) |
| `cargo test --features "pipeline-native,vbx"` | Product stack lib tests (kernels) |
| `cargo test --features "pipeline-full,vbx"` | ONNX Runtime stack lib tests |
| `cargo test --features cli --bin polyvoice` | CLI-related (when applicable) |
| `cargo test --features ffi` | C FFI bindings |
| Full DER gates | CI / `polyvoice-bench` with datasets — not default unit tests |

Ignored tests that need models or network are intentional; release DER gates live in CI.

## Areas for contribution

Check [open issues](https://github.com/ekhodzitsky/polyvoice/issues) and the local `roadmap/` tracker. High-impact directions (as of 0.19):

- **Cross-corpus DER** — CALLHOME / DIHARD-style gates beyond VoxConverse + AMI
- **Linux native RTF** — kernels hold AMI DER within the ort ceiling; RTF still trails Linux ort
- **Python kernels path** — pip still ships ONNX Runtime; CLI / FFI / MCP are native
- **Streaming productization** — powerset-quality online path (batch v2 is production)
- **Docs / DX** — keep README feature recipes and readiness version truth current

Already shipped (not open scaffolding): spectral/NME-SC clusterer, RTTM I/O, VoxConverse/AMI bench harness, AS-norm domain profiles, attribution join.

## Removed public API (0.16)

These soft-deprecated (since 0.12) names were **deleted** in 0.16:

| Removed | Use instead |
|---------|-------------|
| `cluster` / `SpeakerCluster` | `clusterer::Clusterer` / `streaming::ArrivalOrderSpeakerCache` |
| `pipeline::Pipeline` / `pipeline::PipelineError` | `pipeline::LegacyPipeline` / `LegacyPipelineError` (crate-root `Pipeline` is v2) |
| `KMeansClusterer` | `KmeansClusterer` |

Library docs: [docs/API.md](docs/API.md).

## License

By contributing, you agree that your contributions will be licensed under MIT.
