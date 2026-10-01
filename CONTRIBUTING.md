# Contributing to polyvoice

Thanks for your interest. This guide matches the **1.0** product and the current development workflows.

## Setup

```bash
git clone https://github.com/ekhodzitsky/polyvoice.git
cd polyvoice

# Ort-free core (default features are empty)
cargo test

# Product stack (kernels, crate-root Pipeline = pipeline v2)
cargo test --features "pipeline-native,vbx"

# Tract ONNX-file stack (no ort)
cargo test --features "pipeline-tract,vbx"

# Product CLI binary (no libonnxruntime)
cargo build --features cli

# Download profile models (~8.4 MB INT8 balanced; signed in release)
cargo run --features cli --bin polyvoice -- download-models --profile balanced
# ONNX research/test assets use scripts/download-models.sh separately.
```


### Feature recipes

| Goal | Features |
|------|----------|
| BYO embedder / library mode | `--no-default-features` (+ optional `clusterer`, `vbx`) — see [docs/library-mode.md](docs/library-mode.md) |
| Production library (kernels) | `pipeline-native` + `vbx` (same as `cli`) |
| Tract ONNX-file library | `pipeline-tract` + `vbx` |
| CLI / FFI / MCP | `cli` / `ffi` / `mcp` (kernels, no ort) |
| Tract ONNX-file CLI | `cli-tract` |
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
3. `cargo fmt` and `cargo clippy --features cli -- -D warnings`
4. If you touch markdown links: `bash scripts/check-docs-links.sh`
5. Keep PRs focused — one feature or fix per PR
6. Update docs if you change public API
7. Do **not** put internal roadmap task numbers in source, commits, or shipped docs (see [AGENTS.md](AGENTS.md))

## Code style

- Comment only when the *why* is non-obvious
- Match existing patterns
- ONNX-file / tract code stays behind `backend-tract` / `infer` / stage feature gates; product path is kernels (`cli`)
- Lib code: domain `thiserror` errors; no `unwrap`/`expect` outside tests (crate deny)

## Testing

| Command | What it tests |
|---------|---------------|
| `cargo test` | Ort-free unit + integration |
| `cargo test -p polyvoice-kernels` | Hand-written ResNet / powerset kernels |
| `cargo test --no-default-features --features cli-native --test cli_native_smoke` | Kernel-only CLI (no ort/tract) |
| `cargo test --features "pipeline-native,vbx"` | Product stack lib tests (kernels) |
| `cargo test --features "pipeline-tract,vbx"` | Tract ONNX-file stack lib tests |
| `cargo test --features cli --bin polyvoice` | CLI-related (when applicable) |
| `cargo test --features ffi` | C FFI bindings |
| Full DER gates | CI / `polyvoice-bench` with datasets — not default unit tests |

Ignored tests that need models or network are intentional; release DER gates live in CI.

## Areas for contribution

Check [open issues](https://github.com/ekhodzitsky/polyvoice/issues). Local work
tracking uses Backlog.md (`backlog/`, gitignored); run `backlog instructions
overview` before starting a tracked change. Current directions include:

- **Dependency reduction** — preserve the no-ORT core and downloader-free local mode.
- **Darwin Rust kernels** — improve the experimental path without weakening the locked resource floors.
- **Quality** — extend the existing VoxConverse, AMI and held-out NOTSOFAR evidence.
- **Streaming** — develop native powerset streaming separately from the stable batch product.
- **Documentation and packaging** — keep published artifacts and support claims aligned.

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

On Linux, `--all-features` and the release checks require LP64 OpenBLAS
development files and `pkg-config` because they enable `system-openblas`.
Ordinary `--features cli` builds use Rust kernels without BLAS. See the
[backend contract](docs/strategy/zero-deps.md#linux-backend-selection).
