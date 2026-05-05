# Contributing to polyvoice

Thanks for your interest! Here's how to get started.

## Setup

```bash
git clone https://github.com/ekhodzitsky/polyvoice.git
cd polyvoice
cargo test                           # run unit + integration tests
cargo test --features onnx           # include ONNX tests (no models needed)
bash scripts/download-models.sh      # download ONNX models (~27 MB)
POLYVOICE_MODEL_DIR=models cargo test --features onnx --test test_e2e_onnx  # e2e tests
```

### Python bindings

```bash
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop --release
pytest tests/ -v
```

## Making changes

1. Fork the repo and create a feature branch
2. Write tests first — we use TDD
3. Run `cargo fmt` and `cargo clippy --all-features -- -D warnings`
4. Keep PRs focused — one feature or fix per PR
5. Update docs if you change public API

## Code style

- No comments unless the *why* is non-obvious
- Match existing patterns in the codebase
- `#[cfg(feature = "onnx")]` gates all ONNX-dependent code

## Testing

| Command | What it tests |
|---------|---------------|
| `cargo test` | Unit + integration (no ONNX) |
| `cargo test --features onnx` | Includes ONNX compilation |
| `cargo test --test loom_pool` | Concurrency (Loom) |
| `cargo test --features ffi` | C FFI bindings |

## Areas for contribution

Check [open issues](https://github.com/ekhodzitsky/polyvoice/issues) labeled `good first issue`. High-impact areas:

- **Spectral clustering** — alternative to AHC for better accuracy
- **RTTM parser** — read ground-truth annotations for DER evaluation
- **More VAD backends** — WebRTC VAD, custom models
- **Benchmarks** — DER on AMI, VoxConverse, DIHARD datasets

## License

By contributing, you agree that your contributions will be licensed under MIT.
