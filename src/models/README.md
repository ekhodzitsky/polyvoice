# src/models

## Purpose

Model registry, manifest parsing, HTTP download, SHA-256 verification,
minisig signature verification, adapter selection by config string, and
self-describing model metadata (ONNX `metadata_props` with manifest fallback).

## Surfaces

- `ModelRegistry` — download/cache/verify models by profile
- `ProfileModels` — paths to cached model files
- `RegistryError` — registry operation errors
- `Manifest` — typed TOML manifest (schema v1 + v2)
- `AdapterRegistry` — select segmentation/embedder/clusterer/scoring/VAD by string
- `ModelConfigMeta` / `load_model_config` — geometry + license from model or manifest
- `DEFAULT_MANIFEST_TOML` — embedded default manifest

## Dependencies

- `types` — Profile
- `ureq` — HTTP downloads
- `sha2` — SHA-256 verification
- `minisign-verify` — signature verification
- `dirs` — cache directory resolution
- `toml` — manifest parsing
- `onnx` (optional) — read ONNX `metadata_props` via ort

## Invariants

- Downloaded files must match manifest SHA-256 before use.
- Downloaded files must pass minisig verification before use.
- Unknown adapter type → descriptive error (never panic).
- Version alias resolution (`latest` → pin) is logged.

## Verification

```bash
cargo test --lib models --features download
cargo test --test manifest_smoke_test --features download
```

## Notes

- Default manifest is embedded at compile time (`manifest.toml`), schema v2.
- Inject ONNX props with `scripts/inject-model-metadata.py` then re-sign
  (`scripts/sign-models.sh`). Release-key re-signing is a separate release step.
- Until ONNX files carry props, schema-v2 manifest fields act as the fallback
  (`tracing::warn`).
