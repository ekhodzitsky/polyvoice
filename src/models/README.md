# src/models

## Purpose

Model registry, manifest parsing, HTTP download, SHA-256 verification, and
minisig signature verification for ONNX model bundles.

## Surfaces

- `ModelRegistry` — download/cache/verify models by profile
- `ProfileModels` — paths to cached model files
- `RegistryError` — registry operation errors
- `Manifest` — typed TOML manifest
- `DEFAULT_MANIFEST_TOML` — embedded default manifest

## Dependencies

- `types` — Profile
- `ureq` — HTTP downloads
- `sha2` — SHA-256 verification
- `minisign-verify` — signature verification
- `dirs` — cache directory resolution
- `toml` — manifest parsing

## Invariants

- Downloaded files must match manifest SHA-256 before use.
- Downloaded files must pass minisig verification before use.

## Verification

```bash
cargo test --lib models --features download
cargo test --test m5_manifest_smoke_test --features download
```

## Notes

- Default manifest is embedded at compile time (`manifest.toml`).
