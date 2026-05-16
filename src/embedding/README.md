# src/embedding

## Purpose

Legacy embedding extraction trait and types. Superseded by `embedder.rs`.
Kept for backward compatibility.

## Surfaces

- `EmbeddingExtractor` trait
- `DummyExtractor`
- `EmbeddingError`

## Dependencies

- `types` — DiarizationConfig

## Invariants

- DummyExtractor is deterministic with fixed seed.

## Verification

```bash
cargo check --all-features
```

## Notes

- Deprecated: use `Embedder` trait from `embedder.rs` for new code.
