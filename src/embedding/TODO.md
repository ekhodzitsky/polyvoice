# TODO — src/embedding

## Current

- Soft-deprecated `EmbeddingExtractor` / `EmbeddingError` remain for external
  implementors; blanket bridge to `Embedder` in `embedder`.
- [x] `DummyExtractor` implements `Embedder` directly (no bridge).

## Next

- [ ] After a deprecation window, remove `EmbeddingExtractor` and the blanket
      bridge.

## Deferred

- [ ] Hard removal of the legacy trait (major release).
