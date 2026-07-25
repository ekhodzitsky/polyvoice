# TODO — src/embedding

## Current

- Soft-deprecated `EmbeddingExtractor` / `EmbeddingError` remain for external
  implementors; blanket bridge to `Embedder` in `embedder`.
- `DummyExtractor` is the supported in-tree test mock (via the bridge).

## Next

- [ ] After a deprecation window, remove `EmbeddingExtractor` and the blanket
      bridge; keep `DummyExtractor` as a direct `Embedder` impl.

## Deferred

- [ ] Hard removal of the legacy trait (major release).
