# Changelog

## [0.6.5] - 2026-05-24

### Changed

- **Migrated to Pipeline v2** (`polyvoice::pipeline_v2`). `Pipeline.balanced()`
  and `Pipeline.mobile()` now build via `PipelineBuilder` instead of the legacy
  pipeline.
- `Pipeline.run()` releases the GIL during inference — Python threads are no
  longer blocked for the duration of diarization.

### Fixed

- `UnsupportedSampleRate` now raises `ValueError` instead of generic
  `RuntimeError`.
- Model and registry errors now raise `OSError` instead of `RuntimeError`.
