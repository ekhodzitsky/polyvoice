# Changelog

## [Unreleased]

## [0.11.0] - 2026-07-25

### Changed

- Align with core **0.11.0**: optional `clusterer` / `vbx_plda_dir` on
  `Pipeline.balanced()` / `.mobile()`; VBx when PLDA is configured (matches
  the CLI default after the full DER gate). Feature `vbx` enabled in the
  python crate's polyvoice dependency.

## [0.10.0] - 2026-07-13

### Added

- Typed `DiarizationResult` with `.to_json()` / `.to_rttm()` / `.to_srt()` /
  `.to_vtt()` / `.to_txt()` / `.to_dict()` projections and
  `DiarizationResult.from_json()`; `Pipeline.run_result()` returns it.
  `Pipeline.run()` (plain dict) is unchanged.
- Type stubs (`_polyvoice.pyi`) and a `py.typed` marker — the package is now
  typed for mypy/pyright users.

### Fixed

- `polyvoice.__version__` now reports the installed package version (was a
  stale hardcoded string).


## [0.6.6] - 2026-05-25

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
