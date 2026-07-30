# Changelog

## [0.13.0] - 2026-07-30

### Changed

- **VBx is now the unconditional default clusterer** (`clusterer=None`),
  matching the CLI: previously VBx was selected only when a PLDA directory
  was configured. PLDA params resolve via `vbx_plda_dir` →
  `POLYVOICE_VBX_PLDA_DIR` → registry download; pass `clusterer="ahc"` to
  opt out. The `Pipeline.balanced()` / `.mobile()` signatures are unchanged.
- Align with the core refactor: the Rust error surface is fully typed
  (`anyhow` / string errors removed from public constructors); Python keeps
  mapping these to `ValueError` / `OSError` / `RuntimeError` with the typed
  detail in the message.
- Build hygiene: the crate declares `rust-version = "1.88"` (tracking the
  core MSRV) and adopts the same clippy lint set as the workspace
  (`unwrap_used = "deny"`).

## [0.12.0] - 2026-07-27

### Changed

- Lockstep version bump with core 0.12.0; no python-side changes.

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
