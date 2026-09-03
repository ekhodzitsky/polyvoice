# Changelog

## [Unreleased]

## [0.19.0] - 2026-09-03

### Changed

- Lockstep with core **0.19.0**. The wheel still takes PCM samples + sample
  rate (no file decode). Rust WAVE ingest moved to `ryf`; Python API unchanged
  on that axis.

## [0.18.0] - 2026-08-25

### Changed

- Lockstep with core **0.18.0**. Python still ships the ONNX Runtime INT8
  stack (`onnx` + stage features). The Rust CLI default moved to hand-written
  kernels; `pip install polyvoice` is unchanged on that axis. Models remain
  the INT8 pair (~8.4 MB).

## [0.17.0] - 2026-08-10

### Changed

- Lockstep with core **0.17.0**: default `Pipeline.balanced()` / `.mobile()`
  download the **INT8** production pair (same as CLI `--profile balanced`).

## [0.16.0] - 2026-08-10

### Changed

- Lockstep with core **0.16.0**. Python API unchanged; core removed soft-deprecated
  Rust aliases (`cluster` / `KMeansClusterer` / `pipeline::Pipeline` rename aliases)
  that the Python bindings did not expose.

## [0.15.0] - 2026-08-05

### Changed

- Lockstep with core **0.15.0**: AS-norm / domain profiles and related CLI
  flags live on the Rust side; Python continues to default to **VBx**
  (`clusterer=None` → VBx; pass `clusterer="ahc"` to opt out).
- Path traversal guard on `models_cache`; pipeline `AudioTooLong` mapped to
  `ValueError` when PCM exceeds the one-hour@16 kHz cap.

## [0.14.0] - 2026-07-31

### Changed

- Lockstep version bump with core 0.14.0; no python-side API changes.
  Headline DER figures (VoxConverse-test 15.2 % no-collar) track core.

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
