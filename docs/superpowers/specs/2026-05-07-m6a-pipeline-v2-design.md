---
title: M6a — Pipeline + Profile API (additive) Design
date: 2026-05-07
status: draft
milestone: M6a
preceding: M0, M1, M2, M3, M4, M5
following: M6b (cleanup + CLI/FFI/Python migration), M7
authors: ekhodzitsky
---

# M6a — Pipeline + Profile API (additive) Design

## Problem

После M0–M5 у polyvoice есть все компоненты v1.0 диаризации:
- M1 powerset segmenter,
- M2 Embedder trait + CAM++ / ResNet34,
- M3 Clusterer trait + NME-SC / AHC,
- M4 OverlapResegmenter,
- M5 INT8-квантованные ONNX-модели в `[profiles.mobile]` / `[profiles.balanced]`,

но они wired только в legacy `polyvoice::Pipeline::new(DiarizationConfig, VadConfig)` с устаревшим
`SileroVad + FbankOnnxExtractor + AHC` контуром. Пользователь не может включить
новый стек одной строкой — нужна `Pipeline::builder()` API.

## Goal

Single milestone (~1 неделя):

1. **Additive new module** `src/pipeline_v2/` с `Pipeline`, `PipelineBuilder`,
   `PipelineConfig`, `PipelineError`, `ClustererKind`, `ExecutionProvider`.
2. **Builder wiring** для `Profile::Mobile / Balanced / Custom`. Mobile/Balanced
   подтягивают ONNX через `ModelRegistry` (M0); Custom — caller подаёт свои
   `Segmenter / Embedder / Clusterer`.
3. **End-to-end run** через `Pipeline::run(&samples, SampleRate)` который
   соединяет M1 → M2 (overlap mask + embedding) → M3 → M4 → output.
4. **Legacy остаётся работать** — `polyvoice::Pipeline` (renamed export to
   `polyvoice::PipelineV0` if needed in `lib.rs`) и все M0-M3 тесты не ломаются.
5. **Synthetic + #[ignore] E2E test** покрытия.

End state: пользователь может позвать `Pipeline::builder().profile(Profile::Mobile).with_models_from(ModelRegistry::default()?).build()?.run(&samples, SampleRate::HZ_16000)?` и получить
`DiarizationResult` через ONNX INT8 контур без знания деталей M1-M5.

## Non-goals

- CLI `polyvoice diarize --profile mobile` (M6b).
- FFI rewrite в `src/ffi.rs` (M6b).
- Python pyo3 bindings rewrite в `python/src/lib.rs` (M6b).
- `polyvoice-bench` rewrite на новый Pipeline (M6b).
- **Удаление legacy**: `src/pipeline.rs`, `src/offline.rs`, `OfflineDiarizer`,
  `DiarizationConfig`, `VadConfig`, `EnergyVad`, `compute_fbank` privatization,
  `OnlineDiarizer` deprecation note — всё в M6b.
- E2E DER на VoxConverse-test/AMI через `polyvoice-bench --profile mobile` —
  M6b после CLI rewrite.
- Migration guide `docs/MIGRATING-FROM-0.5.md` — M6b.
- Streaming pipeline (`OnlineDiarizer`) — deprecated в v1.0, переезжает в v1.1.

## Approach

### Module placement: separate `src/pipeline_v2/` directory

Pattern мatches M2 (`src/embedder.rs` parallel к legacy `src/embedding.rs`)
и M3 (`src/clusterer.rs` parallel к legacy `src/ahc.rs / spectral.rs`):

```
src/
├── pipeline_v2/                  [NEW dir, M6a]
│   ├── mod.rs                    Pipeline struct, run(), PipelineError
│   ├── builder.rs                PipelineBuilder + ConfigError
│   ├── config.rs                 PipelineConfig + ClustererKind + ExecutionProvider
│   └── mocks.rs                  Test-only Mock{Segmenter,Embedder,Clusterer} (#[cfg(test)])
├── pipeline.rs                   [legacy, kept verbatim for M6a]
```

Cargo feature `pipeline_v2 = []` (default-on, requires `onnx`+`segmentation`+`embedder`+`clusterer`+`resegmentation`):

```toml
default = ["spectral", "segmentation", "embedder", "clusterer", "resegmentation", "pipeline_v2"]
pipeline_v2 = []
```

`pipeline_v2` module gated:

```rust
#[cfg(all(
    feature = "pipeline_v2",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub mod pipeline_v2;
```

При попытке включить `pipeline_v2` без onnx/segmentation/embedder и т.д. cargo
выдаст ошибку компиляции с понятным сообщением (через `compile_error!` в
`pipeline_v2/mod.rs`).

### Public API surface

```rust
// pipeline_v2/mod.rs
pub use builder::{PipelineBuilder, ConfigError};
pub use config::{PipelineConfig, ClustererKind, ExecutionProvider};

pub struct Pipeline { /* private — created via builder */ }

impl Pipeline {
    pub fn builder() -> PipelineBuilder { ... }

    pub fn run(
        &self,
        samples: &[f32],
        sr: SampleRate,
    ) -> Result<DiarizationResult, PipelineError>;
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("audio sample rate {actual} unsupported, expected 16000")]
    UnsupportedSampleRate { actual: u32 },
    #[error("segmentation failed: {0}")]
    Segmentation(#[from] SegmentationError),
    #[error("embedding failed: {0}")]
    Embedding(#[from] EmbedderError),
    #[error("clustering failed: {0}")]
    Clustering(#[from] ClustererError),
    #[error("resegmentation failed: {0}")]
    Resegment(#[from] ResegmentError),
    #[error("config error: {0}")]
    Config(#[from] ConfigError),
    #[error("model registry error: {0}")]
    Registry(#[from] RegistryError),
    #[error("model load error: {detail}")]
    ModelLoad { detail: String },
}
```

### `PipelineConfig` — spec §5.2 verbatim

```rust
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub profile: Profile,
    pub sample_rate: SampleRate,
    pub seg_window_secs: f32,
    pub seg_hop_secs: f32,
    pub clusterer: ClustererKind,
    pub max_speakers: u8,
    pub min_cluster_size: usize,
    pub resegment_overlap: bool,
    pub resegment_min_overlap_secs: f32,
    pub min_speech_secs: f32,
    pub max_gap_secs: f32,
    pub embedder_pool_size: usize,
    pub execution_provider: ExecutionProvider,
}

#[derive(Clone, Copy, Debug)]
pub enum ClustererKind {
    NmeSc,
    Ahc { threshold: f32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ExecutionProvider {
    Cpu,
    CoreMl,    // gated `coreml`
    Nnapi,     // gated `nnapi`
    Cuda,      // gated `cuda`
    XnnPack,   // gated `xnnpack`
}

impl ExecutionProvider {
    pub fn auto() -> Self {
        // macOS aarch64 → CoreMl; Linux aarch64 → XnnPack; else Cpu.
    }
}
```

`PipelineConfig::default()` returns:
```rust
profile: Profile::Balanced,
sample_rate: SampleRate::HZ_16000,
seg_window_secs: 10.0, seg_hop_secs: 0.5,
clusterer: ClustererKind::NmeSc,
max_speakers: 20, min_cluster_size: 12,
resegment_overlap: true, resegment_min_overlap_secs: 0.1,
min_speech_secs: 0.25, max_gap_secs: 0.5,
embedder_pool_size: num_cpus::get().min(4).max(1),
execution_provider: ExecutionProvider::auto(),
```

### `PipelineBuilder` API

```rust
let pipeline = Pipeline::builder()
    .profile(Profile::Mobile)
    .with_models_from(ModelRegistry::default()?)
    .resegment_overlap(true)
    .build()?;

let result = pipeline.run(&samples, SampleRate::HZ_16000)?;
```

Custom path:

```rust
let pipeline = Pipeline::builder()
    .config(my_config)
    .with_segmenter(my_segmenter)
    .with_embedder(my_embedder)
    .with_clusterer(my_clusterer)
    .build()?;
```

Builder methods (returning `&mut self`):
- `.config(PipelineConfig)` — replace full config (overrides `.profile`)
- `.profile(Profile)` — set config.profile
- `.with_models_from(ModelRegistry)` — load Mobile/Balanced ONNX через registry
- `.with_segmenter(Box<dyn Segmenter>)` — Custom only
- `.with_embedder(Box<dyn Embedder>)` — Custom only
- `.with_clusterer(Box<dyn Clusterer>)` — Custom only
- `.with_resegmenter(Box<dyn Resegmenter>)` — optional override (default `OverlapResegmenter::default()`)
- `.resegment_overlap(bool)` — set config.resegment_overlap
- `.embedder_pool_size(usize)`, `.max_speakers(u8)`, и т.п. — convenience setters
- `.build() -> Result<Pipeline, ConfigError>` — финализация

`build()` валидация:

```rust
match cfg.profile {
    Profile::Mobile | Profile::Balanced => {
        if registry.is_none() {
            return Err(ConfigError::MissingRegistry { profile: cfg.profile });
        }
        if any_of(custom_seg, custom_emb, custom_clusterer).is_some() {
            return Err(ConfigError::CustomComponentInProfile {
                profile: cfg.profile,
                offending: ...
            });
        }
        // Resolve registry → segmenter (PowersetSegmenter), embedder (CamPlusPlus or ResNet34Adapter), clusterer (NmeScClusterer or AhcClusterer per cfg.clusterer)
    }
    Profile::Custom => {
        if registry.is_some() {
            return Err(ConfigError::RegistryInCustomProfile);
        }
        if !(custom_seg && custom_emb && custom_clusterer) {
            return Err(ConfigError::MissingCustomComponent {
                missing: vec![...]
            });
        }
    }
}
```

`ConfigError` enum:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("profile {profile:?} requires .with_models_from() call")]
    MissingRegistry { profile: Profile },

    #[error("profile {profile:?} cannot accept .with_{offending}() — Custom only")]
    CustomComponentInProfile { profile: Profile, offending: &'static str },

    #[error("Custom profile cannot accept .with_models_from() — supply components individually")]
    RegistryInCustomProfile,

    #[error("Custom profile missing required components: {missing:?}")]
    MissingCustomComponent { missing: Vec<&'static str> },

    #[error("ONNX model not found in registry: {model_id}")]
    UnknownModel { model_id: String },

    #[error("registry resolution failed: {0}")]
    Registry(#[from] RegistryError),
}
```

### `Pipeline::run()` flow

```text
samples (16 kHz f32 mono PCM, sr asserted)
  ↓
self.segmenter.segment(samples) -> Vec<RawSegment>             (M1)
  ↓
extract_overlap_time_ranges(&raw_segments) -> Vec<(TimeRange, u8 lo, u8 hi)>  (M4)
  ↓
For each non-overlap RawSegment:
  audio_chunk = samples[seg.time.start_samples..seg.time.end_samples]
  audio_masked = apply_overlap_mask(audio_chunk, overlap_in_chunk_secs, sr)   (M2)
  emb = embedder_pool.embed(&audio_masked)                                    (M2)
  l2_normalize(&mut emb)
  ↓
embeddings: Vec<Vec<f32>>, primary_turns: Vec<SpeakerTurn>
  ↓
labels = clusterer.cluster(&embeddings)                       (M3)
  ↓
centroids = compute_centroids(&embeddings, &labels)           (M4)
  ↓
overlap_inputs = build_overlap_region_inputs(
    overlap_time_ranges,
    primary_turns,
    embedder_pool,
    samples,
)                                                              (M4 helper)
  ↓ if cfg.resegment_overlap
all_turns = resegmenter.resegment(ResegmentInputs {
    primary_turns: &primary_turns,
    speaker_centroids: &centroids,
    overlap_regions: &overlap_inputs,
})?
  ↓
all_turns = merge_segments(all_turns, cfg.max_gap_secs)        (utils, legacy reused)
all_turns = filter_short(all_turns, cfg.min_speech_secs)
  ↓
DiarizationResult { turns: all_turns, segments: legacy_view, num_speakers: count(unique speakers) }
```

### Concurrency

- `EmbedderPool` (M2) уже crossbeam-queue; pool_size from config.
- Segmentation последовательная (single ONNX session, lightweight).
- Clustering последовательная (faer eigen multithreaded внутри).
- Pipeline сам Send + Sync (composition of Send + Sync trait objects).

### Re-exports (lib.rs)

```rust
#[cfg(all(
    feature = "pipeline_v2",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub mod pipeline_v2;

#[cfg(all(
    feature = "pipeline_v2",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]
pub use pipeline_v2::{
    ClustererKind, ConfigError, ExecutionProvider, Pipeline as PipelineV2,
    PipelineBuilder, PipelineConfig, PipelineError as PipelineV2Error,
};
```

`polyvoice::Pipeline` (legacy) остаётся доступным; пользователь импортирует
`polyvoice::PipelineV2` для новой API. M6b сделает rename `PipelineV2 → Pipeline`
после удаления legacy.

## File layout

| Path | Action | Responsibility |
|---|---|---|
| `Cargo.toml` | modify | Add `pipeline_v2` feature (default-on, gates on existing onnx+segmentation+embedder+clusterer+resegmentation) |
| `src/pipeline_v2/mod.rs` | create | `Pipeline` struct, `run()`, `PipelineError`, `compile_error!` if feature gates absent |
| `src/pipeline_v2/builder.rs` | create | `PipelineBuilder` + `ConfigError` |
| `src/pipeline_v2/config.rs` | create | `PipelineConfig` + `ClustererKind` + `ExecutionProvider` + Default |
| `src/pipeline_v2/mocks.rs` | create | Test-only `MockSegmenter / MockEmbedder / MockClusterer` (`#[cfg(test)]`) |
| `src/lib.rs` | modify | `pub mod pipeline_v2;` gated, re-exports |
| `tests/pipeline_v2_synthetic_test.rs` | create | Builder validation + Custom profile end-to-end on synthetic data |
| `tests/pipeline_v2_e2e_test.rs` | create | `#[ignore]` integration test on real ONNX (Balanced profile, single voxconverse-test WAV) |
| `CHANGELOG.md` | modify | Unreleased M6a section |

Total roughly 1100 LOC Rust + ~200 lines test + ~150 lines doc.

## Acceptance criteria

1. `cargo test --features download,onnx,segmentation,embedder,clusterer,resegmentation,pipeline_v2 --lib` зелёный (~5 new unit tests in builder/config/mod modules).
2. `cargo test --features download,onnx,segmentation,embedder,clusterer,resegmentation,pipeline_v2 --test pipeline_v2_synthetic_test` зелёный (~6 builder validation + Custom-profile end-to-end tests).
3. `cargo test --features download,onnx,segmentation,embedder,clusterer,resegmentation,pipeline_v2 --test pipeline_v2_e2e_test -- --ignored` локально зелёный (требует `models/` + `data/voxconverse-test/audio/abc.wav`; 30s runtime).
4. `cargo clippy --all-targets --all-features -- -D warnings` clean.
5. `cargo fmt --check` clean.
6. `cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib` зелёный — `pipeline_v2` automatically gated out (требует onnx).
7. `cargo check --no-default-features` зелёный — pipeline_v2 gated out.
8. Legacy `polyvoice::Pipeline` всё ещё работает: existing tests `pipeline::tests::*` (если они есть) + `offline::tests::*` зелёные.
9. `bash scripts/release-gate.sh` выходит exit 2 (все pending steps пока не M6b/M9 — DER row остаётся PENDING потому что DER замер появляется в M6b).

## Tests catalogue

```text
src/pipeline_v2/builder.rs::tests
  builder_default_profile_balanced
  builder_set_profile_mobile
  build_mobile_without_registry_errors
  build_custom_without_components_errors
  build_custom_with_registry_errors
  build_balanced_with_custom_segmenter_errors
  build_custom_full_succeeds

src/pipeline_v2/config.rs::tests
  pipeline_config_default
  clusterer_kind_ahc_threshold
  execution_provider_auto_returns_platform_default

src/pipeline_v2/mod.rs::tests
  pipeline_dyn_compatible
  pipeline_run_unsupported_sample_rate_errors

tests/pipeline_v2_synthetic_test.rs (integration)
  builder_validation_mobile_missing_registry
  builder_validation_custom_missing_components
  pipeline_run_synthetic_two_speakers_through_custom_profile
  pipeline_run_silence_returns_empty
  pipeline_resegment_overlap_disabled_no_secondaries
  pipeline_run_returns_sorted_turns

tests/pipeline_v2_e2e_test.rs (#[ignore])
  e2e_balanced_profile_voxconverse_clip
```

Total: ~13 new tests.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Build fails when `pipeline_v2` enabled without onnx | medium | `compile_error!` macro в `pipeline_v2/mod.rs` с актionable message; integration test в Cargo CI matrix |
| INT8 model load fails (provisional manifest sha256) | high | M5 использовал provisional sha256; ModelRegistry::ensure() проверит, скачает реальный INT8 если URL accessible. Если sha mismatch — actionable error через RegistryError. E2E test будет `#[ignore]` пока M5 follow-up завершён |
| ResNet34 / CAM++ ONNX input shape `[B, T, 80]` отличается от существующего FbankOnnxExtractor (`[B, 80, T]`?) | medium | M2 уже wraps FbankOnnxExtractor в ResNet34Adapter — shape handling there. M6a доверяет M2 wrapper. Если работает в M5 quantization — работает в M6a. |
| `extract_overlap_time_ranges` возвращает empty Vec из-за aggregator не помечающего pairs корректно | medium | Existing M1 unit tests в `aggregator.rs` показывают pairs созданы. M6a integration test покрывает этот path. Если empty — DiarizationResult без overlap turns (graceful, not error). |
| Custom profile mocks не покрывают real Segmenter/Embedder Send + Sync requirements | low | `MockSegmenter` etc. derive Send + Sync через trivial impl; trait objects `Box<dyn Segmenter>` + `Send + Sync` проверяется compile-time. |
| `merge_segments` (legacy from utils.rs) нарушает overlap turns (joins different speakers) | low | merge_segments matches на `speaker == next.speaker` — overlap turns с разными speakers не сольются. |
| `num_cpus` not available without dep | low | Use `std::thread::available_parallelism().map(NonZero::get).unwrap_or(1).min(4)` instead of `num_cpus::get()`. |

## Dependencies on previous milestones

- **M0**: `ModelRegistry`, `Profile`, `ProfileModels` — required для Mobile/Balanced loaders.
- **M1**: `PowersetSegmenter`, `Segmenter` trait — required для segmenter wiring.
- **M2**: `Embedder` trait, `CamPlusPlusExtractor`, `ResNet34Adapter`, `EmbedderPool`, `apply_overlap_mask` — required для embedder wiring.
- **M3**: `Clusterer` trait, `NmeScClusterer`, `AhcClusterer` — required.
- **M4**: `Resegmenter`, `OverlapResegmenter`, `compute_centroids`, `extract_overlap_time_ranges` — required.
- **M5**: INT8 manifest entries — used by registry для Mobile/Balanced. M6a accepts provisional sha256 (M5 follow-up will repin once VoxConverse-dev calibration completes).

M6a doesn't add new external deps. Optionally adds `num_cpus` if `std::thread::available_parallelism()` rejected (preferred std-only path).

## Out of scope (M6b follow-up)

- Удаление `src/pipeline.rs`, `src/offline.rs`, `OfflineDiarizer`, `DiarizationConfig`, `VadConfig`, `EnergyVad`, `VoiceActivityDetector` trait, `segment_speech` fn (privatize в `vad/silero.rs`), `DummyExtractor` (move to `embedding/mock.rs` `#[cfg(test)]`), `OnnxEmbeddingExtractor` (replaced by ResNet34Adapter), `compute_fbank` privatization.
- Renaming `pipeline_v2 → pipeline`, `PipelineV2 → Pipeline`, removing `_v2` suffix.
- CLI rewrite: `polyvoice diarize --profile mobile|balanced`, removing legacy CLI args.
- FFI rewrite: `src/ffi.rs` accepts `Pipeline`/`PipelineConfig` instead of `OfflineDiarizer`/`DiarizationConfig`. New ABI bump.
- Python pyo3 rewrite: `python/src/lib.rs` exposes `Pipeline.builder()` instead of `OfflineDiarizer`.
- `polyvoice-bench` rewrite using new Pipeline.
- E2E DER замер на VoxConverse-test/AMI через polyvoice-bench → `tests/der_baseline.json`.
- Migration guide `docs/MIGRATING-FROM-0.5.md` per spec §11.
- `OnlineDiarizer` deprecation: `#[deprecated(since="2.0", note="redesigned in v1.1")]` annotation — kept as stub.

## References

- Roadmap: `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md`
  §3.1 (pipeline diagram), §5.2 (PipelineConfig), §5.3 (Segmenter / Embedder / Clusterer traits),
  §5.4 (Pipeline + Builder), §5.5 (что удаляется в M6), §10.1 (M6 row).
- M0 plan: `docs/superpowers/plans/2026-05-07-m0-plumbing-and-registry-plan.md`
- M1 plan: `docs/superpowers/plans/2026-05-07-m1-powerset-segmenter-plan.md`
- M2 plan: `docs/superpowers/plans/2026-05-07-m2-cam-pp-embedder-plan.md`
- M3 plan: `docs/superpowers/plans/2026-05-07-m3-nme-sc-clusterer-plan.md`
- M4 plan: `docs/superpowers/plans/2026-05-07-m4-overlap-resegmenter-plan.md`
- M5 plan: `docs/superpowers/plans/2026-05-07-m5-int8-quantization-plan.md`

## Open questions (closed)

- ✅ M6 split в M6a (additive new Pipeline) + M6b (CLI/FFI/Python migration + legacy delete).
- ✅ Module placement: `src/pipeline_v2/` directory parallel к legacy `src/pipeline.rs` (variant 1 from brainstorming).
- ✅ Builder API per spec §5.4 verbatim.
- ✅ Custom profile contract: requires explicit `with_segmenter` + `with_embedder` + `with_clusterer`; conflict с `with_models_from` returns `ConfigError::RegistryInCustomProfile`.
- ✅ Tests: synthetic + `#[ignore]` E2E.

## Follow-ups

1. После одобрения spec: invoke `superpowers:writing-plans` для генерации M6a implementation plan в `docs/superpowers/plans/2026-05-07-m6a-pipeline-v2-plan.md` (стиль M3/M4 plan: TDD-задачи, atomic commits per task, git tag `m6a-complete`).
2. M6b в follow-up: cleanup legacy + CLI/FFI/Python migration + DER baseline closure.
