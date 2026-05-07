---
title: M4 — Overlap Resegmenter Design
date: 2026-05-07
status: draft
milestone: M4
preceding: M0, M1, M2, M3
following: M6 (Pipeline integration), M7
authors: ekhodzitsky
---

# M4 — Overlap Resegmenter Design

## Problem

После M1 (powerset segmenter) + M2 (Embedder + overlap mask) + M3 (Clusterer) полный pipeline умеет обнаруживать overlap-фреймы и присваивать **первого** спикера каждому overlap-региону через clustering чистых embeddings. Но overlap-регион по определению содержит **двух** спикеров. Без дополнительного pass'а второй спикер теряется и DER на VoxConverse недосчитывает overlap miss-rate.

## Goal

Добавить пост-процессинговый `OverlapResegmenter`: для каждого overlap-региона приписать второго спикера, выбирая **ближайший cosine-кластер ≠ primary** среди уже найденных кластеров. Pass обязан:

- Не ломать single-speaker / silence результаты (regress = 0).
- Поправить overlap-mass без повторного запуска segmenter / embedder / clusterer (post-processing only).
- Уложиться в ±0.5% DER на VoxConverse-smoke (acceptance из roadmap §10).
- Быть pure-Rust + wasm32-clean (соответствует `clusterer::AhcClusterer` паттерну).

## Non-goals

- VBx / HMM resegmentation (сдвинуто в v1.2 по roadmap §2.3).
- Re-running segmenter на overlap-регионах (упомянуто в roadmap §3.1 диаграмме как комментарий, но реальный механизм — cosine assignment, см. ту же диаграмму).
- Изменение `Clusterer` / `Embedder` / `Segmenter` traits.
- Wiring в `Pipeline` — это M6.
- Изменение FFI / Python — это M7.

## Existing components M4 опирается на

- `segmentation::RawSegment` (M1): два сегмента с одинаковым `time` и разными `local_speaker_idx`, оба с `is_overlap = true` для каждого overlap-фрейма. См. `aggregator.rs::single_window_overlap_yields_two_segments_same_time`.
- `embedder::apply_overlap_mask` (M2): обнуляет overlap regions перед основным embedding pass'ом.
- `clusterer::Clusterer::cluster(embeddings) -> Vec<usize>` (M3): primary speaker labels для каждого clean-embedded segment.
- `types::SpeakerTurn { time, speaker, confidence }`, `types::SpeakerId(u32)`, `types::TimeRange`.

## Approach: Variant A — pure-Rust post-clustering pass

Caller (M6 Pipeline) подаёт три уже-готовых входа:

1. `primary_turns: &[SpeakerTurn]` — single-speaker turns после clustering, без overlap.
2. `speaker_centroids: &[SpeakerCentroid]` — `(SpeakerId, Vec<f32>)`, L2-normalized центроиды кластеров (mean of L2-normalized embeddings).
3. `overlap_regions: &[OverlapRegionInput]` — для каждой overlap-области: `time`, `primary_speaker: SpeakerId`, `embedding: Vec<f32>` (L2-normalized embedding извлечённого overlap-региона; caller сам решает как — обычно прямой embed без mask, либо отдельный specialised pass).

`OverlapResegmenter::resegment(...)` для каждого overlap-региона:

1. Skip, если `(time.end - time.start) < min_overlap_secs` (сегмент слишком короткий для надёжного assignment).
2. Найти `secondary` = центроид с максимальным cosine similarity к `embedding`, **исключая** `primary_speaker`.
3. Если `cosine(embedding, secondary_centroid) > threshold` → создать дополнительный `SpeakerTurn { time, speaker: secondary.id, text: None }`. Текущий `SpeakerTurn` не несёт confidence-поля; cosine score не сохраняется в M4 — при необходимости его можно прокинуть через возвращаемый `ResegmentDiagnostics` в M6.
4. Если кластеров < 2 (один спикер во всём файле) → пропустить.

Output: `primary_turns ++ secondaries`, отсортированный по `time.start`. Time-spans могут пересекаться — это и есть overlap representation в существующей `Vec<SpeakerTurn>` модели. (Полноценный `DiarizationResultV2` со специальным overlap field — отдельная работа, не в M4.)

### Pseudocode

```text
resegment(primary_turns, centroids, overlap_regions, cfg):
    if centroids.len() < 2: return primary_turns.clone()
    let mut out = primary_turns.to_vec()
    for region in overlap_regions:
        if region.time.duration() < cfg.min_overlap_secs: continue
        let mut best = None
        for (id, c) in centroids:
            if id == region.primary_speaker: continue
            let s = cosine(region.embedding, c)
            if best.map(|(_, sb)| s > sb).unwrap_or(true):
                best = Some((id, s))
        if let Some((id, s)) = best:
            if s > cfg.threshold:
                out.push(SpeakerTurn {
                    time: region.time,
                    speaker: id,
                    confidence: clamp01(s),
                })
    out.sort_by(|a, b| a.time.start.partial_cmp(&b.time.start).unwrap())
    out
```

## API surface

### Module placement

Single file `src/resegmentation.rs` (mirrors `src/clusterer.rs`, `src/embedder.rs` — все per-milestone модули в M2/M3).

Cargo feature `resegmentation = []`, default-on, pure Rust (нет ONNX зависимости).

### Trait + types

```rust
//! v1.0 OverlapResegmenter — overlap-aware post-processing pass.
//! Added in v0.6 (M4). See docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md §3.1.

pub trait Resegmenter: Send + Sync {
    /// Take primary turns + speaker centroids + overlap region inputs and
    /// return a (possibly overlap-aware) flat list of turns.
    ///
    /// **Requires:** all centroid vectors and all overlap embeddings have the
    /// same dimension and are approximately L2-normalized.
    /// **Guarantees on Ok:** every turn in `primary_turns` is preserved
    /// verbatim; secondary turns (if any) carry an existing `SpeakerId` from
    /// `centroids`; output is sorted by `time.start`.
    fn resegment(
        &self,
        inputs: ResegmentInputs<'_>,
    ) -> Result<Vec<SpeakerTurn>, ResegmentError>;
}

#[derive(Debug, Clone)]
pub struct ResegmentInputs<'a> {
    pub primary_turns: &'a [SpeakerTurn],
    pub speaker_centroids: &'a [SpeakerCentroid],
    pub overlap_regions: &'a [OverlapRegionInput],
}

#[derive(Debug, Clone)]
pub struct SpeakerCentroid {
    pub speaker: SpeakerId,
    pub embedding: Vec<f32>, // L2-normalized
}

#[derive(Debug, Clone)]
pub struct OverlapRegionInput {
    pub time: TimeRange,
    pub primary_speaker: SpeakerId,
    pub embedding: Vec<f32>, // L2-normalized
}

#[derive(Debug, thiserror::Error)]
pub enum ResegmentError {
    #[error("centroid dimension mismatch at index {index}: expected {expected}, got {actual}")]
    CentroidDimMismatch { index: usize, expected: usize, actual: usize },
    #[error("overlap embedding dimension mismatch at index {index}: expected {expected}, got {actual}")]
    OverlapDimMismatch { index: usize, expected: usize, actual: usize },
    #[error("primary speaker {primary:?} for overlap region {index} not present in centroids")]
    MissingPrimaryCentroid { index: usize, primary: SpeakerId },
}

pub struct OverlapResegmenter {
    threshold: f32,        // default 0.0
    min_overlap_secs: f32, // default 0.1
}

impl OverlapResegmenter {
    pub fn new(threshold: f32, min_overlap_secs: f32) -> Self;
}

impl Default for OverlapResegmenter {
    fn default() -> Self { Self::new(0.0, 0.1) }
}

impl Resegmenter for OverlapResegmenter { ... }
```

### Helpers (same module, public)

```rust
/// Compute per-cluster L2-normalized centroids from clustered embeddings.
/// `labels[i]` is the cluster label of `embeddings[i]`. Empty clusters yield
/// no entry. Output is sorted by SpeakerId.
pub fn compute_centroids(
    embeddings: &[Vec<f32>],
    labels: &[usize],
) -> Vec<SpeakerCentroid>;

/// Find pairs of overlapping `RawSegment`s (same `time`, different
/// `local_speaker_idx`, `is_overlap = true`) and return one
/// `OverlapRegionInput` per pair. Caller must supply `local_to_global`
/// mapping (typically from clustering pipeline) and `embedder` results
/// for each overlap region. This helper only does interval matching.
///
/// Returns time ranges with the **primary** local speaker (the one that
/// already appears in `primary_turns`); secondary embedding lookup remains
/// the caller's responsibility (M6 Pipeline).
pub fn extract_overlap_time_ranges(
    raw_segments: &[RawSegment],
) -> Vec<(TimeRange, u8 /* primary local idx */, u8 /* secondary local idx */)>;
```

`extract_overlap_time_ranges` живёт в `resegmentation.rs` чтобы инкапсулировать всю overlap-aware логику в одном модуле. Pipeline (M6) использует эту функцию + `local_to_global` mapping + embedder pool для построения `OverlapRegionInput[]`.

### Re-exports (lib.rs)

```rust
#[cfg(feature = "resegmentation")]
pub mod resegmentation;

#[cfg(feature = "resegmentation")]
pub use resegmentation::{
    OverlapRegionInput, OverlapResegmenter, ResegmentError, ResegmentInputs,
    Resegmenter, SpeakerCentroid, compute_centroids, extract_overlap_time_ranges,
};
```

## Cosine similarity contract

Используется существующий `crate::utils::cosine_similarity(a: &[f32], b: &[f32]) -> f32`. Centroids и overlap embeddings обязаны быть L2-normalized; resegmenter **не** перенормализует — это контракт caller'а (как в `Embedder::embed` "guarantees on Ok").

`compute_centroids` нормализует sum → mean → L2 на выходе, поэтому output guaranteed-normalized.

## File layout

| Path | Action | Lines |
|---|---|---|
| `Cargo.toml` | modify | +5 (feature `resegmentation` default-on) |
| `src/resegmentation.rs` | create | ~350 (trait + impl + helpers + tests) |
| `src/lib.rs` | modify | +6 (cfg-gated `pub mod` + re-exports) |
| `tests/resegmentation_test.rs` | create | ~100 (synthetic integration test) |
| `tests/miri_resegmentation.rs` | create | ~50 (no-overlap, full-overlap) |
| `CHANGELOG.md` | modify | +9 (Unreleased M4 section) |

Total: ~520 lines new code.

## Acceptance criteria

1. `cargo test --features resegmentation` зелёный.
2. `cargo test --features resegmentation,segmentation,clusterer,spectral` зелёный (full-feature combo).
3. `cargo test --no-default-features --features resegmentation` зелёный (pure-Rust core).
4. `cargo check --target wasm32-unknown-unknown --no-default-features --features resegmentation --lib` зелёный (wasm32-clean).
5. `cargo clippy --features resegmentation --all-targets -- -D warnings` зелёный.
6. `cargo fmt --check` зелёный.
7. `cargo miri test --features resegmentation --test miri_resegmentation` зелёный.
8. Integration test покрывает:
   - **No overlap input**: output identical to `primary_turns` (silence/single-speaker preservation).
   - **Single overlap, two well-separated centroids**: secondary correctly assigned.
   - **One cluster only**: pass-through (no second speaker to assign).
   - **Threshold gating**: при `threshold = 1.0` (невозможно достичь) — secondary не добавляется.
   - **Min duration gating**: overlap < `min_overlap_secs` пропускается.
9. Property tests (≥1000 cases each):
   - `primary_turns ⊆ output` always (set inclusion на `(start, end, speaker)`).
   - `output.is_sorted_by(|t| t.time.start)`.
   - `compute_centroids(emb, labels)` → каждый centroid L2-normalized (`|‖c‖₂ − 1| < 1e-3`).
10. DER на VoxConverse-smoke: pre-M4 baseline и post-M4 — diff не хуже +0.5% (acceptance из roadmap §10). Замер делается через существующий `polyvoice-bench`. **Этот замер не блокирует merge M4** — он выполняется как отдельный artifact в plan'е и закрепляется в `tests/der_baseline.json` после M5/M6 (когда Pipeline собран).

## Tests catalogue (TDD plan delivers these)

```text
src/resegmentation.rs::tests
  resegmenter_trait_object_is_dyn_compatible
  compute_centroids_l2_normalized
  compute_centroids_drops_empty_clusters
  compute_centroids_sorted_by_speaker_id
  extract_overlap_time_ranges_returns_pairs
  extract_overlap_time_ranges_ignores_non_overlap
  resegment_no_overlap_passes_through
  resegment_single_cluster_passes_through
  resegment_picks_secondary_excluding_primary
  resegment_threshold_blocks_low_confidence
  resegment_min_duration_blocks_short_regions
  resegment_output_sorted_by_start
  resegment_missing_primary_centroid_errors
  resegment_dim_mismatch_errors

tests/resegmentation_test.rs (integration)
  end_to_end_synthetic_two_speakers_overlap
  end_to_end_three_speakers_two_pairs
  rttm_round_trip_preserves_overlap_turns

tests/miri_resegmentation.rs
  miri_resegment_no_overlap
  miri_resegment_single_overlap
  miri_compute_centroids
```

## Risks & mitigations

| Риск | Вероятность | Mitigation |
|---|---|---|
| `cosine_similarity` не нормализуется на потенциально не-L2 vector → wrong scores | низкая | Property test: shuffled L2-normalized vectors → cosine ∈ [-1, 1]. Document contract в trait doc. |
| Default `threshold = 0.0` слишком агрессивен → ложные secondary в speech-only регионах | средняя | Caller (M6 Pipeline) обязан подавать только overlap regions, помеченные segmenter'ом как `is_overlap = true`. Поэтому false-positive overlap detection локализуется в M1 segmenter, а не в M4. Если потребуется более консервативный default — поднимем до 0.3 в M5/M6 после VoxConverse-smoke бенча. |
| Caller передаст overlap embedding для региона где primary не присутствует в centroids | низкая | `ResegmentError::MissingPrimaryCentroid` (нет silent failure). |
| Output ломает downstream RTTM writer (turns с пересечением time) | средняя | Existing `rttm.rs` уже допускает многострочные SPEAKER записи с пересечением. Тестируется явно в integration test через `rttm::write` round-trip. |
| Property test instability (random embeddings → degenerate cosines) | низкая | Use seeded `rand::SeedableRng`; clamp results before assertions. |

## Dependencies on other milestones

- **Inputs:** M1 (`RawSegment`), M2 (`Embedder` for caller, не для M4), M3 (`Clusterer` для caller).
- **Used by:** M6 (Pipeline wires everything together), then M7/M9.
- **Independent of:** M5 (INT8 quantization), M8 (Android/multi-platform).

M4 mergeable до M5 — INT8 артефактов не требует. Параллельно с M4 могут идти M5 calibration работы.

## Open questions

(закрыты в pre-design discussion)

- ✅ Variant A (pure-Rust post-clustering pass) chosen over B (full orchestrator) and C (RawSegment-driven without cosine).
- ✅ Output: flat `Vec<SpeakerTurn>` with possibly overlapping time spans (not a structured `ResegmentOutput`).
- ✅ Default `threshold = 0.0`, `min_overlap_secs = 0.1`.
- ✅ `extract_overlap_time_ranges` lives in `src/resegmentation.rs`.

## Follow-ups

1. После одобрения spec: invoke `superpowers:writing-plans` для генерации M4 implementation plan в `docs/superpowers/plans/2026-05-07-m4-overlap-resegmenter-plan.md` (стиль M3 plan: 5 tasks, TDD, atomic commits per task, git tag `m4-complete`).
2. После M5+M6: запустить `polyvoice-bench --profile balanced` на VoxConverse-smoke до и после `resegment_overlap = true`, зафиксировать DER delta в `tests/der_baseline.json`.

## References

- Roadmap §3.1 (pipeline diagram, "Resegmentation pass"), §3.2 ("Resegmentation: −1…−2% DER на VoxConverse"), §10.1 (M4 row).
- Existing impl pattern: `src/clusterer.rs` (M3) — single-file trait + adapter + tests, feature-gated, default-on.
- Existing helper: `src/embedder.rs::apply_overlap_mask` (M2).
- Aggregator overlap output proof: `src/segmentation/aggregator.rs::tests::single_window_overlap_yields_two_segments_same_time`.
- Powerset paper: Plaquet & Bredin, INTERSPEECH 2023, [arXiv:2310.13025](https://arxiv.org/html/2310.13025v1) (overlap binarization).
