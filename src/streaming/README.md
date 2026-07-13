# src/streaming

## Purpose

Online (streaming) diarization pipeline: windowed processing, incremental
clustering, and streaming state management.

## Surfaces

- `StreamingPipeline<V, E>`
- `StreamingError`

## Dependencies

- `types` — DiarizationConfig, SpeakerTurn, ClusterConfig
- `vad` — VoiceActivityDetector, VadConfig
- `window` — WindowBuffer
- `cluster` — SpeakerCluster
- `embedder` — Embedder

## Invariants

- Output speaker turns are monotonically ordered by time.
- `turns()` returns the cumulative history of every turn emitted across
  `feed()`/`flush()`; `flush()` does not reset it.

## Verification

```bash
cargo test --lib streaming
cargo test --test streaming_test --features onnx
```

## Notes

- Generic over VAD and Embedder types for flexibility.

## Differentiation thesis (vs Sortformer / parakeet-rs)

parakeet-rs wraps NVIDIA Sortformer: end-to-end neural, ASR-first, hard
**4-speaker cap**, GPU-leaning, CC-BY-NC weights with a broken ONNX export
(see `docs/strategy/2026-06-20-wavlm-eend-spike.md`). Chasing it head-on
loses; polyvoice's streaming lane commits to four axes it can defend, each
with a verifiable metric. Competitive context: `docs/COMPETITORS.md` (diart
is the closest streaming OSS peer at 16.8 % VoxConverse online).

**Non-goals:** ASR-first end-to-end neural diarization; a Sortformer ONNX
backend (export broken + non-commercial weights + 4-speaker cap); training
our own EEND weights (frozen by the 2026-07 no-spend decision).

1. **CPU/edge realtime.** Pure Rust + ONNX on CPU, no Python/GPU. Measured
   offline v2 RTFx 11.2 on Apple M1 (release build, `polyvoice-bench`
   RTFx artifact). *Metric:* streaming RTF measured by the same bench
   machinery, > 1x realtime on a 4-core CPU — the streaming-RTF benchmark is
   the first follow-up in TODO.md; no aspirational number until it runs.
2. **Arbitrary speaker count.** `SpeakerCluster` has no architectural cap —
   `ClusterConfig::max_speakers` defaults to 64 (vs Sortformer's hard 4).
   *Metric:* a test streaming ≥ 5 synthetic speakers yields ≥ 5 distinct ids
   (no collapse into 4).
3. **Cannot-link constraints** (spec below): callers with side knowledge
   (separate mic channels, enrolled speakers, a prior diarization) can assert
   "these two regions are different speakers" and the online clusterer will
   never merge them. Neither Sortformer nor diart exposes this.
   *Metric:* the correctness test in the spec.
4. **Explicit latency modes** (spec below): a documented latency budget per
   mode via the bound already stated in `mod.rs` (window_secs + VAD
   look-ahead). *Metric:* per-mode budget table below, asserted by the
   latency benchmark.

**Build vs adopt:** no Rust crate provides online constrained (cannot-link)
clustering (linfa has no constrained variants; COP-KMeans implementations are
Python and offline). The primitive is a candidate filter inside our own
`SpeakerCluster::assign` (~30 lines) — building it is strictly cheaper than
adopting; a streaming Sortformer backend is rejected under non-goals above.

### Spec: cannot-link primitive (design only — implementation is a TODO item)

```rust
/// Caller-defined origin tag (mic channel, enrolled identity, region id).
pub struct SourceTag(pub u32);

impl SpeakerCluster {
    /// Declare that embeddings tagged `a` and embeddings tagged `b` may never
    /// share a speaker id.
    pub fn cannot_link(&mut self, a: SourceTag, b: SourceTag);

    /// `assign` with provenance: the embedding may not join any centroid that
    /// has absorbed a tag cannot-linked with `tag`; the chosen centroid
    /// absorbs `tag`. `None` behaves exactly like `assign`.
    pub fn assign_tagged(&mut self, embedding: &[f32], tag: Option<SourceTag>)
        -> (SpeakerId, f32);
}
```

Integration point: the candidate loop in `SpeakerCluster::assign`
(`src/cluster/mod.rs`) skips centroids whose absorbed-tag set conflicts with
`tag`; at the `max_speakers` ceiling the embedding goes to the closest
*permitted* centroid (a fully-conflicted frame creates no new speaker and
keeps its best permitted assignment). `StreamingPipeline` forwards an
`Option<SourceTag>` per `feed` call (`feed_tagged`); nothing new is stored on
`SpeakerTurn` — tags are input context, not output data.

**Invariant:** for any `cannot_link(a, b)`, no `SpeakerId` is ever returned
for both an `a`-tagged and a `b`-tagged embedding.

**Test sketch:** feed the *identical* embedding twice with tags `a` then `b`
after `cannot_link(a, b)` — ids must differ (without the constraint the
second call provably joins the first centroid); property test: random
embeddings + random constraint set, invariant holds after every call.

### Spec: latency modes (design only — implementation is a TODO item)

```rust
pub enum LatencyMode { LowLatency, Balanced, Accuracy }
```

`LatencyMode::apply(&mut DiarizationConfig)` sets the presets;
`StreamingPipeline::with_latency_mode(vad, extractor, mode)` is constructor
sugar. Budget formula (documented in `mod.rs`): `window_secs` + VAD
look-ahead (EnergyVad frame = 512 samples = 32 ms @ 16 kHz).

| mode | window_secs | hop_secs | latency budget |
|---|---|---|---|
| LowLatency | 1.0 | 0.5 | ≤ 1.04 s |
| Balanced (today's defaults) | 1.5 | 0.75 | ≤ 1.54 s |
| Accuracy | 2.0 | 1.0 | ≤ 2.04 s |

Balanced must stay byte-identical to today's `DiarizationConfig::default()`
so existing callers see no change; the mode enum is additive.
