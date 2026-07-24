# src/streaming

## Purpose

Online (streaming) diarization pipeline: windowed processing, AOSC-style
arrival-order speaker cache, named latency presets, and provisional labels.

## Surfaces

- `StreamingPipeline<V, E>` — `new`, `with_latency_preset`, `with_params`, `feed`, `flush`
- `StreamingError`
- `LatencyPreset` / `StreamingParams` — `realtime` | `balanced` | `accurate`
- `ArrivalOrderSpeakerCache` / `AssignResult`
- `prefer_current_speaker`, `label_flip_rate`

## Dependencies

- `types` — DiarizationConfig, SpeakerTurn (`stable` flag), ClusterConfig
- `vad` — VoiceActivityDetector, VadConfig
- `window` — WindowBuffer
- `embedding` — EmbeddingExtractor (legacy; see embedder migration)
- `utils` — cosine similarity, L2 normalize

## Invariants

- Output speaker turns are monotonically ordered by time.
- `turns()` returns the cumulative history of every turn emitted across
  `feed()`/`flush()`; `flush()` does not reset it.
- Speaker cache length never exceeds `speaker_cache_cap` (overflow force-merges).
- Arrival-order IDs are stable across chunks (no global recluster / Hungarian).
- `SpeakerTurn.stable == false` means provisional; once true for a cache entry,
  that speaker ID is immutable (history is not rewritten).

## Latency presets

| Preset     | window | hop  | right ctx | cache | budget @16 kHz (EnergyVad 512) |
|------------|--------|------|-----------|-------|--------------------------------|
| `realtime` | 1.0 s  | 0.5  | 0.0       | 16    | ≈ 1.03 s                       |
| `balanced` | 1.5 s  | 0.75 | 0.0       | 32    | ≈ 1.53 s                       |
| `accurate` | 2.0 s  | 1.0  | 0.25      | 64    | ≈ 2.28 s                       |

`balanced` matches `DiarizationConfig::default()` window geometry.
CLI: `--latency-preset realtime|balanced|accurate` (applies window geometry).

## Verification

```bash
cargo test --lib streaming
cargo test --test streaming_test --features onnx
```

## Notes

- Generic over VAD and Embedder types for flexibility.
- Structure only for AOSC — does not pull Sortformer weights.

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
2. **Arbitrary speaker count.** Cache cap defaults to 32–64 (vs Sortformer's
   hard 4); overflow merges rather than dropping audio.
   *Metric:* a test streaming ≥ 5 synthetic speakers yields ≥ 5 distinct ids
   when under the cap (no collapse into 4).
3. **Cannot-link constraints** (spec below): callers with side knowledge
   (separate mic channels, enrolled speakers, a prior diarization) can assert
   "these two regions are different speakers" and the online clusterer will
   never merge them. Neither Sortformer nor diart exposes this.
   *Metric:* the correctness test in the spec.
4. **Explicit latency modes:** documented latency budget per preset via
   window + right context + VAD frame. *Metric:* per-mode budget table above;
   measured latency/RTF/DER in `docs/BENCHMARKS.md` once the bench runs.

**Build vs adopt:** no Rust crate provides online constrained (cannot-link)
clustering (linfa has no constrained variants; COP-KMeans implementations are
Python and offline). The primitive is a candidate filter inside our own
assign path (~30 lines) — building it is strictly cheaper than adopting; a
streaming Sortformer backend is rejected under non-goals above.

### Spec: cannot-link primitive (design only — implementation is a TODO item)

```rust
/// Caller-defined origin tag (mic channel, enrolled identity, region id).
pub struct SourceTag(pub u32);

impl ArrivalOrderSpeakerCache {
    /// Declare that embeddings tagged `a` and embeddings tagged `b` may never
    /// share a speaker id.
    pub fn cannot_link(&mut self, a: SourceTag, b: SourceTag);

    /// `assign` with provenance: the embedding may not join any centroid that
    /// has absorbed a tag cannot-linked with `tag`; the chosen centroid
    /// absorbs `tag`. `None` behaves exactly like `assign`.
    pub fn assign_tagged(&mut self, embedding: &[f32], tag: Option<SourceTag>)
        -> AssignResult;
}
```

Integration point: the candidate loop skips centroids whose absorbed-tag set
conflicts with `tag`; at the cap the embedding goes to the closest *permitted*
centroid. `StreamingPipeline` forwards an `Option<SourceTag>` per `feed` call
(`feed_tagged`); nothing new is stored on `SpeakerTurn` — tags are input
context, not output data.

**Invariant:** for any `cannot_link(a, b)`, no `SpeakerId` is ever returned
for both an `a`-tagged and a `b`-tagged embedding.

**Test sketch:** feed the *identical* embedding twice with tags `a` then `b`
after `cannot_link(a, b)` — ids must differ; property test: random embeddings
+ random constraint set, invariant holds after every call.
