# TODO — src/streaming

## Current

- [x] AOSC-style arrival-order speaker cache (bounded, recency/confidence).
- [x] Named latency presets `realtime` / `balanced` / `accurate` (+ CLI flag).
- [x] Provisional labels via `SpeakerTurn.stable` + convergence docs.
- [x] `prefer_current_speaker` hysteresis + `label_flip_rate` helper.
- [x] Bounded state tests (cache never exceeds cap).

## Next (ordered)

- [ ] Streaming RTF/latency benchmark: measure streaming RTF and per-mode
      latency with the polyvoice-bench RTFx machinery; fill TBD cells in
      `docs/BENCHMARKS.md` and `benchmarks/results/streaming-latency-methodology.md`.
- [ ] Cannot-link primitive per the README spec (`SourceTag`,
      `cannot_link`, `assign_tagged`, `feed_tagged`; invariant + identical-
      embedding test + property test).
- [ ] ≥ 5-distinct-speakers streaming test (arbitrary-speaker-count metric;
      no collapse into a Sortformer-style cap).
- [ ] ≥1 h long-stream bench artifact (per-chunk latency series start vs end).

## Known Gaps

- DummyExtractor embeddings are call-order pseudo-random, so end-to-end
  multi-speaker identity tests need a content-hashing stub or real ONNX.
- Right-context is accounted in the latency budget but does not yet delay
  emission (window readiness still drives `feed` returns).

## Deferred

- [ ] Support for online speaker re-identification across sessions (enrollment).
