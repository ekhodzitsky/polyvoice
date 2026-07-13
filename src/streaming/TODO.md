# TODO — src/streaming

## Current

## Next (ordered; specs live in README.md "Differentiation thesis")

- [ ] Streaming RTF/latency benchmark: measure streaming RTF and per-mode
      latency with the polyvoice-bench RTFx machinery (labels the report,
      no aspirational numbers before this lands).
- [ ] `LatencyMode` enum + presets per the README spec (Balanced ==
      today's defaults, additive API).
- [ ] Cannot-link primitive per the README spec (`SourceTag`,
      `cannot_link`, `assign_tagged`, `feed_tagged`; invariant + identical-
      embedding test + property test).
- [ ] ≥ 5-distinct-speakers streaming test (arbitrary-speaker-count metric;
      no collapse into a Sortformer-style cap).

## Known Gaps

## Deferred

- [ ] Support for online speaker re-identification across chunks.
