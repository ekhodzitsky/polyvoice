# Benchmarks

All polyvoice DER figures below are sourced from
[`tests/der_baseline.json`](../tests/der_baseline.json) (schema
`polyvoice-der-baseline-v2`) and labelled with pipeline, dataset, file count, and
collar. **Diarization Error Rate (DER) is meaningless without a stated collar** —
compare systems only on a matched collar.

## At a glance

| | polyvoice | pyannote 3.1 | whisperX |
|--|-----------|--------------|----------|
| **VoxConverse DER**¹ | **13.83%** | ~12% | ~15% |
| **Model size** | **~30 MB** | ~100 MB | ~1 GB |
| **Runtime** | **CPU only** | GPU recommended | GPU required |
| **Dependencies** | **No Python / PyTorch**² | PyTorch + ONNX | PyTorch + faster-whisper |
| **Languages** | **Rust / Python / C / CLI** | Python only | Python only |
| **Streaming** | **Yes** | No | No |

~80% of pyannote's accuracy at **10× less RAM** and **no GPU**. Runs at **~10×
realtime** on CPU — 9.3× average over a VoxConverse subset
([artifact](../benchmarks/results/voxconverse-test-10files-20260516.json)).

¹ Legacy pipeline, VoxConverse-test (232 files), **0.25 s collar**. The 232-file
no-collar figure was not measured, but on a 10-file subset no-collar DER is
**25.99%** vs 17.43% at 0.25 s collar — expect the strict number several points
higher. Competitor figures use their own conventions and are **not
collar-matched** — compare only on a matched collar.

² The C++ ONNX Runtime is downloaded at build time via the `ort` crate
(`download-binaries`); for hermetic builds use a static-linked / vendored ORT
(see [PRODUCTION-READINESS.md](../PRODUCTION-READINESS.md) §2). No Python/PyTorch
runtime.

## Canonical numbers

**CI-gated** marks rows enforced by the release DER-regression gate.

| Pipeline | Dataset | Files | DER (0.25 s collar) | DER (no-collar) | CI-gated |
|----------|---------|-------|---------------------|-----------------|----------|
| Legacy (Silero + AHC) | VoxConverse-test | 232 | 13.83% | not measured | no |
| Legacy (Silero + AHC) | VoxConverse-test subset | 10 | 17.43% | 25.99% | yes |
| Legacy (Silero + AHC) | e2e smoke (26 s clip) | 1 | 6.62% | not measured | yes |
| Legacy (Silero + AHC) | AMI EN2002a (1 meeting) | 1 | 36.30% | 44.73% | yes |
| v2 (Powerset + ResNet34 + AHC) | e2e smoke (26 s clip) | 1 | 4.43% | not measured | yes |
| Hybrid (Powerset + ResNet34 + AHC) | e2e smoke (26 s clip) | 1 | 4.43% | not measured | no |
| Hybrid (Powerset + ResNet34 + AHC) | VoxConverse-test subset | 3 | 8.27% | not measured | no |
| Hybrid (Powerset + ResNet34 + AHC) | VoxConverse-test subset | 10 | 15.03% | not measured | no |
| Hybrid (Powerset + ResNet34 + AHC) | AMI EN2002a (1 meeting) | 1 | 24.95% | not measured | no |

Notes:

- **No-collar DER is materially higher** than the 0.25 s-collar figure (e.g. the
  10-file legacy subset is 17.43% collar vs **25.99%** no-collar). Compare against
  other systems only on a matched collar.
- The previously headlined "14.12% (232-file, Hybrid + K-means)" number had no
  committed artifact and was withdrawn pending a reproducible, provenance-stamped
  re-run.
- AMI rows are a single meeting (EN2002a, ~79% overlap), not a multi-meeting
  average.
- Automatic speaker count uses silhouette-based k selection with a single-speaker
  guard (no 20-speaker predictions on 1-speaker files).

## Other Rust diarizers

`sherpa-rs` (now archived), `pyannote-rs`, and `speakrs` are the closest Rust
options. None publishes a collar-matched VoxConverse DER, so the comparison above
covers only the established Python systems. polyvoice's differentiators are
maintenance, a pure-Rust core, streaming, and four bindings — see the README's
"Why polyvoice" section.
