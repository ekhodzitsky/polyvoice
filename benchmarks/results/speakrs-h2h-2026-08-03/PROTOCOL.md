# Protocol — polyvoice × speakrs head-to-head

**Date:** 2026-08-03  
**Machine:** Apple M1 Pro (arm64), macOS  
**Scorer:** `benchmarks/der.py` (NIST 10 ms frames, Hungarian map)  
**Workspace:** clone at `~/src/polyvoice-h2h` (Documents path blocked by OS TCC for this agent)

## Locked parameters

| Parameter | Value |
|---|---|
| Collar (primary) | **0** (no forgiveness) |
| Collar (secondary) | 0.25 s |
| Overlap | **scored** |
| Aggregate | micro primary; macro + bootstrap CI reported |
| Mapping | Hungarian optimal 1:1 |
| Failures | empty hypothesis → full miss |

## Engines

| Key | Engine | Notes |
|---|---|---|
| `polyvoice` | polyvoice CLI default | v2 + VBx, balanced fp32, ORT CPU |
| `polyvoice-legacy` | `--legacy` | Silero + AHC baseline |
| `speakrs-cpu` | speakrs-rttm `--mode cpu` | ORT CPU |
| `speakrs-coreml` | speakrs-rttm `--mode coreml` | native CoreML (macOS) |
| `speakrs-coreml-fast` | `--mode coreml-fast` | 2 s segmentation step |

## Timing

- Harness RTF includes **cold CLI/process** load per file (conservative).
- Warm/in-process RTFx is optional follow-up via `polyvoice-bench` / multi-file speakrs loop.

## Data

- Smoke: VoxConverse-test **10-file** subset (`scripts/download-voxconverse-test.sh`)
- Headline (when available): full VoxConverse-test 232 + AMI-test 16

## Forbidden mid-run

- Hyperparameter retuning to chase a peer number
- Mixing collars in one leaderboard cell
- Comparing M1 RTF to third-party M4 Pro published figures without labeling HW
