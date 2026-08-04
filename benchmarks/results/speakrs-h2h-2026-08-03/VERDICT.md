# VERDICT — speakrs × polyvoice H2H (final Vox 232)

**Date:** 2026-08-03/04  
**Host:** Apple M1 Pro, macOS arm64  
**Scorer:** `benchmarks/der.py` (NIST 10 ms, Hungarian, overlap scored)  
**Worktree:** `~/src/polyvoice-h2h` branch `feat/speakrs-h2h-harness`

## Full VoxConverse-test 232 — matched, same scorer

| Engine | DER₀ micro | DER₀.₂₅ micro | miss | FA | conf | spk exact / ±1 / ≥2 | RTFx |
|---|---:|---:|---:|---:|---:|---:|---:|
| **speakrs-coreml (warm)** | **11.08** | **6.70** | 3.35 | 4.10 | **3.63** | **115** / 58 / 59 | **~144×** warm |
| **polyvoice v2+VBx** | **15.22** | **10.47** | 3.20 | 3.98 | **8.04** | 84 / 67 / 81 | **~40×** cold CLI |

**Gap: 4.14 pp** no-collar micro (15.22 − 11.08).  
**Dominant residual:** confusion (+4.41 pp). Miss and FA are essentially tied.

Artifacts:
- `full-232-matched-score.json`
- `polyvoice-232.json` (cold CLI harness)
- `speakrs-coreml-232-score.json`
- RTTMs: `results_full/voxconverse_test/{polyvoice,speakrs-coreml-warm}/`

## Interpretation

1. **speakrs accuracy claim is real** — 11.08% on our scorer ≈ their published 11.1%.
2. **polyvoice re-measure matches release baseline** (15.24% → 15.22%).
3. **Accuracy gap is structural (speaker assignment)**, not VAD miss.
4. **Speed:** polyvoice cold CLI ~40× remains strong; speakrs CoreML warm ~144× on M1 Pro
   (their 631× is M4 Pro / different protocol). Do not race RTF as the product story.
5. Speaker count: speakrs exact 115/232 vs polyvoice 84/232 — same weak axis.

## Product decisions

| Do | Don't |
|---|---|
| Publish **measured** speakrs row in BENCHMARKS | Claim parity with community-1 ports |
| Sprint on **confusion / speaker count** | Optimize RTF vs speakrs CoreML |
| Keep MIT / ungated / multi-surface pitch | Train own EEND |

## Accuracy sprint gate

- Full-232 polyvoice no-collar micro **≤14.0%** (half the gap)  
- Confusion ≤ **6.0** (from 8.04)  
- Cold CLI RTFx not below ~35× on M1 Pro  

## Still open

- [ ] AMI-test 16 matched H2H  
- [ ] Ship docs PR (BENCHMARKS + COMPETITORS)  
- [ ] Merge harness into main Documents worktree  
