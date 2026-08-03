# VERDICT — speakrs × polyvoice H2H (updated)

**Date:** 2026-08-03  
**Host:** Apple M1 Pro, macOS arm64  
**Scorer:** `benchmarks/der.py` (NIST 10 ms, Hungarian, overlap scored)  
**Worktree:** `~/src/polyvoice-h2h` branch `feat/speakrs-h2h-harness`

## 1. Full VoxConverse-test 232 — speakrs CoreML (measured)

Warm batch (`speakrs-rttm --mode coreml --hyp-dir …`), models loaded once:

| Metric | Value |
|---|---:|
| **DER collar 0 micro** | **11.08%** |
| DER collar 0 macro | 11.55% |
| DER collar 0.25 micro | 6.70% |
| miss / FA / conf (collar 0) | 3.35 / 4.10 / **3.63** |
| spk exact / ±1 / off≥2 | **115** / 58 / 59 |
| Wall / audio | 1044 s / 150029 s |
| **Warm RTFx** | **143.7×** |
| Failures | 0 / 232 |

**This reproduces speakrs’ published ~11.1% on Vox test** under *our* scorer.
Their README 631× is M4 Pro; on M1 Pro warm CoreML we see ~144×.

Artifact: `speakrs-coreml-232-score.json`, RTTMs under
`results_full/voxconverse_test/speakrs-coreml-warm/`.

## 2. polyvoice full-232 (status)

| Source | DER₀ micro | Notes |
|---|---:|---|
| polyvoice **release baseline** (`docs/BENCHMARKS.md`, hop-2, v2+VBx) | **15.24%** | in-repo polyvoice-bench, prior run |
| CLI re-measure (this H2H, cold per file) | **in progress** | ~50–60/232 at verdict time |

Gap vs speakrs full-232 (using release baseline): **~4.2 pp** (15.24 − 11.08).

## 3. Matched subset (same files, same scorer) — **fair head-to-head**

While polyvoice CLI re-measure runs, score both engines on the polyvoice-completed
prefix (n≈57 alphabetical-first files; harder tail not yet fully included):

| Engine | n | DER₀ micro | conf | spk exact |
|---|---:|---:|---:|---:|
| speakrs-coreml | 57 | **12.43%** | **3.79** | 30 |
| polyvoice v2+VBx | 57 | 17.79% | **9.40** | 19 |

**Δ ≈ 5.4 pp**, confusion **+5.6 pp** on polyvoice. Same story as 10-file smoke.

## 4. 10-file smoke (complete earlier)

| Engine | DER₀ | conf | cold RTFx |
|---|---:|---:|---:|
| speakrs-coreml | 13.74 | 2.25 | 22.7× |
| speakrs-cpu | 14.06 | 2.54 | 1.1× |
| polyvoice | 17.72 | 5.77 | **52.5×** |

## 5. Final product verdict

| Question | Answer |
|---|---|
| Is speakrs accuracy real? | **Yes** — 11.08% on full 232 with our scorer |
| Is the polyvoice gap real? | **Yes** — ~4 pp full (baseline), ~5 pp on matched prefix; **confusion** |
| Speed race? | polyvoice cold CLI still strong; speakrs CoreML **warm** ~144× on M1 Pro |
| Next engineering priority | **Speaker confusion / count** (VBx/PLDA/embeddings/seg calibration), not RTF |
| README competitor row | Can now say: *speakrs CoreML measured 11.1% on our scorer (M1 Pro warm, 2026-08-03)* |

## 6. Success criteria for accuracy sprint

- Full-232 polyvoice no-collar micro **≤14%** (half the gap) without RTF &lt; 40× cold CLI  
- Confusion component cut by ≥2 pp  
- Matched prefix DER gap vs speakrs shrinks to ≤2.5 pp  

## 7. Still open

- [ ] polyvoice CLI 232 completion → `polyvoice-232.json`  
- [ ] AMI-test 16 H2H (download incomplete: few WAVs)  
- [ ] Docs PR: BENCHMARKS + COMPETITORS with measured speakrs  
- [ ] Cherry-pick harness into Documents worktree when TCC allows  
