# Scoring-chain cheap steps (R2–R6) — 2026-09-18

Host: Linux x86_64 (Ryzen AI 9 HX 370), kernels (`--features cli`),
pipeline v2 + VBx, collar 0, overlap scored, `--jobs 3`. INT8 pair
unchanged at 8 414 314 B. Production construction stays env-free;
overlays use `POLYVOICE_VBX_FROM_ENV=1`.

Speakrs H2H (same scorer, 2026-08-03): VoxConverse-test DER₀ **11.08 %**,
confusion **3.63**. Linux kernel baseline (this binary): **13.34 / 6.59**
confusion, AMI-test **24.19 / 8.39** — bit-match of
`linux-cpu-native-der-2026-09-13-vbx-ahc/`.

No production default was merged. The cheap steps do not close the
confusion gap; the heavy reconstruction step remains **go**.

## Held-out (gate)

| Step | Vox-232 DER₀ | Miss | FA | Conf | AMI-16 DER₀ | Miss | FA | Conf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (shipped) | **13.34** | 3.86 | 3.80 | 6.59 | **24.19** | 11.73 | 3.25 | 8.39 |
| R2 clean-mask + clean-duration filter | 13.34 | 3.86 | 3.80 | 6.59 | 24.19 | 11.73 | 3.25 | 8.39 |
| R3 AHC Euclidean on L2, thr 0.75 | 13.54 | 3.86 | 3.79 | 6.43 | 23.64 | 11.73 | 3.25 | 7.85 |
| R4 Fa=0.07, GMM (`loop_prob=0`) | 24.34 | 4.89 | 3.62 | 19.34 | 23.81 | 11.71 | 3.26 | 8.27 |
| R6 `emb_scale=1` | 39.73 | 4.60 | 3.69 | 26.77 | 66.79 | 19.42 | 2.73 | 41.49 |
| R5 prior π>1e-7 + soft reassign | 13.24 | 3.86 | 3.80 | 6.50 | 24.21 | 11.74 | 3.25 | 8.40 |
| R3+R5 cumulative | 13.41 | 3.86 | 3.80 | 6.34 | 23.35 | 11.73 | 3.25 | 7.61 |

JSON: `voxconverse-test/`, `ami-test/`. Wall RTFx stayed ~200–220× on both
splits (jobs=3); no new dependency; model bytes 8 414 314.

### Verdicts

- **R2 — no-op.** Per-file DER is bit-identical to baseline on both full
  splits. Clean-duration never changed which embeddings the 1.6 s filter
  kept, and every unit had ≥2 s unmasked speech so the mask fallback
  never fired. Keep the overlay; do not change the default.
- **R3 — no-go as a default.** Dev-30 liked Euclidean 0.75 (8.85→8.45).
  Held-out Vox **regresses** 13.34→13.54 (speaker-count off-by-≥2:
  84→95/232). AMI improves 24.19→23.64. One global default cannot take
  the AMI win. The 30-file grid overfit a handful of long files.
- **R4 — no-go.** Vox 13.34→24.34. AMI is slightly better (24.19→23.81)
  and speaker-count MAE collapses 2.75→0.38 (11/16 exact vs 1/16) — GMM
  + Fa=0.07 is a meeting-corpus prior, not a Vox one. Not shippable as
  one global default.
- **R6 — no-go.** Dropping `emb_scale=4.88` dumps unit-norm embeddings
  into a PLDA mean-center that was fit on the 4.88-scaled WeSpeaker
  recipe. Vox 39.73 / AMI 66.79.
- **R5 — not merged.** Vox 13.34→13.24 (94 files better, 55 worse, 83
  same; speaker count unchanged). AMI 24.19→24.21, a real +0.02 on a
  deterministic run. Too small a Vox win to pay an AMI tick-up. Overlay
  stays available (`POLYVOICE_VBX_SOFT_REASSIGN=1`).
- **R3+R5 — not merged.** Best AMI (23.35) and best confusion on both
  splits (Vox 6.34 / AMI 7.61), but Vox DER 13.41 still worse than 13.34.

## Dev screening (VoxConverse-dev, first 30 files, sorted)

| Step | DER₀ micro | Miss | FA | Conf |
|---|---:|---:|---:|---:|
| baseline | 8.85 | 2.64 | 2.08 | 3.46 |
| R2 | 8.85 | 2.64 | 2.08 | 3.46 |
| R3 thr 0.60 | 8.70 | 2.63 | 2.07 | 3.18 |
| R4 | 19.31 | 3.88 | 2.00 | 11.22 |
| R6 | 25.29 | 3.39 | 2.00 | 12.88 |
| R5 | 8.79 | 2.63 | 2.07 | 3.40 |

R3 Euclidean threshold grid on the same 30 files. Community-1's 0.6
distance cut is stricter than shipped cosine 0.6 (≈ Euclidean 0.89 on
the unit sphere). Dev minimum was **0.75** — it did not survive held-out.

| thr | DER₀ | Conf |
|---|---:|---:|
| 0.45 | 8.90 | 4.02 |
| 0.50 | 8.90 | 4.02 |
| 0.55 | 9.24 | 4.65 |
| 0.60 | 8.70 | 3.18 |
| 0.65 | 8.62 | 3.01 |
| 0.70 | 8.56 | 2.84 |
| **0.75** | **8.45** | **2.80** |
| 0.80 | 8.60 | 2.89 |
| 0.85 | 8.58 | 2.88 |
| 0.90 | 9.18 | 3.08 |
| 0.95 | 9.85 | 3.56 |
| 1.00 | 20.10 | 8.28 |

JSON: `dev30/`, `r3-grid/`.

## R1 go / no-go

Threshold: remaining confusion vs speakrs 3.63 must stay **> 1 pp** to
justify the heavy per-(window, speaker) masked-embedding + reconstruct
step.

| Condition | Vox confusion | gap to 3.63 |
|---|---:|---:|
| shipped baseline | 6.59 | 2.96 |
| best cheap step (R5) | 6.50 | 2.87 |
| best cheap combo (R3+R5) | 6.34 | 2.71 |

All gaps **> 1 pp**. Speaker-count error is also unchanged under R5
(exact 83/232, off-by-≥2 **84/232** — same as baseline; speakrs was
115/232 exact). The cheap chain does not replace reconstruction.

**GO** for the reconstruction step.

## How to reproduce overlays

```bash
# shipped defaults
polyvoice-bench data/voxconverse-test --collar 0 --jobs 3 --pipeline v2 --clusterer vbx

# R5 only (the least-harmful cheap step)
POLYVOICE_VBX_FROM_ENV=1 POLYVOICE_VBX_SOFT_REASSIGN=1 \
  polyvoice-bench data/voxconverse-test --collar 0 --jobs 3 --pipeline v2 --clusterer vbx

# helper
scripts/measure-scoring-chain.sh data/voxconverse-dev 30
```
