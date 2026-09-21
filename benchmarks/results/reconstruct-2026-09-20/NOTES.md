# Per-(window, speaker) reconstruct — 2026-09-20

Host: Linux x86_64, kernels (`--features cli`), pipeline v2 + VBx, collar 0,
`--jobs 3`. INT8 pair unchanged at 8 414 314 B. Gated by
`PipelineConfig.reconstruct` / `polyvoice-bench --reconstruct` (default
**off**).

Shipped baseline (same binary, reconstruct off): VoxConverse-test **13.34 /**
6.59 confusion**, AMI-test **24.19 / 8.39**, wall RTFx ~206× / ~219×.

## What landed

Powerset windows stay local. Each (10 s window, local speaker) with ≥0.20 s
speech is cropped to the mask bounding box (full-window PCM zeros polluted
WeSpeaker stats-pooling), clustered with VBx (`min_embedding_secs = 5` on
this path only; 1.6 s over-clusters), then reconstructed by overlap-adding
mapped masks and taking per-frame top-k from the instantaneous speaker
count. Same-window locals that collide are reassigned.

## Dev (VoxConverse-dev, first 30 files)

| Step | DER₀ | Miss | FA | Conf | RTFx wall |
|---|---:|---:|---:|---:|---:|
| baseline | 8.85 | 2.64 | 2.08 | 3.46 | 183.5 |
| reconstruct, full-window PCM zero | 14.19 | 2.43 | 2.29 | 8.33 | 39.8 |
| + crop to mask | 10.24 | 2.41 | 2.25 | 4.79 | 41.4 |
| + VBx min 5 s (embed shorts, reassign) | **8.65** | 2.39 | 2.26 | 3.06 | 47.6 |
| crop + skip-embed <5 s (no reassign) | 10.03 | 4.49 | 1.81 | 2.67 | 53.5 |

VBx min-embed grid on the crop path: 2.5 s → 10.26, 3.5 → 9.32, **5.0 →
8.65**, 8.0 → 9.24. JSON: `dev30/`, `dev30-retune/`.

## Held-out (gate)

| | Vox-232 DER₀ | Miss | FA | Conf | AMI-16 DER₀ | Miss | FA | Conf | RTFx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped | **13.34** | 3.86 | 3.80 | 6.59 | **24.19** | 11.73 | 3.25 | 8.39 | ~206 / ~219 |
| reconstruct + crop + min 5 s | **14.51** | 3.29 | 4.11 | 7.10 | **22.79** | 10.00 | 3.43 | 8.88 | 55.6 / 55.7 |

Vox speaker-count MAE 1.40 (exact 82/232, off-by-≥2 78) vs shipped 1.51
(83 exact, 84 off-by-≥2). Count is not the Vox regression; FA and
confusion are. AMI MAE 3.25 vs shipped 2.75.

JSON: `held-out/`. Embedder wall is ~5× the shipped path (many overlapping
10 s units). Vox-3 scoreboard was not run on `--reconstruct`; default-off
path still holds the floors.

## Verdict

- **Not merged.** One global default cannot take the AMI win (−1.40 pp)
  against a Vox loss (+1.17 pp). Target was Vox ≤ 11.5 % and AMI ≤ 21 %.
- **RTFx floor:** ~56× vs shipped ~200× and vs Darwin floor 117×. Cropping
  shorts does not recover it; a 1 s segmentation hop would make this worse.
  Stats-pool masking inside the ResNet (instead of PCM) is the remaining
  cost lever and is not in this change.
- **R7** (min_speech / gap → 0) and **R8** (hop 2→1 s) were not gated.
  The parent step already fails the Vox DER gate and the RTFx floor.
- Overlay stays available: `polyvoice-bench --reconstruct`.

Dev-30 liked the path (8.85→8.65). Held-out Vox did not; speaker-count
error is still the bulk of the confusion gap to speakrs 3.63.
