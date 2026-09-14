# Linux native VBx AHC seed retune (2026-09-13)

Host: Ryzen AI 9 HX 370. Engine: `--features cli` (kernels). Collar 0.
INT8 pair unchanged. One global VBx default — tuned on VoxConverse-dev only.

VBx AHC seed 0.5 → **0.6** (VoxConverse-dev 216: 10.57 % → 9.57 % DER₀).
`fa` / `emb_scale` / `min_embedding_secs` 1-D grids on a 30-file dev subset
were flat around the shipped values.

Held-out (same binary, `--jobs 3`):

| Split | AHC seed 0.5 | AHC seed 0.6 |
|---|---:|---:|
| VoxConverse-test 232 | 14.86 | **13.34** |
| AMI-test 16 | 24.73 | **24.19** |
| Vox-3 (euqef/fuzfh/msbyq) | 7.03 / 7.36 | 7.03 / 7.36 |

Miss/FA unchanged; the drop is confusion. Vox-3 scoreboard floors still hold.
Same-host ort 14.74 / 24.23 is the previous seed-0.5 protocol (not re-run here).
