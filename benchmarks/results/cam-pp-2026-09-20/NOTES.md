# CAM++ vs ResNet34 — measurement notes

**Date:** 2026-09-20  
**Host:** AMD Ryzen AI 9 HX 370, Linux x86_64, 24 threads  
**Build:** `--features cli,backend-tract` (native powerset + ResNet34 kernels; tract CAM++)  
**Git:** 27883ce (crate 0.21.0)

## Protocol

- Short-seg EER: `polyvoice-measure embedder-short`, VoxConverse-test RTTM pairs
  (400 pairs, 30 files), center-crop 0.5/1/2/3 s, cosine, equal-error rate.
  Same pair construction as the ERes2NetV2 notes.
- DER: `polyvoice-bench --collar 0 --jobs 3`, pipeline v2, native powerset.
  Product ResNet34 arm is VBx (published 13.34 / 24.19). CAM++ is 512-d;
  shipping PLDA is 256-d, so VBx returns a dimension mismatch (see
  `smoke-vbx.err`). CAM++ DER therefore uses AHC (product AHC threshold 0.45).
  A ResNet34 AHC arm is included so the embedder comparison is not confounded
  by the clusterer.
- INT8 CAM++: manifest URL `…/v0.6.0-alpha.2/cam_pp_int8.onnx` returns HTTP 404
  (release gone). EER/DER use `cam_pp_fp32` (sha256 matches the manifest).

## Size floor (native_scoreboard)

| Pair | Bytes |
|------|------:|
| powerset_int8 + resnet34_int8 (shipping) | 8 414 314 |
| powerset_int8 + cam_pp_int8 (manifest size) | 10 452 263 |
| powerset_int8 + cam_pp_fp32 | 30 941 705 |

CAM++ cannot hold the on-disk INT8 pair floor (8 414 314 B) even if the
INT8 artifact were fetchable.

## Short-seg EER

400 RTTM pairs from VoxConverse-test (30 files). ResNet34 = native INT8 kernels;
CAM++ = tract FP32. Artifact: `embedder-short.json`.

| Duration | ResNet34 INT8 EER | CAM++ FP32 EER |
|----------|------------------:|---------------:|
| 0.5 s | 17.79% | **16.88%** |
| 1.0 s | **7.21%** | 10.74% |
| 2.0 s | **4.75%** | 7.05% |
| 3.0 s | **4.60%** | 6.89% |

ResNet34 1.0 s / 2.0 s match the earlier ERes2NetV2 protocol numbers (7.21 / 4.75).
CAM++ is slightly better at 0.5 s and worse at 1–3 s — not an ERes2Net-style
collapse, but not a short-seg win either.

## DER (collar 0)

| Arm | Clusterer | Vox micro | Vox miss/FA/conf | AMI micro | AMI miss/FA/conf | RTFx wall Vox/AMI |
|-----|-----------|----------:|------------------|----------:|------------------|-------------------|
| ResNet34 INT8 (product) | VBx | **13.34** | — / — / 6.59 | **24.19** | — / — / 8.39 | ~206× / ~219× |
| ResNet34 INT8 | AHC 0.45 | 21.24 | 3.83 / 3.81 / 10.94 | 32.84 | 11.71 / 3.28 / 16.76 | 206.6× / 222.0× |
| CAM++ FP32 | AHC 0.45 | 22.69 | 3.82 / 3.81 / 12.13 | 34.97 | 11.71 / 3.29 / 19.48 | 64.2× / 71.7× |

Same-clusterer delta (CAM++ − ResNet34 AHC): Vox **+1.45 pp** (confusion),
AMI **+2.13 pp** (confusion). Miss/FA are bit-identical; the gap is speaker
error. Tract CAM++ is ~3× slower than native ResNet34 on the same AHC path
(64× vs 207× wall), so the published GFLOP win does not show up as pipeline
RTFx under tract.

VBx + CAM++ on one Vox file: `embedding dimension mismatch: expected 256, got 512`.

Speaker-count (AHC 0.45 is a poor AMI counter for both embedders): Vox CAM++
25 exact / 176 off-by-≥2 vs ResNet34 AHC 31 / 179; AMI both 0 exact / 16 off-by-≥2.

## Decision

**Do not switch the default embedder. Do not port CAM++ kernels on this evidence.**

1. DER is worse than 13.34 / 24.19 (and worse than ResNet34 on the same AHC
   clusterer).
2. native_scoreboard INT8 pair floor would be 10.45 MB > 8.41 MB.
3. Shipping VBx PLDA cannot ingest 512-d CAM++.
4. No RTFx win under the only available runtime (tract FP32). A kernel port
   could recover speed but would not fix DER, PLDA dim, or the size floor.
5. Signed `cam_pp_int8` is not downloadable (HTTP 404).
