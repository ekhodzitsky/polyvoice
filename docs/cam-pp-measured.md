# CAM++ vs default embedder — measured notes

**Date:** 2026-09-20  
**Host:** AMD Ryzen AI 9 HX 370, Linux x86_64  
**Artifact:** [`benchmarks/results/cam-pp-2026-09-20/`](../benchmarks/results/cam-pp-2026-09-20/)

WeSpeaker CAM++ is already in the model manifest (512-d). Published VoxCeleb1-O
EER is slightly better than ResNet34, and the forward pass is ~8× cheaper
(1.15 vs 4.55 GFLOPs). Switching the default without a short-seg + DER
measurement would repeat the ERes2NetV2 mistake (paper EER 0.61% vs short-seg
20.09%).

## Short-segment EER (VoxConverse RTTM pairs, not VoxCeleb1)

400 same/different-speaker pairs from VoxConverse-test RTTM (30 files),
center-crop, cosine scoring. ResNet34 is the shipping native INT8 kernels.
CAM++ is tract FP32: the signed `cam_pp_int8` release URL returns HTTP 404.

| Duration | ResNet34 INT8 EER | CAM++ FP32 EER |
|----------|------------------:|---------------:|
| 0.5 s | 17.79% | **16.88%** |
| 1.0 s | **7.21%** | 10.74% |
| 2.0 s | **4.75%** | 7.05% |
| 3.0 s | **4.60%** | 6.89% |

## DER (collar 0, jobs=3)

Product ResNet34 INT8 + VBx: VoxConverse-test **13.34%**, AMI-test **24.19%**.
CAM++ is 512-d; shipping VBx PLDA is 256-d, so VBx rejects CAM++ embeddings
(`expected 256, got 512`). The CAM++ DER arm therefore uses AHC (threshold
0.45, the product AHC default). ResNet34 AHC is the same-clusterer control.

| Arm | Clusterer | Vox DER₀ micro | AMI DER₀ micro | RTFx wall Vox / AMI |
|-----|-----------|---------------:|---------------:|---------------------|
| ResNet34 INT8 (product) | VBx | **13.34%** | **24.19%** | ~206× / ~219× |
| ResNet34 INT8 | AHC 0.45 | 21.24% | 32.84% | 206.6× / 222.0× |
| CAM++ FP32 | AHC 0.45 | 22.69% | 34.97% | 64.2× / 71.7× |

Same-clusterer gap is confusion: Vox +1.45 pp, AMI +2.13 pp. Miss/FA match.
Tract CAM++ is ~3× slower than native ResNet34 on this path.

## Size floor

`powerset_int8` + `resnet34_int8` = 8 414 314 B (locked). Manifest
`cam_pp_int8` is 8 803 007 B, so the pair would be 10 452 263 B even if the
INT8 file were fetchable.

## License

VoxCeleb-trained WeSpeaker weights follow CC-BY-4.0 (WeSpeaker
`docs/pretrained.md`). Manifest `cam_pp_fp32` / `cam_pp_int8` were labeled
Apache-2.0; they now say CC-BY-4.0. The zh-cn CAM++ export stays Apache-2.0.

## Verdict

Do **not** switch the default embedder. CAM++ cannot enter the shipping VBx
stack, misses the INT8 pair size floor, is worse on same-clusterer AHC DER
(Vox +1.45 pp, AMI +2.13 pp), is not better on 1–3 s short-seg EER, and is
~3× slower under tract than native ResNet34. Keep the adapter for experiments;
no kernel port on this evidence.
