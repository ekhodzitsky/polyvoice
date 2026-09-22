# mmap kernel weights + bounded scratch — Vox-3 RSS

**Date:** 2026-09-21  
**Host:** AMD Ryzen AI 9 HX 370, Linux x86_64  
**Protocol:** Vox-3 (euqef, fuzfh, msbyq), collar 0, v2+VBx, balanced INT8

ONNX files are mapped with `memmap2` (read-only). INT8 `raw_data` stays a
slice of that mapping during parse; the `Arc<Mmap>` is held on `Powerset`
and `ResNet34` for the model lifetime. No `madvise(DONTNEED)` on weight
pages. Activation scratch is a 1-slot process pool with an 8 MiB f32 soft
cap; `Pipeline::run` drops the slot after each file.

## DER (bit-identical to the locked Linux note)

| | micro | macro |
|--|------:|------:|
| jobs=1 | 7.03 | 7.36 |
| jobs=3 | 7.03 | 7.36 |
| floor | ≤ 7.11 | ≤ 7.39 |

## Peak RSS (`/usr/bin/time -f %M`)

| | this run | previously published (2026-09-08) | floor |
|--|--:|--:|--:|
| jobs=1 | **321 MiB** (328 840 kB) | 297 MiB | ≤ 556 |
| jobs=3 | **463 MiB** (474 208 kB) | 470 MiB | ≤ 556 |

jobs=3 is slightly below the last published figure. jobs=1 is above that
old snapshot (later pipeline + this host) but well under the floor.
Headroom at jobs=3 is ~93 MiB.

## RTF

Linux Vox-3 smoke is not the Darwin 117× floor. This run: jobs=1 wall ~1.4 s
(avg RTFx ~84 on a cold process), jobs=3 wall 0.88 s (`rt_factor_wall` ~137×).
