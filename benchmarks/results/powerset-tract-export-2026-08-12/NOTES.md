# Powerset → tract re-export spike (2026-08-12)

## Goal

Unblock pure-Rust **v2 segmentation** by making powerset ONNX loadable in
`tract` (feature `backend-tract`).

## Result: **PARTIAL SUCCESS**

| Check | Status |
|-------|--------|
| ort numerical parity (rewrite vs original) | **PASS** max-abs ~1e-5 @ T=160000 |
| tract **load** of rewrite | **PASS** (concrete `[1,1,160000]` fact) |
| tract **run** + ort/tract parity | **PASS** (test profile) |
| Dynamic T under tract symbols | **FAIL** (still needs concrete T) |
| Shipping as product default | **No** (fixed-T plan; ~slower than ort; not wired into pipeline) |

## Transforms (`scripts/export-powerset-tract.py`)

1. **Inline identical-branch `If`** — then/else were the same Conv (export junk).
2. **Expand `InstanceNormalization`** → `ReduceMean` / `Sub` / `Mul` / `Sqrt` /
   `Div` + affine (`Reshape` scale/bias to `[1,C,1]`).

## Tract load strategy (`src/onnx/tract_session.rs`)

After direct + symbolic optimize fail, try concrete facts:

- `[1, 1, 160000]` — product 10 s window @ 16 kHz  
- `[1, 1, 16000]` — 1 s fallback  

## Reproduce

```bash
python3 scripts/export-powerset-tract.py --verify
cargo test --lib --features "onnx,backend-tract" \
  powerset_fp32_tract_friendly -- --nocapture
```

## Follow-ups

1. Wire optional pipeline path: pad/crop every powerset window to 160000 and run
   tract segmenter when `POLYVOICE_INFERENCE_BACKEND=tract`.
2. Release-profile RTF table (test build is not product RTF).
3. INT8 powerset rewrite (same graph ops before QDQ) if FP32 path productizes.
4. Still out of scope: Silero nested `If` (legacy VAD only).

## Artifacts

- Generator: `scripts/export-powerset-tract.py`  
- Local model (gitignored): `models/powerset_fp32_tract.onnx`  
- Strategy: `docs/strategy/zero-deps.md`  
