# Powerset → tract re-export spike (2026-08-12)

## Goal

Unblock pure-Rust **v2 segmentation** by making powerset ONNX loadable in
`tract` (feature `backend-tract`).

## Result: **PARTIAL SUCCESS** → pipeline wired (opt-in)

| Check | Status |
|-------|--------|
| ort numerical parity (rewrite vs original) | **PASS** max-abs ~1e-5 @ T=160000 |
| tract **load** of rewrite | **PASS** (concrete `[1,1,160000]` fact) |
| tract **run** + ort/tract parity | **PASS** (test profile) |
| Dynamic T under tract symbols | **FAIL** (still needs concrete T) |
| Pipeline path (`POLYVOICE_INFERENCE_BACKEND=tract`) | **Wired** (remap + N=1; not product default) |
| Shipping as product default | **No** (fixed-T; measure RTF/DER first) |

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

## Pipeline wiring (landed)

With feature `backend-tract` and `POLYVOICE_INFERENCE_BACKEND=tract` (or
`InferenceBackend::force(Tract)`):

1. `PowersetSegmenter::with_config` remaps a shipping powerset path to sibling
   `powerset_fp32_tract.onnx` when that file exists (also checks parent of
   `int8/`).
2. Forces session **pool=1** and micro-batch **N=1** (concrete `[1,1,T]`).
3. Product windows already zero-pad to `window_samples()` (10 s → 160000).
4. ResNet34 / other stages already go through `build_session_with_ep` → tract.

```bash
python3 scripts/export-powerset-tract.py --verify
cargo test --lib --features "onnx,segmentation,backend-tract" tract_backend_segments -- --nocapture
```

## Follow-ups

1. ~~Release-profile RTF + DER smoke~~ — see
   [`../powerset-tract-rtf-der-2026-08-12/NOTES.md`](../powerset-tract-rtf-der-2026-08-12/NOTES.md)
   (~9× slower RTFx; +35 pp DER on 3-file smoke; rewrite OK under ort).
2. 10 s ort/tract logit agreement; then re-gate DER.
3. INT8 powerset rewrite only if FP32 path productizes.
4. Still out of scope: Silero nested `If` (legacy VAD only).

## Artifacts

- Generator: `scripts/export-powerset-tract.py`  
- Local model (gitignored): `models/powerset_fp32_tract.onnx`  
- Strategy: `docs/strategy/zero-deps.md`  
