# src/pipeline_v2 — TODO

Graduation to CLI/Python/FFI default landed in **0.11** (v2 + VBx, full-split
DER gate). Items below are post-graduation hardening — not blockers for the
default path.

## Accuracy / scoring

- [ ] **Modern scoring beyond VBx.** Pluggable scoring backend + AS-norm domain
      profiles on top of PLDA (see roadmap AS-norm work).
- [ ] **VBx/HMM-style resegmentation** upgrading the centroid-nearest pass for
      overlap regions.
- [ ] **Calibrated binarization** of segmentation posteriors before clustering
      (flags exist on CLI/bench; flip defaults only with DER evidence).

## Correctness / feature

- [ ] **Execution providers:** ensure every `ExecutionProvider` variant that is
      advertised is wired into `ort` (or clearly documents silent CPU fallback)
      and is threaded through the builder into every ONNX session.
- [ ] Revisit `MIN_EMBED_SECS` (0.20s) under VBx + dense `embed_window_secs` —
      confirm the floor is still correct, not a legacy heuristic.

## Hygiene

- [ ] Fold or soft-deprecate `HybridPipeline` once dense windows + powerset
      speech regions are fully covered by main `Pipeline` knobs + docs.
- [ ] Optional: split pure orchestration from ONNX builder adapters so
      `Profile::Custom` mock tests do not need the full feature soup.
