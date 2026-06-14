# src/pipeline_v2 — TODO

Open items to graduate `pipeline_v2` from **experimental** to the validated
default (i.e. reverse the 0.6.1 revert). Each must land with DER evidence from
the enforced harness (roadmap task 006 / C-6) before flipping the default.

## Blockers for default (graduation gate)

- [ ] **Long-form DER parity with the legacy pipeline.** The 0.6.1 regression
      was on long-form audio (AMI-scale). Prove v2 ≤ legacy DER on VoxConverse
      (no-collar) AND AMI before changing the CLI/Python default. (tasks 016, 117)
- [ ] **Fix automatic speaker-count.** Suspected bottleneck — wire the corrected
      auto-k / single-speaker guard into the clusterer path. (task 017)
- [ ] **Modern scoring.** Pluggable scoring backend + PLDA + AS-norm to replace
      raw cosine. (tasks 114, 115, 204)
- [ ] **VBx/HMM-style resegmentation** upgrading the centroid-nearest pass.
      (task 116)

## Known gaps (correctness / feature)

- [ ] **Execution providers:** `ExecutionProvider::Nnapi` and `XnnPack` are
      config enums with NO `ort` wiring — they silently fall back to CPU. Wire
      them (and a CUDA option) and thread `config.execution_provider` through the
      builder into every ONNX session. (tasks 127–130, 215–217)
- [ ] **Calibrated binarization** of segmentation posteriors before clustering.
      (task 205)

## Hygiene

- [ ] Once v2 is the default, revisit MIN_EMBED_SECS (0.20s) — confirm it is the
      right floor under the new embedder/scoring path, not a legacy heuristic.
- [ ] Keep this contract's `status: experimental` until the graduation gate
      above is green; flipping to `stable` requires a migration lease.
