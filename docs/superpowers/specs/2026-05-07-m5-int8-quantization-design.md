---
title: M5 — INT8 Quantization Design
date: 2026-05-07
status: draft
milestone: M5
preceding: M0, M1, M2, M3, M4
following: M6 (Pipeline integration), M7, M8 (Android NNAPI), M9 (release polish)
authors: ekhodzitsky
---

# M5 — INT8 Quantization Design

## Problem

После M0–M4 у polyvoice есть полный pure-Rust pipeline (segmenter → embedder → clusterer → resegmenter), но все модели используются как FP32. Roadmap §10.1 требует:

- Mobile bundle ≤ 10 МБ (vs текущие ~27 МБ FP32)
- ARM-CPU ≥ 3× real-time на A78 single-core (Mobile)
- Profile API контракт `Mobile` / `Balanced` подразумевает INT8 модели как дефолтный bundle

Без INT8 модели profile mapping в `src/models/manifest.toml` ссылается на FP32 файлы и обещание "≤10 MB" rolls forward вечно. M5 закрывает это: производит INT8-квантованные ONNX-артефакты для трёх моделей, валидирует их под §9.4 budgets, публикует в GitHub Releases.

## Goal

Single milestone (~2 недели calendar work):

1. **Tooling**: повторяемые scripts для quantization + validation.
2. **Artifacts**: `powerset_int8.onnx` (~1.5 MB), `cam_pp_int8.onnx` (~2 MB), `resnet34_int8.onnx` (~6.4 MB).
3. **Validation**: per-model acceptance gates из spec §9.4.
4. **Publish**: pre-release tag `v0.6.0-alpha.2` с INT8 артефактами как release assets.
5. **Manifest**: `[models.<name>_int8]` entries + Profile mapping switch к INT8 для Mobile/Balanced.

End state: `polyvoice download-models --profile mobile` качает ~10 MB INT8 bundle с проверкой SHA-256.

## Non-goals

- E2E DER measurement на VoxConverse-test/AMI с новым INT8 pipeline — это M6 (Pipeline integration) + M9 (release gate hardening).
- iOS/Windows wheels с INT8 моделями — M8.
- New embedder backends (TitaNet, ERes2NetV2) — post-v1.0 (roadmap §6.3).
- Calibration на других datasets (DIHARD/AMI) — defer на post-M5 если budgets не достигнуты.
- Dynamic quantization, mixed-precision, FP16 — fallback paths (см. Risks), не дефолт.

## Approach

### Quantization mode: static per-channel

`onnxruntime.quantization.quantize_static`:

- **Per-channel weights** (CNN convolutions): минимизирует accuracy hit для embedders.
- **Asymmetric activations** (`QuantType.QInt8` / `QInt8`): стандарт для variable-range audio features.
- **MinMax calibration**: простая, robust к outliers если calibration set достаточно велик. Альтернатива (`Percentile`) активируется как fallback если MinMax даёт >+0.5% DER hit на segmenter.

Static выбран над dynamic, потому что:

- Все три модели — CNN-based (segmentation transformer без attention в pyannote-3.0 powerset; CAM++ — Channel-Attentive ResNet; ResNet34 — обычный ResNet). Dynamic подходит для transformer attention layers, не нужен здесь.
- Static produces real INT8 ops everywhere → ARM acceleration через XNNPACK / NNAPI работает.
- Размер static-quantized почти равен теоретическому (4× compression), dynamic — около 2.5×.

### Calibration data: VoxConverse-dev + VoxCeleb1

Per spec §9.4 (strict path):

- **Segmenter calibration**: 500 random 10-second chunks из VoxConverse-dev, seed 42. Один calibration data reader (`VoxConverseChunkReader`) для всех трёх моделей.
- **Segmenter DER validation**: 100 hold-out files из VoxConverse-dev (не использовавшиеся в calibration). Compute DER FP32 vs INT8.
- **Embedder EER validation**: VoxCeleb1 subset (1k speakers, ~1.5 GB). Если license blocks — fallback на VoxCeleb1-test (public, ~1 GB).
- **Embedder cosine vs FP32**: 200 random 3-second chunks из VoxConverse-dev. Compute mean / p1 (1-st percentile) cosine similarity FP32 → INT8.

Test set (VoxConverse-test) остаётся **sealed** — не используется ни в calibration, ни в validation.

### Validation gates (spec §9.4)

| Model | Metric | Budget |
|---|---|---|
| Powerset | DER hit on hold-out | ≤ +0.5% |
| Powerset | Max softmax KL divergence (output) | ≤ 0.05 |
| CAM++ | EER hit on VoxCeleb1 | ≤ +0.30 |
| CAM++ | Mean cosine vs FP32 | ≥ 0.998 |
| CAM++ | p1 cosine vs FP32 | ≥ 0.991 |
| ResNet34 | EER hit on VoxCeleb1 | ≤ +0.30 |
| ResNet34 | Mean cosine vs FP32 | ≥ 0.998 |
| ResNet34 | p1 cosine vs FP32 | ≥ 0.991 |

`scripts/validate-int8.sh` exit non-zero если любой gate failed → блокирует `publish-models.sh`.

### Publishing

`scripts/publish-models.sh`:

1. Проверяет calibration report exists и all gates passed.
2. Computes SHA-256 для каждого `*_int8.onnx`.
3. Создаёт `gh release create v0.6.0-alpha.2 --prerelease --notes "M5 INT8 artifacts"`.
4. Uploads three `*_int8.onnx` как release assets.
5. Updates `src/models/manifest.toml` с real SHA-256 hashes и URLs.
6. Updates `[profiles.mobile]` / `[profiles.balanced]` mappings к INT8 entries.
7. Commit "feat(models): pin v1.0 INT8 manifest, switch profiles to INT8 default".

`v0.6.0-alpha.2` — pre-release tag (M9 заменит на финальный `v1.0.0`). Manifest URL остаётся стабильным после M9 потому что v1.0.0 release будет иметь те же INT8 файлы (или regenerated с identical hashes).

## File layout

| Path | Action | Responsibility |
|---|---|---|
| `scripts/download-voxconverse-dev.sh` | create | Download VoxConverse-dev RTTM + audio (~5 GB) |
| `scripts/download-voxceleb1-subset.sh` | create | Download 1k speaker VoxCeleb1 subset (~1.5 GB) или fallback на VoxCeleb1-test |
| `scripts/quantize_models.py` | create | Python: orchestrator — static quantize FP32 → INT8 для 3 моделей, MinMax calibration |
| `scripts/quantize-models.sh` | create | bash wrapper: invokes `quantize_models.py` для трёх моделей с правильными аргументами |
| `scripts/validate_int8.py` | create | Python: computes DER FP32 vs INT8 на VoxConverse-dev hold-out, EER + cosine на embedders |
| `scripts/validate-int8.sh` | create | bash wrapper, exit non-zero if any acceptance gate failed |
| `scripts/publish-models.sh` | create | bash: SHA-256, `gh release create`, upload assets, update manifest |
| `scripts/release-gate.sh` | modify | Switch M5 stubs to real budget checks; tighten "Mobile bundle ≤ 10 МБ" |
| `src/models/manifest.toml` | modify | Add `[models.powerset_int8]`, `[models.cam_pp_int8]`, `[models.resnet34_int8]`; switch `[profiles.mobile]` and `[profiles.balanced]` к INT8 |
| `docs/calibration/2026-MM-DD-int8-validation.md` | create | Calibration report: per-model FP32→INT8 metrics, calibration set ID, ONNX runtime version, host CPU |
| `docs/strategy/m5-quantization-notes.md` | create | Engineering notes: квантизатор chosen, fallback decisions taken, how to re-run если models change |
| `tests/m5_manifest_smoke_test.rs` | create | Rust integration test: загружает обновлённый manifest, проверяет что `[models.<name>_int8]` discoverable, dimensionality сохранена через Profile API |
| `Cargo.toml` | unchanged | M5 не добавляет crate dependencies — только Python tooling |
| `python/requirements-dev.txt` | create or modify | Pin `onnxruntime`, `onnxruntime-tools` (если нужно), `pyannote.metrics`, `numpy`, `librosa` (для audio loading), `pyannote.database` (для VoxConverse) |
| `CHANGELOG.md` | modify | M5 section в `[Unreleased]` |

Total roughly 1500 LOC Python + ~300 LOC bash + ~150 LOC Rust test + ~250 lines markdown reports.

## Acceptance criteria

1. `bash scripts/quantize-models.sh` exit 0 → produces three valid `*_int8.onnx` files в `models/int8/`.
2. `bash scripts/validate-int8.sh` exit 0 → all spec §9.4 budgets met. Report saved as `docs/calibration/2026-MM-DD-int8-validation.md`.
3. `bash scripts/publish-models.sh` создаёт `v0.6.0-alpha.2` GitHub pre-release с тремя INT8 .onnx assets.
4. `src/models/manifest.toml` обновлён: real SHA-256 / URL / size для INT8 entries; `[profiles.mobile]` / `[profiles.balanced]` переключены к INT8.
5. `polyvoice download-models --profile mobile` качает 3 файла (silero_vad FP32 ~1.5 MB + powerset_int8 + cam_pp_int8) общим объёмом ≤ 10 МБ. Все SHA-256 verified.
6. `polyvoice download-models --profile balanced` качает silero_vad + powerset_int8 + resnet34_int8 ≤ 35 МБ.
7. `cargo test --features download` зелёный (включая новый smoke test).
8. `cargo clippy --all-targets --all-features -- -D warnings` clean.
9. `cargo fmt --check` clean.
10. `release-gate.sh` exit 0 для трёх обновлённых rows: "Mobile bundle ≤ 10 МБ", "Balanced bundle ≤ 35 МБ", "INT8 calibration validation". DER VoxConverse rows остаются `pending` до M6.

## Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| INT8 quantization gives > +0.5% DER hit на CAM++ или Powerset | medium | Spec §10.3 row: fallback на FP16 для embedder, INT8 только для segmenter. Документируем decision в calibration report. CAM++ может остаться FP32 в Mobile profile если INT8 hit > budget; bundle size грамма в этом случае ~12 MB (over budget by 2 MB) — acknowledged trade-off. |
| VoxCeleb1 license blocks download | medium | Fallback на VoxCeleb1-test (public ~1 GB, ~40 speakers). Документируем как known scope reduction в calibration report. |
| VoxConverse-dev download fails или меняется upstream | low | Pin specific release URL of dev split. Если upstream удаляет — host snapshot в release assets `v0.6.0-alpha.2-data` или внутренний bucket. |
| sherpa-onnx pyannote-segmentation-3-0 ONNX format меняется upstream | low | Pin release tag в `[models.powerset_fp32]`. Regen INT8 каждый раз когда FP32 manifest обновляется. |
| `quantize_static` crashes на specific operator (e.g. unsupported attention pattern) | medium | Use `nodes_to_exclude` параметр чтобы skip проблемные layers. Документируем в calibration notes. Fallback chain: per-channel → per-tensor → dynamic для проблемного layer. |
| ARM-CPU runtime DAG не выгодно для INT8 (некоторые операторы slower в INT8 чем FP32) | medium | M5 фиксирует только размер; perf measurement в M8 (Android RT-bench через QEMU). Если INT8 быстрее не везде — leave FP16 hybrid в M8. |
| Calibration timing: VoxCeleb1 download + 500-sample calibration занимают несколько часов | low | Run в background, document в M5 plan что время на full calibration sweep ~2-4 hours per model. Cache calibration tensors в pickle для re-runs. |
| `gh release create` fails из-за authentication (`GITHUB_TOKEN` not set) | low | `publish-models.sh` checks env vars first и exits early с actionable error. README documents the required token scope (`repo` для public, `repo:write` для private). |

## Out of scope follow-ups

- M6 wires Pipeline к INT8 default + measures end-to-end DER VoxConverse. Если DER hit > spec acceptance — M5 calibration revisited.
- M9 (release polish) re-publishes INT8 artifacts как `v1.0.0` final release с identical content + new release notes. Manifest URLs автоматически migrate (через `publish-models.sh --tag v1.0.0`).
- Quantization для streaming (online) pipeline не включён — `OnlineDiarizer` deprecated в v1.0, переезжает в v1.1.

## Dependencies on previous milestones

- **M0**: ModelRegistry + manifest infrastructure (this milestone modifies the manifest, не reengineers).
- **M1**: Powerset segmenter ONNX path (this milestone quantizes the same model).
- **M2**: CAM++ embedder через trait (this milestone quantizes its FP32).
- **M3+M4**: not consumed by M5; M5 doesn't touch clusterer / resegmenter code.
- **VoxConverse-dev** + **VoxCeleb1**: external datasets, downloaded via new scripts.

## Tests

| Test | Location | Coverage |
|---|---|---|
| `m5_manifest_smoke_test::resolves_int8_paths_for_mobile_profile` | `tests/m5_manifest_smoke_test.rs` | Mobile profile resolves to INT8 entries; embedding_dim() returns 192 |
| `m5_manifest_smoke_test::resolves_int8_paths_for_balanced_profile` | tests/m5_manifest_smoke_test.rs | Balanced profile resolves to INT8 entries; embedding_dim() returns 256 |
| `m5_manifest_smoke_test::sha256_format_validates` | tests/m5_manifest_smoke_test.rs | All INT8 entries have 64-char hex SHA-256 (not "..." placeholder) |
| `m5_manifest_smoke_test::int8_size_under_bundle_budget` | tests/m5_manifest_smoke_test.rs | Sum of Mobile profile INT8 sizes ≤ 10_000_000 bytes |
| `scripts/test-quantize-smoke.py` | python/tests/ | Python: smoke test that quantize_models.py loads ONNX, выдает INT8 with reduced size — runs on a tiny synthetic ONNX |
| `scripts/test-validate-smoke.py` | python/tests/ | Python: smoke test что validate_int8.py correctly fails when delta exceeds budget |

Smoke tests (Rust + Python) runs в CI matrix (no datasets needed). Real calibration / validation runs запускаются manually + в release pipeline (отдельно от PR CI).

## Decomposition note

M5 deliverables tightly coupled (calibration → validation → publish → manifest). Single PR + single milestone tag `m5-complete`. Estimated 2 weeks calendar work matching roadmap §10.1. If timeline slips beyond 3 weeks, decomposition options:

- M5a: tooling only (quantize + validate scripts), local artifacts
- M5b: publishing + manifest update

Default plan keeps single PR.

## References

- Roadmap: `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §6 (manifest), §9.4 (INT8 calibration validation), §10.1 (M5 row)
- M0 plan: `docs/superpowers/plans/2026-05-07-m0-plumbing-and-registry-plan.md`
- ONNX Runtime quantization: <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
- sherpa-onnx pyannote segmentation: <https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models>
- WeSpeaker pretrained: <https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md>
- Powerset paper: Plaquet & Bredin, INTERSPEECH 2023, [arXiv:2310.13025](https://arxiv.org/html/2310.13025v1)

## Open questions (closed)

- ✅ Variant B chosen (quantize all ourselves, including powerset) over Variant A (mixed) and Variant C (deferred publishing).
- ✅ Full publish — INT8 артефакты идут в `v0.6.0-alpha.2` pre-release сразу.
- ✅ Strict calibration path (VoxConverse-dev + VoxCeleb1) over pragmatic / synthetic.
- ✅ Static quantization (per-channel weights, asymmetric activations, MinMax) — default; Percentile / dynamic — fallback.

## Follow-ups

1. После одобрения spec: invoke `superpowers:writing-plans` для генерации M5 implementation plan в `docs/superpowers/plans/2026-05-07-m5-int8-quantization-plan.md`. Стиль M3/M4 plan: TDD-задачи (где applicable), atomic commits per task, git tag `m5-complete`.
2. После M6: запустить `polyvoice-bench --profile mobile` на VoxConverse-test и зафиксировать end-to-end DER в `tests/der_baseline.json`. Если delta > acceptance — pin Percentile calibration или per-tensor quantization, regen INT8.
