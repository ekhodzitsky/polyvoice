# ort execution-provider migration plan

> **Historical ORT migration notes.** The core 1.0 product no longer uses
> `ort`, `ort-sys` or these execution-provider features. The tables below
> describe the former core integration. Current ORT work belongs to the
> separate [Parakeet companion](../polyvoice-asr/README.md); the native core
> accepts CPU/auto only. Do not apply the old feature recipes to core 1.x.

Status: **historical.** The body below is the old core-crate recipe (root
`Cargo.toml`, `src/onnx/mod.rs`, `cargo check --features "onnx,coreml"`).
Do not follow it for polyvoice 1.x. Core has no `ort` dependency and no
`onnx` or `coreml` feature. Current ORT work is the Parakeet companion crate
(`polyvoice-asr`); `scripts/check-ort-version.sh` checks that companion pin
only. Do not bump `ort` in the root `Cargo.toml` and do not edit a core ONNX
Runtime session — neither exists.

## Why this note exists

[pykeio/ort#599](https://github.com/pykeio/ort/pull/599) (“Make EPs less
confusing”, merged 2026-07-16) changes how execution providers are exposed and
how prebuilt ONNX Runtime binaries are selected. The next RC / stable after
rc.12 was expected to carry that work. Former core Cargo features (removed;
do not recreate them):

```toml
coreml  = ["onnx", "ort/coreml"]
nnapi   = ["onnx", "ort/nnapi"]
xnnpack = ["onnx", "ort/xnnpack"]
```

and the single session builder in `src/onnx/mod.rs` (`build_session_with_ep`)
were the only core places that had to stay in lockstep with ort’s EP surface.
That session is gone from the core crate.

## What #599 changes (from the PR)

1. **Compile-time gate EP types behind their Cargo features**  
   EP structs (e.g. `CoreMLExecutionProvider`) become unavailable unless the
   matching `ort/*` feature is enabled — same pattern as the rest of the Rust
   ecosystem. The former core `#[cfg(feature = "coreml")]` / `xnnpack` arms
   mirrored that; the planned bump was supposed to fail at compile time rather
   than only at session-build with a warn+CPU fallback for some paths. Those
   features are not on the core crate.

2. **Explicit dist feature sets for prebuilt binaries**  
   `download-binaries` resolves `(feature set, target) → (URL, SHA-256)` more
   strictly. Combinations that never had a prebuilt row will error instead of
   silently picking a near match (unless the new `lax-feature-matching` feature
   is enabled).

3. **`lax-feature-matching` opt-in**  
   Restores the old “best available prebuilt” behaviour when an exact dist row
   is missing. Prefer **not** enabling it in release builds so missing EP
   binaries fail loudly; useful only as a temporary local workaround.

## Former core wiring (rc.12, historical)

| polyvoice feature | ort feature | session code | status |
|---|---|---|---|
| `coreml` | `ort/coreml` | `CoreMLExecutionProvider` on macOS aarch64 | wired |
| `xnnpack` | `ort/xnnpack` | `XNNPACKExecutionProvider` | wired |
| `nnapi` | `ort/nnapi` | not registered yet | feature exists; falls back to CPU with warn |
| (none) | — | `ExecutionProvider::Cuda` | not a Cargo feature; warn + CPU |
| default `onnx` | no EP features | CPU only | default path |

All EP selection used to funnel through `crate::onnx::build_session_with_ep`.
That function is not in the core crate anymore. Do not reintroduce it to
satisfy this note.

## Pin strategy

| Rule | Detail |
|---|---|
| **Stay on `2.0.0-rc.12`** until a post-#599 RC is published and checklist-green | Historical core rule. The Parakeet companion still pins this via `scripts/check-ort-version.sh`. Do not add `ort` back to the root crate to satisfy the script. |
| **Single version across the workspace** | Historical: core and `polyvoice-asr` had to match. Core no longer depends on `ort`. The script checks the Parakeet companion pin only. |
| **Bump only with an intentional PR** | Never as a drive-by `cargo update` of the companion crate |
| **Re-verify native binary pins** | Update `docs/security/ort-native-binary-provenance.md` from the new `ort-sys` `dist.txt` when the companion pin changes |

When 2.0.0 stable lands, a companion-crate bump can prefer stable over another
RC if the EP API has settled. That bump does not belong in root `Cargo.toml`.

## Historical checklist (do not apply to core)

These steps targeted the removed core ONNX Runtime session. They are kept so
the old plan is readable. Do not bump `ort` in root `Cargo.toml`, do not edit
`src/onnx/mod.rs`, and do not run `cargo check --features "onnx,coreml"` —
those features are not on the core crate. A Parakeet bump, if one is ever
needed, is `ort` in `polyvoice-asr/Cargo.toml` only, then
`scripts/check-ort-version.sh`.

1. Read the target RC changelog / #599 follow-ups for EP type or feature renames.
2. The old recipe bumped `ort` in root `Cargo.toml` and `polyvoice-asr/Cargo.toml` together. Root no longer has that dependency.
3. Update `EXPECTED` in `scripts/check-ort-version.sh` only when the companion pin changes.
4. `cargo update -p ort` (and `ort-sys` if separate) and commit `Cargo.lock`.
5. The old recipe fixed compile breaks in `src/onnx/mod.rs` (and any new EP cfg gates). That file is not the core session anymore.
6. Old smoke commands, which do not apply to core 1.x (no `onnx` / `coreml` features):
   - `cargo test --lib --features "onnx,segmentation,embedder,clusterer,resegmentation"`
   - `cargo check --features "onnx,coreml"` (macOS aarch64 if available)
   - `cargo check --features "onnx,xnnpack"`
   - `cargo check --features "onnx,nnapi"` (even if still a no-op at runtime)
7. Refresh `docs/security/ort-native-binary-provenance.md` hashes for the new ORT.
8. Run `scripts/check-ort-version.sh`.
9. Note the bump in `CHANGELOG.md` under Unreleased / the release section.

## Non-goals for this plan

- Wiring NNAPI / CUDA for real (separate roadmap items).
- Enabling `lax-feature-matching` by default.
- Changing polyvoice’s `ExecutionProvider` public enum without a semver note.
