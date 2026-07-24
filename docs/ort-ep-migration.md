# ort execution-provider migration plan

Status: plan only — **do not bump ort past `2.0.0-rc.12` without executing the
checklist below**. Pin enforced by `scripts/check-ort-version.sh`.

## Why this note exists

[pykeio/ort#599](https://github.com/pykeio/ort/pull/599) (“Make EPs less
confusing”, merged 2026-07-16) changes how execution providers are exposed and
how prebuilt ONNX Runtime binaries are selected. The next RC / stable after
rc.12 is expected to carry that work. Our Cargo features:

```toml
coreml  = ["onnx", "ort/coreml"]
nnapi   = ["onnx", "ort/nnapi"]
xnnpack = ["onnx", "ort/xnnpack"]
```

and the single session builder in `src/onnx/mod.rs` (`build_session_with_ep`)
are the only places that need to stay in lockstep with ort’s EP surface.

## What #599 changes (from the PR)

1. **Compile-time gate EP types behind their Cargo features**  
   EP structs (e.g. `CoreMLExecutionProvider`) become unavailable unless the
   matching `ort/*` feature is enabled — same pattern as the rest of the Rust
   ecosystem. Our `#[cfg(feature = "coreml")]` / `xnnpack` arms already mirror
   that; after the bump, missing features should fail at compile time rather
   than only at session-build with a warn+CPU fallback for some paths.

2. **Explicit dist feature sets for prebuilt binaries**  
   `download-binaries` resolves `(feature set, target) → (URL, SHA-256)` more
   strictly. Combinations that never had a prebuilt row will error instead of
   silently picking a near match (unless the new `lax-feature-matching` feature
   is enabled).

3. **`lax-feature-matching` opt-in**  
   Restores the old “best available prebuilt” behaviour when an exact dist row
   is missing. Prefer **not** enabling it in release builds so missing EP
   binaries fail loudly; useful only as a temporary local workaround.

## Current polyvoice wiring (rc.12)

| polyvoice feature | ort feature | session code | status |
|---|---|---|---|
| `coreml` | `ort/coreml` | `CoreMLExecutionProvider` on macOS aarch64 | wired |
| `xnnpack` | `ort/xnnpack` | `XNNPACKExecutionProvider` | wired |
| `nnapi` | `ort/nnapi` | not registered yet | feature exists; falls back to CPU with warn |
| (none) | — | `ExecutionProvider::Cuda` | not a Cargo feature; warn + CPU |
| default `onnx` | no EP features | CPU only | default path |

All EP selection funnels through `crate::onnx::build_session_with_ep` — keep it
that way so a future API rename is one edit.

## Pin strategy

| Rule | Detail |
|---|---|
| **Stay on `2.0.0-rc.12`** until a post-#599 RC is published and checklist-green | Avoids surprise EP / dist breakage mid-release |
| **Single version across the workspace** | `scripts/check-ort-version.sh` + `polyvoice-asr` pin must match core |
| **Bump only with an intentional PR** | Never as a drive-by `cargo update` |
| **Re-verify native binary pins** | Update `docs/security/ort-native-binary-provenance.md` from the new `ort-sys` `dist.txt` |

When 2.0.0 stable lands, prefer stable over another RC if the EP API has
settled; until then, pin the newest RC that passes the checklist.

## Migration checklist (when bumping ort)

1. Read the target RC changelog / #599 follow-ups for EP type or feature renames.
2. Bump `ort` in root `Cargo.toml` and `polyvoice-asr/Cargo.toml` to the same version.
3. Update `EXPECTED` in `scripts/check-ort-version.sh`.
4. `cargo update -p ort` (and `ort-sys` if separate) and commit `Cargo.lock`.
5. Fix compile breaks in `src/onnx/mod.rs` only (and any new EP cfg gates).
6. Smoke:
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
