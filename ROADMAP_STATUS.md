# ROADMAP_STATUS — dependency + toolchain hygiene

## Done

| Item | Result |
|---|---|
| Drop `crossbeam-queue` | Replaced three pools (`onnx`, `ecapa`, `embedder`) with shared `crate::utils::ObjectPool` (`Mutex<Vec<T>>`): blocking checkout, return on Drop. Pool sizes / warmup unchanged. |
| Drop `fastrand` | In-tree `XorShift64Star` (xorshift64*) used in `kmeans` and legacy `spectral` k-means++. Seed *inputs* unchanged; draw sequence differs from fastrand (documented in kmeans). |
| `rust-version` | Bumped to `1.88.0` in root + `polyvoice-asr`. CI MSRV job pins `1.88.0` and checks lib + ort feature set. CHANGELOG Unreleased notes the MSRV bump. |
| ort EP migration plan | `docs/ort-ep-migration.md` — pin stays `2.0.0-rc.12`; checklist for post-#599 RC. No ort bump. |
| Silero VAD docs | Module comment + `download-models.sh` say v6-generation with pinned sha context. Mirror **not published** (no fake URL). Runbook: `scripts/mirror-silero-vad.md`. |

## Deferred (human)

| Item | Why |
|---|---|
| Publish Silero ONNX as a GitHub Release asset | Needs repo credentials + hash re-verify; then flip `manifest.toml` `url` to the asset (see `scripts/mirror-silero-vad.md`). |
| Optional `fallback_url` in manifest | Not implemented — prefer publish-then-point over inventing URLs / expanding download API without a live mirror. |
| Fix pre-existing clippy `-D warnings` noise | `collapsible_if` / `manual_is_multiple_of` in cluster/der/overlap/resegmentation/vad/silero — present without this work; not fixed surgically. |
| ort RC bump past rc.12 | Explicitly out of scope until EP checklist in `docs/ort-ep-migration.md` is green. |

## Verify commands

```bash
cd /Users/ekhodzitsky/Documents/personal/polyvoice/.worktrees/task-408-hygiene

# Must be empty for normal deps (fastrand may still appear via tempfile → dev-deps only)
cargo tree -e normal | rg "crossbeam-queue|fastrand" || true

cargo test --lib --features "onnx,segmentation,embedder,clusterer,resegmentation"
cargo test --lib --features "spectral"
cargo test --test loom_pool --features "onnx,segmentation,embedder,clusterer,resegmentation"
cargo test --test property_kmeans_test --features "clusterer"
cargo check --lib --no-default-features
# optional if target installed:
# cargo check --target wasm32-unknown-unknown --no-default-features

# Clippy: pre-existing failures under -D warnings on this toolchain (see Deferred)
cargo clippy --all-targets --features "onnx,segmentation,embedder,clusterer,resegmentation" -- -D warnings
```

## Results (this worktree)

- `cargo tree -e normal | rg "crossbeam-queue|fastrand"` → empty
- `cargo test --lib` (ort feature set) → **251 passed**
- `cargo test --lib --features spectral` → **159 passed**
- `cargo test --test loom_pool` → **2 passed**
- `cargo test --test property_kmeans_test` → **3 passed**
- `cargo check --lib --no-default-features` → ok
- clippy `-D warnings` → fails on pre-existing lints only (none introduced in pool/PRNG/MSRV files)
