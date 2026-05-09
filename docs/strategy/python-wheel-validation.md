# Python Wheel Validation Report

**Date:** 2026-05-08
**Agent:** Kimi Code CLI
**Task:** Validate that the Python bindings build correctly into a wheel and can be imported.

---

## Summary

| Check | Status |
|-------|--------|
| `maturin` available | ✅ Yes (`/opt/homebrew/bin/maturin`) |
| `maturin build --release` | ✅ Succeeds |
| Wheel produced | ✅ Yes |
| `import polyvoice` | ✅ Succeeds (with caveats, see below) |
| `polyvoice.Pipeline.balanced()` | ✅ Succeeds (models cached) |

---

## 1. Build Results

### Command
```bash
cd python
maturin build --release
```

### Outcome
Build succeeds and produces platform-specific wheels:

- **Python 3.11:** `python/target/wheels/polyvoice-0.6.0a3-cp311-cp311-macosx_11_0_arm64.whl`
- **Python 3.14:** `python/target/wheels/polyvoice-0.6.0a3-cp314-cp314-macosx_11_0_arm64.whl`

The build was tested explicitly for both interpreters:
```bash
maturin build --release --interpreter python3.11
maturin build --release --interpreter python3.14
```

### Issue Discovered: Stale `.so` artifact shadowing

An old `_polyvoice.cpython-311-darwin.so` existed in `python/polyvoice/` (dated 2025-05-06). Because `pyproject.toml` sets `python-source = "."`, maturin packages the entire `polyvoice/` directory. The stale `.so` was being bundled into wheels, causing the installed package to expose the **old v0.5.x API** (`Turn`, callable `Pipeline`) instead of the rewritten v0.6 API (`balanced()`, `mobile()`, `run()`).

**Fix:** Remove the stale native extension before building:
```bash
rm -f python/polyvoice/_polyvoice.cpython-*.so
```

After removal and rebuild, the wheel correctly exposes the new API.

---

## 2. Import & Runtime Test

### Test Environment
```bash
cd python
uv venv .venv-test --python python3.11
uv pip install --python .venv-test/bin/python3 target/wheels/polyvoice-0.6.0a3-cp311-cp311-macosx_11_0_arm64.whl
```

### Import Test
```bash
python3 -c "import polyvoice; print('import OK')"
# => import OK
```

### `Pipeline.balanced()` Test
```bash
python3 -c "import polyvoice; p = polyvoice.Pipeline.balanced(); print('OK')"
# => OK
```

> **Note:** This test succeeds because ONNX models are already cached locally (`~/Library/Caches/polyvoice/models`). On a clean CI runner without cached models, `Pipeline.balanced()` would attempt a network download and may fail if models are unavailable or the download times out.

### Python 3.14 Compatibility
The cp314 wheel also builds and imports successfully. However, Python 3.14 is currently a release candidate (`3.14.0rc3`), so CI should probably stick to the stable 3.11–3.13 matrix until 3.14 is officially released.

---

## 3. Test Suite Mismatch

The existing Python tests in `python/tests/test_pipeline.py` still use the **old API**:

```python
# Old API (no longer present)
polyvoice.Pipeline(MODEL_DIR)
pipeline(wav_path)
pipeline(samples)
```

The rewritten `lib.rs` exposes the **new API**:

```python
# New API (current)
polyvoice.Pipeline.balanced(models_cache=...)
polyvoice.Pipeline.mobile(models_cache=...)
pipeline.run(samples, sample_rate)
```

**Consequence:** Running `pytest tests/` against the current wheel will fail.

**Proposed Fix:** Update `python/tests/test_pipeline.py` to match the new API surface:
- Replace `polyvoice.Pipeline(MODEL_DIR)` with `polyvoice.Pipeline.balanced(MODEL_DIR)`
- Replace callable `pipeline(...)` with `pipeline.run(samples, sample_rate)`
- Update `test_missing_models` to use `balanced("/nonexistent/path")`

---

## 4. CI Recommendation

The existing `.github/workflows/python.yml` already builds wheels and runs pytest, but it can be strengthened:

### Suggested additions to `.github/workflows/python.yml`:

1. **Clean stale artifacts before build:**
   ```yaml
   - name: Clean stale native extensions
     run: rm -f polyvoice/_polyvoice.cpython-*.so
   ```

2. **Add a smoke import step before pytest:**
   ```yaml
   - name: Smoke import
     run: |
       python3 -c "import polyvoice; print('import OK')"
       python3 -c "import polyvoice; print([a for a in dir(polyvoice.Pipeline) if not a.startswith('_')])"
   ```

3. **Add a dedicated wheel-build job** (matrix across OS + Python) that:
   - Builds with `maturin build --release`
   - Installs the wheel into a fresh venv
   - Runs only the smoke import test
   - Uploads the wheel as an artifact

4. **Hold off on Python 3.14** in the matrix until it reaches GA; keep `["3.11", "3.12", "3.13"]`.

5. **Cache ONNX models** in CI (or skip model-dependent tests when `POLYVOICE_MODEL_DIR` is unset) to avoid network flakiness.

---

## 5. Final Recommendation

| Priority | Action |
|----------|--------|
| **P0** | Remove stale `python/polyvoice/_polyvoice.cpython-311-darwin.so` from the repo (add to `.gitignore` if needed). |
| **P0** | Update `python/tests/test_pipeline.py` to match the new `balanced()` / `run()` API. |
| **P1** | Add a wheel-build + smoke-import job to `.github/workflows/python.yml`. |
| **P1** | Add `rm -f polyvoice/_polyvoice.cpython-*.so` as a pre-build step in CI. |
| **P2** | Cache ONNX models in CI to speed up tests and avoid download failures. |
