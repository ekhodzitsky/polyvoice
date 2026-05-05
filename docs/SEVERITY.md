# SEVERITY.md — Issue Classification for polyvoice

## CRITICAL — Data loss, security, or crash

- Panic in production path
- unwrap/expect without proof
- Memory unsafety (use-after-free, data race)
- Incorrect speaker assignment that leaks private information

## MAJOR — Incorrect behavior, API breakage

- Missing precondition checks
- Contract violation in public API
- Breaking change without version bump
- Missing error handling for I/O or ONNX failures

## MINOR — Performance, maintainability, style

- Clone where reference suffices
- Suboptimal algorithmic complexity
- Missing documentation
- Dead code

## INFO — Suggestions, future work

- Additional tests could cover more cases
- Alternative design patterns
- Performance optimizations with benchmarks needed

---

## Decision Matrix

| Severity | Action | Examples |
|----------|--------|----------|
| CRITICAL | Block merge, fix immediately | unwrap in lib.rs, use-after-free in ffi.rs |
| MAJOR | Must fix before release | Missing Result propagation, API contract break |
| MINOR | Fix in next sprint | Redundant clone, missing docs |
| INFO | Track in backlog | Benchmark different clustering algorithms |
