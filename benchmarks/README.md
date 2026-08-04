# polyvoice cross-engine diarization benchmark

Honest, reproducible comparison of polyvoice against open-source speaker-diarization
engines. **Every engine emits an RTTM hypothesis that is scored through one
implementation ([`der.py`](der.py)) at one disclosed collar**, so the rows are
like-for-like. Diarization Error Rate (DER) is meaningless without a stated
collar — compare systems only on a matched collar and a matched overlap policy.

> **Competitor rows are cited, not yet re-measured here.** Running pyannote /
> WhisperX / diart like-for-like needs PyTorch and the *gated*
> `speaker-diarization-3.1` weights (an HF token + license acceptance); sherpa-onnx
> needs its ONNX models. The runners under [`runners/`](runners/) are wired and
> skip-if-absent, so when that stack is present the competitor rows fill in
> automatically. Until then `docs/BENCHMARKS.md` cites each competitor's
> **published** numbers with the source URL and their original protocol (which is
> usually NOT collar-matched to ours). Treat cross-source numbers as indicative,
> not head-to-head.

## What is measured

- **DER** (lower is better) `= (miss + false_alarm + confusion) / scored_reference_speech`,
  the canonical NIST `md-eval` 10 ms-frame model.
- **Decomposition** — miss / false-alarm / confusion, separately. Over-clustering
  shows up as confusion; missed overlap shows up as miss.
- **Speaker-count accuracy** — exact / ±1 / off-by-2-or-more, per file. A system
  can have a low DER while badly miscounting speakers; this exposes it.
- **RTF** (real-time factor = processing ÷ audio; lower is faster). See the timing
  caveat below.

Each metric is reported at **two collars** in one run: the **0.25 s forgiveness
collar** (the lenient convention) and **no-collar** (collar 0, strict). Overlap is
**scored** by default (a frame may have several active speakers), matching the
strict protocol pyannote 3.1 reports against; pass `--skip-overlap` for the
overlap-excluded variant. Both **micro** (frame/duration-weighted across files)
and **macro** (mean of per-file DER) aggregates are emitted, with a **95 % bootstrap
confidence interval** (1000 resamples of files, deterministic LCG).

## Methodology

### One scorer, cross-checked
`der.py` scores every engine identically. Its frame model and Hungarian-optimal
speaker mapping match `md-eval` / `pyannote.metrics` semantics, and its polyvoice
numbers have been cross-checked against the in-tree `polyvoice-bench` Rust DER on
the shipped splits. The reference and hypothesis go through the exact same
collar/overlap/mapping logic — there are no per-engine branches.

### Collar discipline
pyannote 3.1's published VoxConverse/AMI numbers use **no forgiveness collar** and
**score overlap**. To compare against them, read the **no-collar** column. The
0.25 s-collar column is for comparison against systems that report with a collar.
Never mix the two.

### Timing / RTF caveat
polyvoice is driven through its shipped **CLI**, which cold-loads the model on
every file, so the harness RTF includes per-file process spawn + model load and is
a **lower bound on speed** (a conservative, worst-case number). polyvoice's
steady-state, in-process realtime factor (~10× realtime on CPU) is measured
separately by the Rust harness and reported in `docs/BENCHMARKS.md`. Competitor
RTFs, when run, are measured the same end-to-end way for fairness.

### Failure handling
If an engine errors on a file, an empty hypothesis is written and the file counts
as a full miss (visible in the `failures` counter), instead of being silently
dropped from the denominator.

### Datasets
Generated into `manifests/` by `make_manifests.py` from the local `data/` splits;
audio is **not** redistributed (download with `../scripts/download-*.sh`). See
[`DATA_LICENSE`](DATA_LICENSE).

| Manifest | Split | Files | Notes |
|---|---|---|---|
| `voxconverse_dev` | VoxConverse dev | 216 | tuning split — never reported as the headline |
| `voxconverse_test` | VoxConverse test | 232 | the headline held-out split |
| `ami_test` | AMI test (Mix-Headset) | 16 | long meetings, heavy overlap |

## Reproduce

```bash
cd benchmarks
python make_manifests.py                 # (re)build manifests from data/
# polyvoice only (no extra installs needed):
python benchmark.py --dataset voxconverse_test --runners polyvoice,polyvoice-v2
# everything the host has installed:
python benchmark.py --dataset voxconverse_test --runners all
# score an arbitrary hypothesis RTTM against a reference, directly:
python der.py ../data/voxconverse-test/rttm hyp_dir --collar 0
```

Enabling competitors (each skips until satisfied):

| Engine | Needs |
|---|---|
| `pyannote`, `whisperx`, `diart` | `pip install pyannote.audio torch` (+ `whisperx` / `diart`); `HF_TOKEN` with `speaker-diarization-3.1` access |
| `sherpa-onnx` | `pip install sherpa-onnx soundfile`; `SHERPA_SEGMENTATION_MODEL` + `SHERPA_EMBEDDING_MODEL` |
| `speakrs-cpu` / `speakrs-coreml` / `speakrs-coreml-fast` | build `benchmarks/tools/speakrs-rttm` (path dep on [speakrs](https://github.com/avencera/speakrs)); optional `SPEAKRS_MODELS_DIR` |

## Output

`results/<dataset>__*.json` (schema `polyvoice-diarization-benchmark-v1`): per-engine
DER at both collars (micro/macro + CI), miss/fa/confusion decomposition,
speaker-count accuracy, RTF, failures, plus host/dataset metadata. Per-file
hypothesis RTTMs are cached under `results_full/<dataset>/<engine>/` and reused on
re-runs (pass `--no-cache` to force).
