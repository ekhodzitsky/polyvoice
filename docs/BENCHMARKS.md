# Benchmarks

Honest, reproducible Diarization Error Rate (DER) for polyvoice, with a
collar-disclosed, protocol-annotated comparison to open-source diarizers.

**DER is meaningless without a stated collar and overlap policy.** Every number
below is labelled with its collar (0.25 s "forgiveness" vs **0** "strict") and
whether overlapping speech was scored. polyvoice's numbers are produced by
[`polyvoice-bench`](../src/bin/polyvoice-bench.rs) and reproduced by the
cross-engine [`benchmarks/`](../benchmarks/) harness, whose single DER scorer
([`benchmarks/der.py`](../benchmarks/der.py)) agrees with `polyvoice-bench` to
~0.02 pp. **Competitor numbers are their own published figures** (sourced below),
measured under their own protocols — they are NOT yet re-run through our scorer.
The `benchmarks/` runners are wired to re-measure them like-for-like once their
(often gated) stacks are installed; until then, compare only within a matched
collar.

## At a glance

| | polyvoice (legacy) | pyannote 3.1 | WhisperX | NeMo Sortformer |
|--|--|--|--|--|
| **VoxConverse-test DER** | 18.5 % ¹ | **11.3 %** ¹ | 11.3 % (= pyannote) | not published |
| **Model size** | **~30 MB** | ~32.5 MB | ~32.5 MB + Whisper | 123 M params |
| **Runtime** | **CPU, ~10× realtime** | CPU/GPU (PyTorch) | GPU recommended | GPU |
| **Weights** | **MIT, ungated** | MIT code, **gated** (HF token) | gated (pyannote) | **CC-BY-NC** (non-commercial) |
| **Dependencies** | **pure Rust, no Python** | PyTorch | PyTorch + Whisper | PyTorch / NeMo |
| **Bindings** | **Rust / Python / C / CLI** | Python | Python | Python |
| **Streaming** | **Yes** | No | No | No |

¹ VoxConverse-test, **no forgiveness collar (collar 0), overlap scored** — the
strict protocol pyannote 3.1 reports against, so these two are collar-matched.
polyvoice trails the accuracy leader by ~7 DER points and trades that for
deployability: a pure-Rust, CPU, MIT, **ungated** engine with four bindings and
streaming. It is **not** the accuracy leader.

## The collar caveat (read this first)

A 0.25 s forgiveness collar removes ~3–8 DER points versus collar 0 on the same
audio, because errors near every speaker boundary are forgiven. **pyannote 3.1
and diart publish at collar 0; NeMo's CALLHOME and most toolkit recipes use 0.25.**
Treat collar-0 and collar-0.25 as two different leaderboards. Where we compare to
pyannote we use polyvoice's **no-collar** number.

## Accuracy — VoxConverse test (232 files)

**Collar 0, overlap scored, Hungarian mapping** (the pyannote 3.1 protocol).

| System | DER % | miss | FA | conf | collar | source |
|---|---|---|---|---|---|---|
| pyannote.audio 3.1 | **11.3** | 3.4 | 4.1 | 3.8 | 0 | model card ² |
| pyannote community-1 | 11.2 | — | — | — | 0 | pyannote benchmark ⁷ |
| speakrs (Rust + ONNX, Apache-2.0) | 11.1 | — | — | — | 0 | speakrs README ⁸ |
| VBx (offline baseline) | 11.1 | 4.6 | 3.1 | 3.4 | 0 | diart paper ³ |
| 3D-Speaker toolkit | 11.75 | — | — | — | unstated | repo ⁴ |
| **polyvoice (legacy, shipped)** | **18.54** | 4.49 | 3.19 | 4.99 | 0 | this repo ⁵ |
| diart (online, 5 s latency) | 16.8 | 4.9 | 3.8 | 8.2 | 0 | diart paper ³ |
| diart (online, 1 s latency) | 20.1 | 3.3 | 5.1 | 11.7 | 0 | diart paper ³ |

For reference, polyvoice (legacy) at the **0.25 s collar** is **12.91 %** (macro
12.66; miss 4.49 / FA 3.19 / conf 4.99) — but no collar-0.25 pyannote number is
published, so do not compare that figure across systems.

## Accuracy — AMI test (16 meetings, Mix-Headset)

**Collar 0, overlap scored.** AMI is long-form with heavy overlap; miss
dominates, and speaker mis-counting drives confusion.

| System | DER % | miss | FA | conf | setup | source |
|---|---|---|---|---|---|---|
| DiariZen (WavLM, CC-BY-NC) | 15.4 | — | — | — | SDM | 3D-Speaker eval ⁴ |
| pyannote community-1 | 17.0 | — | — | — | IHM | pyannote benchmark ⁷ |
| pyannote.audio 3.1 | **18.8** | 9.5 | 3.6 | 5.7 | Mix-Headset | model card ² |
| pyannote community-1 (SDM) | 19.9 | — | — | — | SDM | pyannote benchmark ⁷ |
| 3D-Speaker toolkit | 21.76 | — | — | — | SDM | repo ⁴ |
| pyannote 3.1 (SDM) | 22.4 | 11.2 | 3.8 | 7.5 | array1-ch1 | model card ² |
| VBx (offline baseline) | 24.1 | 17.2 | 3.1 | 3.8 | — | diart paper ³ |
| diart (online, 5 s) | 27.5 | 10.0 | 5.0 | 12.4 | headset | diart paper ³ |
| **polyvoice (legacy, shipped)** | **32.87** | — | — | — | Mix-Headset | this repo ⁵ |

polyvoice (legacy) at the 0.25 s collar: **25.20 %** (macro 24.75).

## polyvoice speaker-count & error decomposition

A low DER can hide bad speaker counting; we report it explicitly.

| Split | DER (collar 0) | miss | FA | conf | spk exact | spk ±1 | spk off-by-2+ |
|---|---|---|---|---|---|---|---|
| VoxConverse-test (232) | 18.54 % | 4.49 | 3.19 | 4.99 | 57 | 46 | 129 |
| AMI-test (16) | 32.87 % | — | — | — | 1 | 0 | 15 |

The dominant error mode is **over-/mis-counting speakers** (confusion + the
off-by-2+ tail), especially on long meetings — the honest weak spot versus
pyannote's end-to-end segmentation. Reducing it is the active accuracy work
(pipeline v2 + the VBx clusterer below).

## Experimental pipelines: v2 and VBx

polyvoice ships **legacy** as the default; pipeline **v2** (powerset segmentation
→ ResNet34 → clustering, CLI `--v2`) and the opt-in **VBx** clusterer (VB-HMM +
PLDA, automatic speaker count; `vbx` feature) are experimental. The numbers below
are **macro DER on deterministic 60-file seed-42 subsets** of VoxConverse (full
16-meeting AMI), each with a 95 % bootstrap CI — gigastt-style slices, run sharded
across cores. Legacy rows are the **full** splits (from `der_baseline.json`), so
legacy-vs-v2 comparisons are indicative, not strictly matched.

**No-collar macro DER % (95 % CI):**

| Pipeline | VoxConverse-test | VoxConverse-dev | AMI |
|---|---|---|---|
| legacy (shipped, full split) | **18.54** | — | **32.87** |
| v2 + AHC (subset) | 20.4 [16.4–24.9] | 13.2 [10.5–16.1] | 35.8 [29.8–41.9] |
| v2 + VBx (subset) | **17.0** [13.5–20.8] | 13.6 [10.6–16.9] | 37.0 [31.0–43.1] |

**VoxConverse-test decomposition & speaker count (collar 0):**

| Pipeline | DER | miss | FA | conf | spk exact | spk off-by-2+ |
|---|---|---|---|---|---|---|
| legacy (232 files) | 18.54 | 4.49 | 3.19 | 4.99 | 57/232 | 129/232 |
| v2 + AHC (60) | 20.37 | 3.15 | 1.11 | **11.73** | 9/60 | 43/60 |
| v2 + VBx (60) | 16.99 | 3.13 | 1.12 | **8.35** | 10/60 | 44/60 |

**Honest reading:**

- **v2 + AHC over-clusters on held-out test** — confusion (11.73) is the dominant
  error and it trails legacy. The powerset front-end lowers miss/FA but produces
  many embeddings that fixed-threshold AHC splits into spurious speakers.
- **v2 + VBx fixes most of that.** Its automatic speaker count cuts test
  confusion 11.73 → 8.35 and lands at **17.0 % no-collar [13.5–20.8]**, whose CI
  overlaps legacy's 18.54 % — i.e. VBx is **competitive with legacy** on
  conversational audio and is the best v2 clusterer. (On the dev subset the two
  v2 clusterers tie; VBx's edge shows on the held-out test.)
- **AMI (long-form, heavy overlap) still favours legacy.** Both v2 variants trail
  it; miss dominates and automatic speaker counting does not help here.
- **Caveat:** v2/VBx are 60-file subsets with wide CIs; "competitive" means
  overlapping CIs, not a measured win. Legacy remains the robust default; VBx is
  the most promising path on conversational audio and the focus of ongoing
  accuracy work. Reproduce with
  [`benchmarks/bench_subset.py`](../benchmarks/bench_subset.py).

## Speed — real-time factor (RTF, CPU; lower = faster)

| Engine | RTF | Notes |
|---|---|---|
| **polyvoice (legacy)** | **~0.10** (~10× realtime) | pure-Rust, CPU; 9.3× average on a VoxConverse subset ⁶ |
| pyannote 3.1 | not published | PyTorch; GPU recommended for throughput |
| WhisperX | > 1 on CPU | Whisper + pyannote; GPU recommended |
| NeMo Sortformer | GPU-only | ~48 GB GPU for ~12-min recordings |

The cross-engine harness measures polyvoice end-to-end through its CLI (which
cold-loads the model per file), a conservative lower bound; the ~10× figure is
the steady-state in-process number.

## Footprint, license & gating

| Engine | Deployable size | License | Gated weights? | Runtime |
|---|---|---|---|---|
| **polyvoice** | **~30 MB** | **MIT** | **No** | pure Rust, CPU |
| pyannote 3.1 | ~32.5 MB (seg 5.9 + embed 26.6) | MIT code | **Yes** (HF token + accept) | PyTorch, CPU/GPU |
| WhisperX | ~32.5 MB + Whisper model | BSD code | Yes (pyannote) | PyTorch, GPU |
| sherpa-onnx | seg ~5.9 MB + embed (int8 avail.) | Apache-2.0 | No | ONNX, CPU |
| NeMo Sortformer | 123 M params | **CC-BY-NC** | No token, **non-commercial** | PyTorch, GPU |
| diart | pyannote@2021 models | MIT code | Yes (pyannote) | streaming, CPU/GPU |

polyvoice's footprint is **comparable** to pyannote's (both ~30 MB) — not a size
win there — but it is far smaller than the WhisperX/NeMo stacks, and uniquely
**MIT + ungated + pure-Rust + CPU**. sherpa-onnx is the closest architectural
peer (ONNX, CPU) but publishes **no DER**.

## Datasets

| Dataset | Files | Source | License |
|---|---|---|---|
| VoxConverse dev / test | 216 / 232 | [voxconverse](https://github.com/joonson/voxconverse) | annotations CC-BY-4.0; audio from YouTube (not redistributed) |
| AMI test (Mix-Headset) | 16 | [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | CC-BY-4.0 |

Audio is downloaded by `scripts/download-*.sh`; this repo redistributes none of
it. See [`benchmarks/DATA_LICENSE`](../benchmarks/DATA_LICENSE).

## Reproduce

```bash
# polyvoice, full decomposition (Rust harness, the source of our numbers):
scripts/run-der-sweep.sh                       # dev/test/AMI, collar 0.25 + no-collar in one run

# cross-engine, single scorer (like-for-like when competitors are installed):
cd benchmarks && python make_manifests.py
python benchmark.py --dataset voxconverse_test --runners all
```

## Caveats & protocol notes (why cross-source numbers are not head-to-head)

1. **Collar** is the biggest confound: pyannote/diart at 0; NeMo CALLHOME and
   most recipes at 0.25; a 0.25 s collar is ~3–8 DER points more forgiving.
2. **Overlap**: pyannote 3.1, diart, NeMo, and our scorer all score overlapped
   speech (`skip_overlap=False`). Older literature that excludes overlap reports
   deflated DER.
3. **VAD / speaker-count oracle**: pyannote 3.1 and polyvoice use automatic VAD
   and automatic speaker counting (no oracle). Some published numbers use oracle
   VAD or oracle speaker count, which is far easier.
4. **Dataset/split drift**: VoxConverse v0.0.2 vs v0.3; AMI Mix-Headset (easier)
   vs SDM/array; DIHARD "Full" vs ≤4-speaker subsets; CALLHOME split by speaker
   count. Match all of these before comparing.
5. **Scoring tool**: most use `pyannote.metrics` (Hungarian mapping); wespeaker's
   recipe uses NIST `md-eval.pl`. Our scorer follows the `md-eval` frame model
   with Hungarian mapping and is cross-checked against `polyvoice-bench`.
6. **Cross-repo numbers are unreliable**: e.g. DiariZen on VoxConverse was
   reported as 28.39 % by one harness and ~5.2 % by another — a version/config
   mismatch. Reproduce before trusting any third-party number.

## Sources

- ² pyannote 3.1 (VoxConverse, AMI, DIHARD; protocol; MIT; gated): https://huggingface.co/pyannote/speaker-diarization-3.1 — footprint: [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 5.91 MB + [wespeaker-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) 26.6 MB
- ³ diart (Coria et al., ASRU 2021; collar 0, overlap scored): https://arxiv.org/abs/2109.06483
- ⁴ 3D-Speaker toolkit (VoxConverse 11.75, AMI_SDM 21.76; collar unstated): https://github.com/modelscope/3D-Speaker
- ⁵ polyvoice: [`tests/der_baseline.json`](../tests/der_baseline.json) (schema `polyvoice-der-baseline-v2`), shipped FP32 (Silero VAD + WeSpeaker ResNet34 + AHC, threshold 0.45, singleton prune), reproduced by `benchmarks/der.py`
- ⁶ RTF artifact: [`benchmarks/results/voxconverse-test-10files-20260516.json`](../benchmarks/results/voxconverse-test-10files-20260516.json)
- ⁷ pyannote official benchmark (updated 2025-09; collar 0, overlap scored; community-1 weights CC-BY-4.0 but still HF-gated): https://www.pyannote.ai/benchmark + https://huggingface.co/pyannote/speaker-diarization-community-1 — on VoxConverse community-1 ties 3.1 (11.2 vs the 11.3 model-card figure; annotation-version drift), so the README headline comparison vs 3.1 stands
- ⁸ speakrs (pure Rust + ONNX, Apache-2.0 code; VoxConverse-test 11.1 % at collar 0, CoreML): https://github.com/avencera/speakrs (retrieved 2026-07-13)
- NeMo Sortformer (CC-BY-NC; DIHARD 16.28, CALLHOME): https://huggingface.co/nvidia/diar_sortformer_4spk-v1
- WhisperX (bundles pyannote 3.1): https://github.com/m-bain/whisperX
- sherpa-onnx (no DER published): https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html
- "Benchmarking Diarization Models" (DiariZen ~5.2 VoxConverse @0.25 collar): https://arxiv.org/abs/2509.26177

## Streaming latency presets (methodology)

Named presets for `polyvoice::streaming::LatencyPreset` / CLI `--latency-preset`:

| Preset     | window_secs | hop_secs | right_context_secs | cache_cap | input-buffer latency¹ | RTF | DER (collar 0) |
|------------|-------------|----------|--------------------|-----------|----------------------|-----|----------------|
| `realtime` | 1.0         | 0.5      | 0.0                | 16        | ≈ 1.03 s             | TBD | TBD            |
| `balanced` | 1.5         | 0.75     | 0.0                | 32        | ≈ 1.53 s             | TBD | TBD            |
| `accurate` | 2.0         | 1.0      | 0.25               | 64        | ≈ 2.28 s             | TBD | TBD            |

¹ **Input-buffer latency** is a configuration number:
`window_secs + right_context_secs + vad_frame_secs` with EnergyVad frame 512
samples @ 16 kHz (`vad_frame_secs ≈ 0.032 s`). It is **not** wall-clock RTF.

**Reporting convention (diart / NeMo):** publish **latency**, **RTF**, and **DER**
as three separate numbers. Never publish a latency figure without DER and a
stated methodology (hardware, chunk schedule, collar, overlap policy).

### How to fill the table

1. Stream a fixed subset (recommended: VoxConverse-test 30-file subset, or full
   test when budget allows) through `StreamingPipeline::with_latency_preset`
   with the Balanced ONNX embedder on a **named** CPU (cores, arch, OS).
2. Record per-chunk wall time series (`t_end - t_start` around `feed`), then
   RTF = total_feed_wall / audio_duration. Also record start-of-stream vs
   end-of-stream mean chunk latency on a ≥1 h synthetic or long-form file to
   confirm bounded state (cache cap prevents O(t) growth).
3. Score DER with `benchmarks/der.py` / `polyvoice-bench` at **collar 0**,
   overlap scored, Hungarian mapping — same protocol as the offline tables.
4. Optionally report label flip rate via
   `polyvoice::streaming::label_flip_rate` (first-emitted vs final labels).

Artifacts belong under `benchmarks/results/` (per-chunk latency series + DER
JSON). Until a measured run lands, cells stay **TBD** — do not invent numbers.

See also `benchmarks/results/streaming-latency-methodology.md`.
