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

| | polyvoice (v2+VBx, default) | pyannote 3.1 | WhisperX | NeMo Sortformer |
|--|--|--|--|--|
| **VoxConverse-test DER** | **15.4 %** ¹ | **11.3 %** ¹ | 11.3 % (= pyannote) | not published |
| **Model size** | **~30 MB** (+ PLDA for VBx) | ~32.5 MB | ~32.5 MB + Whisper | 123 M params |
| **Runtime** | **CPU / CoreML, ~33× realtime** | CPU/GPU (PyTorch) | GPU recommended | GPU |
| **Weights** | **MIT, ungated** | MIT code, **gated** (HF token) | gated (pyannote) | **CC-BY-NC** (non-commercial) |
| **Dependencies** | **pure Rust, no Python** | PyTorch | PyTorch + Whisper | PyTorch / NeMo |
| **Bindings** | **Rust / Python / C / CLI** | Python | Python | Python |
| **Streaming** | **Yes** | No | No | No |

¹ VoxConverse-test, **no forgiveness collar (collar 0), overlap scored** — the
strict protocol pyannote 3.1 reports against, so these two are collar-matched.
polyvoice trails the accuracy leader by ~4 DER points and trades that for
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
| **polyvoice (v2+VBx, default)** | **15.37** | 2.29 | 1.86 | 6.97 | 0 | this repo ⁵ |
| diart (online, 5 s latency) | 16.8 | 4.9 | 3.8 | 8.2 | 0 | diart paper ³ |
| polyvoice (legacy, `--legacy`) | 18.54 | 4.49 | 3.19 | 4.99 | 0 | this repo ⁵ |
| diart (online, 1 s latency) | 20.1 | 3.3 | 5.1 | 11.7 | 0 | diart paper ³ |

For reference, polyvoice (v2+VBx) at the **0.25 s collar** is **11.12 %** micro
(macro 11.50; miss 2.29 / FA 1.86 / conf 6.97). Legacy at collar 0.25 is
**12.91 %** micro. No collar-0.25 pyannote number is published, so do not
compare that figure across systems.

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
| **polyvoice (v2+VBx, default)** | **25.17** | 7.69 | 1.70 | 7.47 | Mix-Headset | this repo ⁵ |
| diart (online, 5 s) | 27.5 | 10.0 | 5.0 | 12.4 | headset | diart paper ³ |
| polyvoice (legacy, `--legacy`) | 32.87 | 17.09 | 2.44 | 5.21 | Mix-Headset | this repo ⁵ |

polyvoice (v2+VBx) at the 0.25 s collar: **17.66 %** micro (macro 16.86).
Legacy at collar 0.25: **25.20 %** micro (macro 24.75).

## polyvoice speaker-count & error decomposition

A low DER can hide bad speaker counting; we report it explicitly.

| Split | DER (collar 0) | miss | FA | conf | spk exact | spk ±1 | spk off-by-2+ |
|---|---|---|---|---|---|---|---|
| VoxConverse-test (232, v2+VBx) | 15.37 % | 2.29 | 1.86 | 6.97 | 87 | 65 | 80 |
| AMI-test (16, v2+VBx) | 25.17 % | 7.69 | 1.70 | 7.47 | 3 | 3 | 10 |
| VoxConverse-test (232, legacy) | 18.54 % | 4.49 | 3.19 | 4.99 | 57 | 46 | 129 |
| AMI-test (16, legacy) | 32.87 % | 17.09 | 2.44 | 5.21 | 1 | 0 | 15 |

The dominant residual error is still **speaker mis-counting** (confusion + the
off-by-2+ tail), especially on long meetings — better than legacy but still the
honest weak spot versus pyannote's end-to-end segmentation.

## Default pipeline: v2 + VBx (since 0.11)

As of **0.11**, the CLI default is **pipeline v2 + VBx** after a hard full-split
gate (2026-07-25): no-collar micro DER ≤ legacy on **both** VoxConverse-test
(232) and AMI-test (16). Artifacts:
[`benchmarks/results/full-der-2026-07-25/`](../benchmarks/results/full-der-2026-07-25/)
(`VERDICT.md`).

| Path | CLI | Vox no-collar micro | AMI no-collar micro |
|---|---|---:|---:|
| **v2 + VBx (default)** | `polyvoice file.wav` (+ PLDA) | **15.37** | **25.17** |
| legacy | `polyvoice --legacy file.wav` | 18.54 | 32.87 |
| v2 + AHC | `polyvoice --clusterer ahc file.wav` | (subset; see archive) | (subset) |

VBx PLDA weights auto-download via the model registry; optional override
`--vbx-plda-dir <dir>` / `POLYVOICE_VBX_PLDA_DIR` (see
[`docs/vbx-plda-release.md`](vbx-plda-release.md)). Use `--clusterer ahc` or
`--legacy` for non-VBx paths.

Earlier **subset** bootstrap numbers (60-file Vox / full AMI, pre-gate) are
retained below for history only — prefer the full-split rows above.

**No-collar macro DER % (95 % CI) — historical subsets:**

| Pipeline | VoxConverse-test | VoxConverse-dev | AMI |
|---|---|---|---|
| legacy (full split) | 18.54 | — | 32.87 |
| v2 + AHC (subset) | 20.4 [16.4–24.9] | 13.2 [10.5–16.1] | 35.8 [29.8–41.9] |
| v2 + VBx (subset) | 17.0 [13.5–20.8] | 13.6 [10.6–16.9] | 37.0 [31.0–43.1] |
| **v2 + VBx (full split, default)** | **15.83 macro / 15.37 micro** | — | **24.12 macro / 25.17 micro** |

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
- ⁵ polyvoice: [`tests/der_baseline.json`](../tests/der_baseline.json) (schema `polyvoice-der-baseline-v2`) and full-split gate [`benchmarks/results/full-der-2026-07-25/`](../benchmarks/results/full-der-2026-07-25/). Default (0.11+): pipeline v2 + VBx + WeSpeaker ResNet34; legacy numbers via `--legacy`
- ⁶ RTF artifact: [`benchmarks/results/voxconverse-test-10files-20260516.json`](../benchmarks/results/voxconverse-test-10files-20260516.json)
- ⁷ pyannote official benchmark (updated 2025-09; collar 0, overlap scored; community-1 weights CC-BY-4.0 but still HF-gated): https://www.pyannote.ai/benchmark + https://huggingface.co/pyannote/speaker-diarization-community-1 — on VoxConverse community-1 ties 3.1 (11.2 vs the 11.3 model-card figure; annotation-version drift), so the README headline comparison vs 3.1 stands
- ⁸ speakrs (pure Rust + ONNX, Apache-2.0 code; VoxConverse-test 11.1 % at collar 0, CoreML): https://github.com/avencera/speakrs (retrieved 2026-07-13)
- NeMo Sortformer (CC-BY-NC; DIHARD 16.28, CALLHOME): https://huggingface.co/nvidia/diar_sortformer_4spk-v1
- WhisperX (bundles pyannote 3.1): https://github.com/m-bain/whisperX
- sherpa-onnx (no DER published): https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html
- "Benchmarking Diarization Models" (DiariZen ~5.2 VoxConverse @0.25 collar): https://arxiv.org/abs/2509.26177

## Streaming latency presets (measured)

Named presets for `polyvoice::streaming::LatencyPreset` / CLI `--latency-preset`.

**Protocol (2026-07-24):** VoxConverse-test **first 10 files** (sorted names),
overlap scored, Hungarian DER, collar 0 and 0.25. `StreamingPipeline` + Silero
VAD + WeSpeaker ResNet34, feed chunks of 3200 samples (~200 ms) @ 16 kHz.
Release build on **Apple M1 Pro (10 cores)**. Full artifact:
[`benchmarks/results/streaming-latency-measured.json`](../benchmarks/results/streaming-latency-measured.json).

| Preset     | window | hop  | right ctx | cache | input-buffer latency¹ | RTF   | DER collar 0 | DER collar 0.25 |
|------------|--------|------|-----------|-------|----------------------|-------|--------------|-----------------|
| `realtime` | 1.0 s  | 0.5  | 0.0       | 16    | **1.032 s**          | **0.117** | **42.15%** | **32.74%** |
| `balanced` | 1.5 s  | 0.75 | 0.0       | 32    | **1.532 s**          | **0.109** | **29.99%** | **20.15%** |
| `accurate` | 2.0 s  | 1.0  | 0.25      | 64    | **2.282 s**          | **0.111** | **30.10%** | **20.85%** |

¹ **Input-buffer latency** is configuration:
`window_secs + right_context_secs + vad_frame_secs` with frame 512 @ 16 kHz
(`≈ 0.032 s`). It is **not** wall-clock RTF. RTF = total wall / total audio
(~6701 s audio × 3 presets on this run).

**Notes:** online DER is expected to trail the offline legacy baseline
(~18.5% no-collar full test). On this 10-file slice `balanced` beats
`realtime` by ~12 pp (collar 0); `accurate` does **not** improve further
(right-context currently contributes to the latency budget but does not yet
delay emission — see streaming module docs). Reproduce:

```bash
cargo run --release --features "cli,onnx,download" --bin polyvoice-measure -- streaming \
  --dataset data/voxconverse-test --max-files 10 \
  --output benchmarks/results/streaming-latency-measured.json
```

See also `benchmarks/results/streaming-latency-methodology.md`.

## VAD parity: Silero vs earshot (measured)

**Protocol:** same VoxConverse-test **10-file** slice, legacy offline pipeline
(WeSpeaker ResNet34 + AHC threshold 0.45), only the VAD backend swapped.
Release build, Apple M1 Pro. Artifact:
[`benchmarks/results/vad-parity-earshot-silero.json`](../benchmarks/results/vad-parity-earshot-silero.json).

| VAD | frame | DER collar 0 | DER collar 0.25 | RTF |
|-----|-------|--------------|-----------------|-----|
| **Silero** (default) | 512 | **23.89%** | **15.82%** | 0.102 |
| **earshot** (`vad-earshot`) | 256 | **26.54%** | **18.68%** | 0.099 |
| **Δ (earshot − Silero)** | — | **+2.65 pp** | **+2.86 pp** | −0.0025 |

**Parity gate** (notes file): |Δ DER| ≤ **0.3 pp** absolute. **Failed** on both
collars → keep earshot **optional only**; do **not** switch the default VAD.
Vendor “40× faster / more accurate” claims remain **unverified / not supported**
by this DER gate (RTF nearly tied; accuracy worse).

```bash
cargo run --release --features "cli,vad-earshot,onnx,download" --bin polyvoice-measure -- vad-parity \
  --dataset data/voxconverse-test --max-files 10 \
  --output benchmarks/results/vad-parity-earshot-silero.json
```

## Embedder short-segment: ResNet34 vs ERes2NetV2 (measured)

**EER protocol:** 400 same/different-speaker pairs built from **VoxConverse-test
RTTM segments** (VoxCeleb1 audio not present in this tree). Center-crop to
0.5 / 1 / 2 / 3 s, cosine scoring, equal-error rate. **Not** the official
VoxCeleb1 `veri_test` protocol — domain is in-the-wild multi-party English.

**DER protocol:** same 10-file Vox slice, legacy pipeline, Silero VAD fixed;
only the embedder ONNX swapped (ResNet34 256-d vs ERes2NetV2 192-d zh-cn
optional download). Artifact:
[`benchmarks/results/embedder-short-eres2net.json`](../benchmarks/results/embedder-short-eres2net.json).

| Duration | ResNet34 EER % | ERes2NetV2 EER % |
|----------|----------------|------------------|
| 0.5 s | **18.86** | 27.46 |
| 1.0 s | **7.21** | 20.09 |
| 2.0 s | **4.75** | 13.03 |
| 3.0 s | **3.84** | 10.74 |

| Embedder | DER collar 0 | DER collar 0.25 |
|----------|--------------|-----------------|
| WeSpeaker ResNet34 (default) | **23.89%** | **15.82%** |
| ERes2NetV2 (zh-cn common ONNX) | **53.84%** | **49.18%** |

**Verdict:** the shipped **zh-cn** ERes2NetV2 optional weights are **not** a
drop-in upgrade on English VoxConverse under our fbank front-end: both short-seg
EER and full-file DER regress hard. Keep the adapter for CJK / experiment
paths; do **not** make it default. A VoxCeleb-English ERes2Net export (if
ungated Apache) would need a separate measurement before any accuracy claim.

```bash
cargo run --release --features "cli,embedder,onnx,download" --bin polyvoice-measure -- embedder-short \
  --veri-list data/voxceleb1-subset/lists/veri_test.txt \
  --wav-root data/voxceleb1-subset \
  --der-dataset data/voxconverse-test --der-max-files 10 \
  --output benchmarks/results/embedder-short-eres2net.json
```
