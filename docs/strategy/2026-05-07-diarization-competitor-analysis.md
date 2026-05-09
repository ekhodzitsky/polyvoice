# Diarization Competitor Analysis

Date: 2026-05-07

## Executive Summary

polyvoice is already a real Rust diarization library: Silero VAD, WeSpeaker/ECAPA ONNX embeddings, AHC/k-means/spectral clustering, RTTM, DER, CLI, Python bindings, C FFI, online/offline APIs, and safety-oriented Rust checks. The current public positioning is "Rust diarization without Python" with reported VoxConverse DER around 16.4% and AMI DER around 24.5%.

The market has moved. The competitor to beat is no longer just `pyannote-rs` or a basic WeSpeaker + clustering pipeline. The hardest direct Rust competitor is `speakrs`, which claims a full pyannote `community-1` style pipeline in Rust: segmentation, powerset decode, overlap-add aggregation, binarization, embeddings, PLDA, and VBx. Python-side, `diarize` claims strong CPU-only VoxConverse numbers with a simple install. Platform-side, `sherpa-onnx` owns embedded/mobile/runtime breadth. Research-side, pyannote, NeMo/Sortformer, SpeechBrain, WeSpeaker, and diart define the algorithmic bar.

To beat them, polyvoice needs to move from "VAD + sliding embeddings + clustering" to an overlap-aware neural diarization engine with a public, reproducible benchmark harness. The roadmap should optimize in this order:

1. Truth first: reproducible DER/JER/speaker-count/overlap benchmarks across AMI, VoxConverse, DIHARD/CALLHOME-like stress sets, plus competitor runners.
2. Accuracy core: pyannote-style segmentation/powerset decoding, overlap-add aggregation, calibrated binarization, PLDA scoring, VBx/HMM resegmentation, and robust speaker-count estimation.
3. Rust advantage: CPU-fast default, optional CoreML/CUDA/OpenVINO/WebGPU execution providers, single-binary CLI, Python/C/FFI/WASM surfaces, no mandatory Python runtime.
4. Product edge: reliable word-level speaker assignment, streaming diarization, model cache, model license metadata, and predictable deployment.

## Current polyvoice Baseline

Current codebase capabilities:

- End-to-end `Pipeline` in `src/pipeline.rs`: VAD -> windowed embeddings -> AHC/k-means/spectral/auto clustering -> merged turns.
- Config surface in `src/types.rs`: threshold, speaker cap, window/hop sizes, minimum speech and turn durations, minimum embeddings per speaker, sample rate, clustering backend.
- Metrics in `src/der.rs`: frame-based DER with collar, miss, false alarm, confusion, and speaker mapping.
- RTTM parser/writer in `src/rttm.rs`.
- Overlap utility in `src/overlap.rs`, but it detects overlap only after labeled intervals already exist; it does not infer simultaneous speakers from audio.
- CLI in `src/bin/polyvoice.rs` with text/json/rttm output and model download.
- Benchmark runner in `src/bin/polyvoice-bench.rs`.

Important gaps:

- No neural frame-level speaker segmentation model.
- No powerset decoding, overlap-add aggregation, or calibrated binarization.
- No PLDA scoring backend yet.
- No VBx/HMM resegmentation.
- Overlap is not modeled as a first-class multi-speaker output.
- Speaker counting is heuristic and not yet benchmarked as a first-class metric.
- `benches/der_ami.rs` still has an older simplified DER implementation instead of using `src/der.rs`.
- Public benchmark matrix is much narrower than competitors' claims.

## Main Competitors

| Competitor | Type | What they do well | Weakness or opening for polyvoice | Beat strategy |
|---|---|---|---|---|
| pyannote.audio | Python/PyTorch reference stack | Best-known OSS/commercial diarization ecosystem; neural segmentation, embedding, overlap-aware pipeline, pretrained model hub, community and precision tiers. GitHub search showed about 9.9k stars. | Heavy Python/PyTorch deploy surface, model access/licensing friction, less attractive for embedded Rust/FFI deployments. | Match accuracy on public benchmarks while being easier to ship: no Python runtime, stable Rust API, CLI, FFI, Python wheel backed by Rust. |
| speakrs | Direct Rust crate | Claims full pyannote `community-1` style pipeline in Rust: segmentation, powerset decode, overlap-add, binarization, embedding, PLDA, VBx. Claims VoxConverse dev 7.1% DER at 529x realtime on CoreML and VoxConverse test 11.1% DER. | Very new, small GitHub presence, narrower product surface than polyvoice, likely model/license/runtime complexity. | Do not compete with AHC only. Implement comparable neural segmentation + PLDA/VBx, then beat with public reproducibility, CPU-first path, stronger docs, Python/FFI/CLI polish, and benchmark honesty. |
| diarize | Python package | Very easy `pip install diarize`; CPU-only; Silero + WeSpeaker + GMM/BIC + spectral; claims about 4.8% weighted DER on VoxConverse dev and 14.96% on AMI preliminary. Apache 2.0. | Python runtime, no overlap modeling, CPU-only, many-speaker speaker-count weakness reported by its own README. | Beat through Rust deployability, overlap-aware multi-speaker output, many-speaker robustness, and comparable CPU ease via CLI/Python wheels. |
| sherpa-onnx | C++/ONNX platform with Rust wrapper | Huge cross-platform local speech stack: ASR, TTS, VAD, speaker diarization, speaker ID, verification, WebAssembly, mobile, many APIs. GitHub search showed about 12k stars. | Broad platform, not a focused best-in-class diarization library; API and model choices can feel large and toolkit-like. | Focus polyvoice as the ergonomic, high-accuracy diarization specialist while borrowing the cross-platform mindset. |
| NVIDIA NeMo / Sortformer | Research and model stack | Sortformer-style streaming diarization and multitalker ASR; strong GPU ecosystem; Parakeet and Nemotron speech models are active. | Python/PyTorch/NVIDIA-centric; deployment and licensing/model distribution complexity. | Add Sortformer-compatible ONNX backend as an optional model family, but keep a vendor-neutral Rust abstraction. |
| parakeet-rs | Rust ASR + diarization crate | ONNX Parakeet ASR, multitalker ASR, Sortformer v2/v2.1 streaming diarization up to 4 speakers, many ORT execution-provider features. | Diarization is one part of a larger ASR crate; limited speaker counts in Sortformer path; model setup complexity. | Integrate with it or offer adapters. Beat on general diarization quality, benchmark breadth, and speaker-count flexibility. |
| pyannote-rs / native-pyannote-rs | Rust pyannote-style crates | Direct Rust attempts around pyannote models, simple comparison point. | Smaller surface; `speakrs` README reports weak pyannote-rs results on VoxConverse dev subset. | Treat as baseline competitor, not the north-star threat. |
| diart | Python streaming framework | Real-time streaming diarization, overlap-aware low-latency design, tuning/benchmarking, WebSocket serving. | Python stack, pyannote model access, less focused on Rust embedding. | Build `OnlineDiarizerV2` around rolling local segmentation, overlap-aware embeddings, cannot-link constraints, and latency modes. |
| SpeechBrain | Python research toolkit | Strong AMI recipe with ECAPA-TDNN + spectral clustering; docs report low DER under oracle/estimated setups with overlap ignored. | Research recipe, Python stack, not a lightweight deployable Rust library. | Use as algorithmic reference and benchmark control, not direct product competitor. |
| WeSpeaker | Python toolkit/model source | Production/research speaker embeddings, PLDA, AS-Norm, score calibration, VoxConverse recipe, many embedding backbones. | More speaker verification/embedding toolkit than turnkey Rust diarization. | Use model families and scoring ideas; add Rust adapters/export validation and model cards. |
| WhisperX / meeting apps | Downstream apps | Popular user-facing ASR + diarization flows; WhisperX uses pyannote diarization and word speaker assignment. | Diarization not their core engine; overlap not solved; often needs HF token/GPU/Python. | Make polyvoice the diarization engine that Whisper-like apps want to embed. |

## Rust Crate Landscape

Data was gathered with `cargo search` / `cargo info` on 2026-05-07.

| Crate | Version | License | Positioning | Relevance |
|---|---:|---|---|---|
| `polyvoice` | 0.5.2 | MIT | Current repo: Rust diarization, online/offline, ONNX-powered. | Baseline. |
| `speakrs` | 0.4.0 | Apache-2.0 | Fast Rust speaker diarization with pyannote-level accuracy, CoreML/CUDA acceleration. | Highest direct threat. |
| `pyannote-rs` | 0.3.4 | MIT | Speaker diarization using pyannote in Rust; CoreML/DirectML features. | Direct but likely less complete than `speakrs`. |
| `native-pyannote-rs` | 0.1.4 | MIT | Speaker diarization using pyannote in Rust. | Direct, early-stage. |
| `parakeet-rs` | 0.3.5 | MIT OR Apache-2.0 | Fast ASR and speaker diarization with NVIDIA Parakeet/Sortformer via ONNX. | Streaming and ASR-adjacent threat/opportunity. |
| `sherpa-onnx` | 1.13.0 | Apache-2.0 | Safe Rust wrapper for sherpa-onnx speech toolkit. | Platform competitor and integration target. |
| `whisperforge-diarize` | 0.2.0 | MIT | Embedding clustering for WhisperForge. | Narrow ASR-adjacent competitor. |
| `silero-vad-rust` | 6.2.1 | See crate | Bundled Silero VAD in Rust. | Component competitor; not full diarization. |
| `fast-vad` | 0.2.1 | See crate | Fast Rust VAD with Python bindings and streaming. | Component competitor; useful benchmark. |

## What "Beat Them" Means

polyvoice should not define victory as "one good DER number on one dataset." Competitors can cherry-pick. Victory means:

1. Accuracy: better or equal to the best open systems on at least three benchmark families: broadcast/interview, meeting-room, and hard overlap/noise.
2. Reproducibility: every README claim has a command, model hash, dataset manifest, and result JSON.
3. Deployment: lowest-friction Rust-native deployment among accurate diarizers.
4. Latency: offline batch, low-latency streaming, and high-throughput queue modes are all first-class.
5. Output quality: stable speaker IDs, overlap-aware multi-speaker segments, word-level assignment, confidence, and speaker-count diagnostics.
6. Trust: no hidden Python runtime path, no untracked model licenses, no unverifiable benchmark tables.

## Source Notes

- pyannote.audio: https://github.com/pyannote/pyannote-audio
- pyannote community model used by downstream tools: https://huggingface.co/pyannote/speaker-diarization-community-1
- speakrs README and benchmarks: https://github.com/avencera/speakrs
- diarize README and benchmarks: https://github.com/FoxNoseTech/diarize
- sherpa-onnx README: https://github.com/k2-fsa/sherpa-onnx
- parakeet-rs README: https://github.com/altunenes/parakeet-rs
- diart README and paper summary: https://github.com/juanmc2005/diart
- SpeechBrain AMI diarization recipe: https://github.com/speechbrain/speechbrain/tree/develop/recipes/AMI/Diarization
- WeSpeaker README: https://github.com/wenet-e2e/wespeaker
- WhisperX README: https://github.com/m-bain/whisperX
