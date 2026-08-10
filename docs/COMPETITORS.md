# Open-source speaker diarization competitors

> Research snapshot: 2026-06-23. Stars are approximate and change daily.

Polyvoice positions itself as a **Rust-native, CPU-first, MIT-licensed, ungated**
speaker diarization engine (ort-free core; production path uses ONNX Runtime)
with first-class streaming and a small (~8.4 MB INT8) footprint. The list below
compares the main open-source alternatives and what polyvoice can learn from them.

## Polyvoice baseline

| Property | Value |
|---|---|
| Repository | https://github.com/ekhodzitsky/polyvoice |
| Language | Rust (pure-Rust core, ONNX via `ort`) |
| License | MIT (code and shipped models) |
| Deployment | CPU-first, ~8.4 MB INT8, no Python runtime |
| Bindings | Rust library, Python (maturin), C FFI, CLI, MCP server |
| Streaming | First-class `streaming::StreamingPipeline` |
| DER benchmark | 15.22% on VoxConverse-test (collar 0, overlap-scored; v2+VBx, H2H 2026-08 vs speakrs 11.08%) |

## Competitor comparison

| Competitor | Stars | Stack | Code license | Model license | Main advantage over polyvoice | Where polyvoice wins |
|---|---|---|---|---|---|---|
| **speakrs** | ~★ growing | Rust / ONNX+CoreML | Apache-2.0 | ungated HF models | **pyannote-level DER (11.08% Vox test, our scorer)**; CoreML speed | streaming surfaces, multi-bindings, MIT end-to-end, measured multi-corpus story |
| **pyannote.audio** | 10 160 | Python / PyTorch | MIT | Gated (HuggingFace token + terms) | SOTA accuracy (DER ~11.2% on VoxConverse), mature pipelines, fine-tuning | CPU-only, no Python, no HF token, streaming, ~8.4 MB INT8 |
| **WhisperX** | 22 620 | Python / faster-whisper / PyTorch | BSD-2-Clause | pyannote gated + Whisper MIT | ASR + diarization + word-level timestamps in one CLI, 99 languages | Pure diarization, Rust-native, CPU, MIT without gated models |
| **NVIDIA NeMo** | 17 459 | Python / PyTorch / CUDA | Apache-2.0 | NGC / Riva terms (not pure OSS) | GPU SOTA (MSDD, Sortformer), enterprise ecosystem, ASR/TTS/LLM | Rust-native, CPU, MIT, simple deploy |
| **FunASR** | ~18 500 | Python / PyTorch (+ C++ runtime) | MIT | Custom FunASR Model License | Industrial ASR+VAD+diarization, 50+ languages, Docker/API | Size, license simplicity, Rust integration |
| **Kaldi** | 15 418 | C++ / Shell / Python | Apache-2.0 | Apache-2.0, some models trained on LDC data | Research standard, x-vector + PLDA recipes, ASR+speaker | Ready pipeline, streaming, no C++ build |
| **sherpa-onnx** | 13 132 | C++ / ONNX | Apache-2.0 | pyannote gated / 3D-Speaker Apache / Silero MIT | Offline SDK, 12 language bindings, ASR+TTS+VAD+diarization, edge platforms | Rust core + streaming diarization, fewer app deps |
| **SpeechBrain** | 11 644 | Python / PyTorch | Apache-2.0 | Apache-2.0 | Universal speech toolkit, 200+ recipes, embedding training | CPU-only deploy, Rust, size, streaming |
| **diart** | 1 988 | Python / PyTorch / RxPY | MIT | pyannote gated | Real-time streaming diarization, DER 16.8% on VoxConverse | CPU-first, no Python/HF token, Rust-native |
| **Resemblyzer** | 3 274 | Python / PyTorch | Apache-2.0 | Apache-2.0 / MIT | Popular embeddings, multi-task use cases | Ready diarization pipeline, streaming, Rust |
| **VBx** | 287 | Python / PyTorch / Kaldi | Apache-2.0 | Possible CN-Celeb restrictions | Classic SOTA baseline (CALLHOME 4.42%), flexibility | Easier to use, streaming, Rust, no Kaldi |
| **simple-diarizer** | 157 | Python / PyTorch | **GPL-3.0** | MIT / Apache | Minimal Python pipeline | GPL copyleft, Rust, CPU-only, streaming |

## What polyvoice can borrow

1. **Accuracy improvements from pyannote / NeMo / EEND**
   - Overlap-aware segmentation quality and automatic speaker-count estimation.
   - End-to-end neural diarization back-ends (Sortformer, EEND) as optional ONNX profiles.
   - Better embedding extractors (WavLM, ECAPA-TDNN v3, CAM++ improvements).
   - These are already on the roadmap, but the gap to pyannote 11.2% is the biggest risk.

2. **Integrated "who said what" product shape from WhisperX / FunASR**
   - Word-level timestamps + speaker labels are the user-facing end state.
   - Polyvoice already has attribution (`attribution` module) and `polyvoice-transcribe`; double down on this instead of competing as a raw diarization library alone.

3. **Packaging and platform reach from sherpa-onnx**
   - Pre-built binaries, Docker images, mobile examples, and broader language bindings lower friction.
   - Sherpa-onnx proves there is demand for local, offline, multi-platform speech stacks.

4. **Streaming latency/accuracy trade-offs from diart**
   - Diart shows real-time diarization is possible with tunable latency.
   - Polyvoice can publish latency vs. DER curves and recommend profiles for real-time use.

5. **Training recipes and data augmentation from SpeechBrain**
   - Fine-tuning recipes for domain adaptation (meetings, call centers, podcasts).
   - Augmentation (reverb, noise, speed) could improve robustness without growing runtime.

6. **CJK/industrial coverage from FunASR**
   - Chinese, Japanese, Korean ASR + diarization demand is large.
   - Consider a SenseVoice / Paraformer companion ASR backend, not just Parakeet.

7. **Simplicity from simple-diarizer**
   - A one-liner Python API (`pipe = Diarizer(); pipe("file.wav")`) helps adoption.
   - Polyvoice Python bindings can emulate this without sacrificing the Rust core.

## Strategic assessment

**Should polyvoice continue? Yes.**

Polyvoice owns a defensible niche that the Python/C++ incumbents do not serve
well: a **Rust-native, CPU-first, MIT-licensed, ungated, streaming diarization
engine** that embeds into apps without a PyTorch stack (production ONNX path
still uses the `ort` native runtime). Competitors are stronger on accuracy,
ecosystem, and bindings, but they are heavier, GPU-oriented, and often gated
or commercially restricted.

The main **peer threat** is **speakrs** (same Rust/ONNX niche): measured **11.08%** vs our **15.22%** on VoxConverse-test under one scorer (gap ~4 pp, confusion-dominated). VBx + PLDA already cut an earlier ~7 pp gap to pyannote (~11.2%); if the residual stalls, accuracy-sensitive users will pick speakrs or the heavier Python stacks. The roadmap already targets the rest with better segmentation, embeddings, and EEND/Sortformer spikes. The priority should be:

1. **Close the DER gap** with optional ONNX accuracy profiles (EEND/Sortformer, WavLM/CAM++ embeddings) while keeping the CPU-first default.
2. **Own the deployability story** — pre-built binaries, Docker, WASM, mobile, and clear "no Python, no GPU, no token" messaging.
3. **Productize who-said-what** — diarization + ASR attribution as the primary user-facing surface, not just raw speaker turns.
4. **Stay license-clean** — do not introduce gated or copyleft models; use permissive weights or ship conversion/provenance docs.

If the project cannot keep the accuracy gap within ~3–5 DER points of pyannote on standard benchmarks over the next major releases, it risks remaining a curiosity rather than a production option. With the current trajectory, it is worth continuing.
