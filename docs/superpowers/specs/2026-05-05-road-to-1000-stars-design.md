# Road to 1000 GitHub Stars — Design Spec

## Context

polyvoice is a Rust speaker diarization library (v0.4.3, ~2650 LOC). It has online/offline modes, ONNX-based ECAPA-TDNN extractor, C FFI, lock-free session pool, and a strong CI pipeline (Miri, Loom, fuzz, cross-platform). MIT licensed.

**Current gap:** no working end-to-end pipeline on real audio. No downloadable models, no WAV I/O, no real DER benchmarks. The library cannot be evaluated by anyone without significant manual setup.

**Goal:** 1000 GitHub stars within ~6 months.

**Strategy:** "Staged rocket" — silently build technical credibility (Phase 1), package for consumption (Phase 2), loud launch (Phase 3), sustain momentum (Phase 4).

**Target audience:** universal — Rust developers, Python/Go/Node developers via FFI/PyPI, ML/audio engineers.

**Quality bar:** DER on AMI test set within 3% of pyannote 3.x.

---

## Phase 1: Silent Quality Push (4-6 weeks)

### Goal

DER on AMI test set ≤ pyannote 3.x + 3%. Fully working end-to-end pipeline on real audio.

### Models

| Component | Model | Source | Format | License |
|-----------|-------|--------|--------|---------|
| Speaker embeddings | WeSpeaker ResNet34 | github.com/wenet-e2e/wespeaker | ONNX (pre-exported) | Apache 2.0 |
| VAD | Silero VAD v5 | github.com/snakers4/silero-vad | ONNX (~2MB) | MIT |

**Why WeSpeaker over SpeechBrain ECAPA-TDNN:**
- Pre-exported ONNX files with known embedding dimensions
- ResNet34-based models achieve better DER on VoxConverse/AMI than ECAPA-TDNN
- Apache 2.0 license

**Why Silero VAD:**
- De-facto standard (used in whisper.cpp, pyannote, dozens of projects)
- ONNX format, small model, fast inference
- Current `EnergyVad` is a placeholder, not production quality

### Algorithmic Work

1. **Silero VAD integration** — new `SileroVad` struct implementing `VoiceActivityDetector` trait. ONNX inference via existing `ort` dependency.

2. **Agglomerative Hierarchical Clustering (AHC)** — replaces online cosine-threshold clustering for offline mode:
   - Bottom-up: each segment starts as its own cluster
   - Merge by maximum cosine similarity until threshold
   - Standard algorithm used by pyannote and SpeechBrain

3. **WAV I/O** — PCM WAV parsing (16-bit, mono, 16kHz) via `hound` crate (~400 LOC, well-maintained, handles edge cases).

4. **Model download script** — `scripts/download-models.sh` fetches WeSpeaker + Silero VAD ONNX files to `models/`.

### Benchmarks

- Dataset: AMI test set ("headset mix") — standard diarization benchmark
- Metric: DER (miss + false alarm + confusion), collar = 0.25s
- Baseline comparison: pyannote 3.1 pipeline (~18-22% DER on AMI)
- Implementation: `benches/der_ami.rs` — automated DER computation

### New Files

```
src/
  silero_vad.rs      — Silero VAD ONNX integration
  ahc.rs             — Agglomerative hierarchical clustering
  wav.rs             — WAV parsing via hound
  wespeaker.rs       — WeSpeaker ONNX extractor (if differs from ECAPA)
scripts/
  download-models.sh — model download
benches/
  der_ami.rs         — DER benchmark on AMI
```

---

## Phase 2: Packaging (2 weeks)

### CLI

Separate binary crate `polyvoice-cli` (or binary target in current crate):

```bash
polyvoice diarize meeting.wav
polyvoice diarize meeting.wav --max-speakers 4 --output json
polyvoice download-models
```

**Output formats:**
- Human-readable (default): `SPEAKER_00  0.5s - 3.2s`
- JSON: array of `{speaker, start, end}`
- SRT/RTTM: standard audio/speech formats

**Distribution:**
- `cargo install polyvoice-cli`
- Pre-built binaries for Linux/macOS/Windows via GitHub Releases (CI with `cargo-dist` or `cross`)

### Python Package

**Stack:** PyO3 + maturin.

```python
import polyvoice

pipeline = polyvoice.Pipeline.from_pretrained()  # auto-downloads models
result = pipeline("meeting.wav")

for turn in result.turns:
    print(f"{turn.speaker}: {turn.start:.1f}s - {turn.end:.1f}s")

# Or from numpy array
import numpy as np
samples = np.load("audio.npy")  # float32, 16kHz, mono
result = pipeline(samples, sample_rate=16000)
```

**Key decisions:**
- API mirrors pyannote — lowers migration barrier
- `from_pretrained()` downloads models to `~/.cache/polyvoice/`
- Returns Python objects, not JSON — `result.turns`, `result.num_speakers`
- numpy array and WAV file path support
- Wheels on PyPI for Linux (manylinux), macOS (x86+arm), Windows

**Structure:**

```
python/
  pyproject.toml       — maturin config
  polyvoice/
    __init__.py        — Python API
    _native.pyd        — compiled Rust
  tests/
    test_pipeline.py   — pytest
```

### README Upgrade

1. **DER table** — real numbers next to pyannote:
   ```
   | System          | AMI (DER%) | Speed (RTF) |
   |-----------------|------------|-------------|
   | pyannote 3.1    | ~21%       | 0.8x        |
   | polyvoice 0.5   | ~23%       | 0.15x       |
   ```

2. **GIF/asciicast** — 15-second CLI demo in terminal

3. **"From pyannote?" section** — 5-line migration guide

4. **New badges:** PyPI downloads, crates.io downloads

---

## Phase 3: Launch (1 week)

### Blog Post

**Title:** "Speaker diarization in Rust: matching pyannote at 5x the speed"

**Structure:**
1. Problem — pyannote works but PyTorch + Python runtime = heavy deployment, GIL, slow inference
2. Solution — polyvoice: Rust + ONNX Runtime, same results, different trade-offs
3. Numbers — DER table, latency graphs (RTF), memory footprint
4. Architecture — pipeline design, lock-free pool, AHC
5. Try it — `pip install`, `cargo install`, 3 lines of code
6. Honesty — where pyannote is still better, roadmap

**Publish:** own blog (GitHub Pages / dev.to) + crosspost.

### Launch Platforms (priority order)

| Platform | Format | Potential |
|----------|--------|-----------|
| Hacker News | "Show HN: polyvoice — speaker diarization in Rust, matching pyannote at 5x speed" | 200-500 stars |
| r/rust | Focus on Rust architecture: lock-free pool, Miri, Loom | 50-150 stars |
| r/MachineLearning | Focus on DER numbers and pyannote comparison | 50-100 stars |
| Twitter/X | Thread with GIF demo, numbers, link | Amplification |
| Python Discord / r/Python | "pip install polyvoice — fast diarization without PyTorch" | Python audience |
| Lobsters | Technical focus | 20-50 stars |

### Timing

- Day 1 (Tuesday or Wednesday): HN Show HN at 9-10am PST (peak activity)
- Day 1-2: r/rust, Twitter/X
- Day 3: r/MachineLearning, r/Python
- Week 1: respond to every issue and comment quickly — first days are critical

### Pre-launch Checklist

- [ ] GitHub repo description and topics set (`rust`, `speaker-diarization`, `onnx`, `pyannote`, `audio`, `speech`)
- [ ] GitHub Releases with pre-built binaries
- [ ] `CONTRIBUTING.md`
- [ ] 5-10 GitHub Issues labeled `good first issue`
- [ ] Social preview image (1280x640)

---

## Phase 4: Post-launch Momentum (ongoing)

### Every Improvement = Content

| Improvement | Post/Announcement |
|-------------|-------------------|
| Spectral clustering | "polyvoice 0.6: spectral clustering drops DER by 3%" |
| PLDA scoring | "polyvoice 0.7: PLDA backend, now within 1% of pyannote" |
| Silero VAD v5 tuning | Tweet + r/rust |
| WASM demo | "Try speaker diarization in your browser" — viral potential |
| `no_std` embedded | "Speaker diarization on Raspberry Pi" — niche but beloved |
| Whisper-rs integration | "Full STT+diarization pipeline in Rust" — killer combo |

### Community Building

- Respond to issues within 24 hours (first 3 months are critical)
- Discord/Matrix when >100 stars (empty chat scares people away)
- "Adopted by X" section in README once real users appear
- whisper-rs integration — 2x visibility because whisper-rs community is large

### Milestone Targets

| Stars | What it takes |
|-------|---------------|
| 0→100 | Successful HN launch |
| 100→300 | Python package + awesome-rust listing |
| 300→500 | whisper-rs integration + WASM demo |
| 500→1000 | DER parity with pyannote + "adopted by" social proof |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| DER significantly worse than pyannote | Focus on speed advantage; position as "fast + good enough" rather than "best quality" |
| WeSpeaker ONNX models don't integrate cleanly | Fall back to SpeechBrain ECAPA-TDNN export; test early in Phase 1 |
| maturin/PyO3 cross-platform wheel issues | Start with Linux-only, add platforms incrementally |
| HN launch doesn't land | Multiple launch vectors (r/rust, r/ML, Twitter); content marketing over time |
| Burnout before launch | Phase 1 has hard 6-week cap; launch with what you have |

## Non-goals

- Fine-tuning or training models (use pre-trained only)
- Real-time WebSocket API (post-1.0)
- GPU inference (ONNX Runtime CPU is fast enough for diarization)
- Mobile SDKs (post-1.0)
