---
title: polyvoice v1.0 — Mobile-Quantized Diarization Roadmap
date: 2026-05-07
status: approved (brainstorming)
target_release: v1.0.0
preceding_release: v0.5.2
authors: ekhodzitsky
---

# polyvoice v1.0 — Mobile-Quantized Diarization Roadmap

## Executive summary

polyvoice v1.0 — major redesign оси точности и размера. Цель: достичь паритета с pyannote 3.1 (DER ≤12% VoxConverse, ≤18.5% AMI) при суммарном размере моделей ≤10 МБ в Mobile-профиле и работе в реальном времени на ARM phone CPU.

Ключевые архитектурные изменения относительно v0.5.2:

1. Замена бинарного Silero VAD на **powerset-сегментер** (sherpa-onnx-pyannote-segmentation-3-0): 7-классовая softmax-классификация (silence + 3 спикера + 3 пары), нативная поддержка overlap.
2. Добавление **CAM++** (192-d, ~7M params, ~2 МБ INT8) как дефолтного embedder для Mobile профиля; ResNet34 переезжает в Balanced.
3. Замена AHC default'а на **NME-SC** (auto-K через normalized maximum eigengap).
4. Новый **OverlapResegmenter** post-processing pass.
5. **INT8 квантизация** всех моделей через ONNX Runtime quantization tooling с calibration validation gate.
6. **Two-tier API**: `Profile::Mobile` и `Profile::Balanced`; пользователь не видит разницы в API, только разный bundle моделей.
7. **Breaking redesign API** — релиз номеруется как `v1.0.0`, маркетинговая веха стабильности.

Стриминговый pipeline (`OnlineDiarizer`) **не апгрейдится** в v1.0 — сохраняется с deprecation note, переработка через отдельный спек в v1.1.

Целевые платформы релиза v1.0: Linux x86_64, Linux aarch64, macOS aarch64, Android aarch64. iOS/Windows — позднее без breaking.

---

## 1. Положение polyvoice на рынке

### 1.1 Прямые Rust-конкуренты (по downloads и фичам)

| Crate | Downloads | Архитектура | Ключевое отличие |
|---|---|---|---|
| sherpa-rs (k2-fsa) | ~59 564 | C-биндинг к sherpa-onnx | Лидер дискавери, не идиоматичный Rust API |
| pyannote-rs (thewh1teagle) | ~28 868 | ort + knf-rs + WeSpeaker | Базовый порт, без streaming/Python/DER |
| speakrs (avencera, апр 2026) | новый | Полный клон pyannote community-1: powerset + WeSpeaker + PLDA + VBx | Главный технический конкурент: matched accuracy, CoreML/CUDA, без Python/CLI |
| native-pyannote-rs | ~314 | burn + kaldi-fbank | Pure-Rust без ort, низкая зрелость |
| parakeet-rs | ~139 | NVIDIA Sortformer ONNX | ASR + диаризация до 4 спикеров, English-first |
| polyvoice (текущий) | 266 | Silero + WeSpeaker ResNet34 + AHC/spectral | Лидер DX (Python wheel + DER + RTTM + CLI) |

### 1.2 SOTA, недостижимый при mobile-ограничениях

- **DiariZen** (BUT FIT, 2024): 9.2% VoxConverse / 14.0% AMI. WavLM Large frontend, ~317M params, ~1.2 ГБ FP32. **PyTorch only, ONNX export не существует.** Не вписывается в мобильный бюджет даже после INT8 квантизации.
- **Sortformer** (NVIDIA, 2024): 14.76% DIHARD-III, ~123M params. ONNX export streaming-варианта сломан (NeMo issue #15077).
- **PyannoteAI commercial**: 5.2% VoxConverse, closed-source.

### 1.3 Уникальная ниша polyvoice

Среди Rust-крейтов **никто** не предлагает: (a) Python wheel + Rust crate из одной кодбазы, (b) встроенный DER + RTTM tooling, (c) явный mobile profile с INT8. v1.0 закрепляет эту нишу контрактно: «pyannote-quality диаризация в crate, который реально влезает в телефон».

---

## 2. Цели и non-goals

### 2.1 KPI v1.0.0

| Метрика | v0.5.2 (now) | v1.0 цель | v1.0 stretch |
|---|---|---|---|
| DER VoxConverse @ 0.25s collar | 16.4% | ≤12.0% | ≤10.5% |
| DER AMI @ 0.25s collar | 24.5% | ≤19.0% | ≤17.5% |
| Размер моделей (Mobile profile) | 27 МБ | ≤10 МБ | ≤6 МБ |
| Размер моделей (Balanced profile) | 27 МБ | ≤35 МБ | — |
| Peak RAM (Mobile, 1ч аудио) | ~150 МБ | ≤250 МБ | ≤180 МБ |
| RT-фактор на M2 single-core (Mobile) | 10x | ≥15x | ≥20x |
| RT-фактор на ARM Cortex-A78 single-core (Mobile) | n/a | ≥3x | ≥5x |

### 2.2 In-scope для v1.0

- Powerset segmenter с aggregator
- CAM++ embedder + рефакторинг ResNet34
- NME-SC clusterer + AHC fallback
- Overlap-aware resegmentation pass
- INT8 калибровка для всех моделей
- Two-tier Profile API
- Model registry с manifest TOML и checksum-верификацией
- Android cross-compile + NNAPI EP
- macOS CoreML EP
- Breaking API redesign (v1.0)
- Migration guide v0.5 → v1.0

### 2.3 Out of scope (фиксируем как post-v1.0)

| Сдвинуто на | Почему |
|---|---|
| v1.1 — Streaming v2 | Отдельный сложный спек, требует дискуссии latency/throughput tradeoffs |
| v1.2 — VBx HMM resegmentation | Большая задача, AMI-only выигрыш, низкий impact для mobile |
| v1.3 — ResNet152_LM / ERes2NetV2 (max-accuracy desktop tier) | Расширение Profile набора, не блокирует mobile mission |
| v1.4 — Symphonia (multi-format input MP3/FLAC/OGG) | DX-улучшение, можно потом |
| v2.0 — iOS/Windows wheels + Kotlin/Swift bindings | Отдельный delivery с своим CI |

---

## 3. Архитектура

### 3.1 Pipeline после v1.0 (offline-mode)

```
WAV / PCM (16 kHz mono)
       │
       ▼
┌─────────────────────────────┐
│  Segmenter (powerset)       │   NEW. ONNX, ~1.5M params
│  10s windows, 500ms hop     │   7 классов: silence + S1/S2/S3 + 3 пары
│  → frame-level powerset     │   INT8: ~1.5 MB
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Aggregator                 │   NEW. Pure Rust
│  - sliding-window stitching │   Hungarian matching across windows
│  - speaker-permutation      │
│  - decode → segments        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Embedder                   │   UPGRADE. Trait + 2 реализации
│  CAM++ (Mobile) /           │   overlap-frames МАСКИРУЮТСЯ
│  ResNet34 (Balanced)        │   embedding_exclude_overlap = true
│  + INT8 квантизация         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Clustering                 │   POLISH. Pure Rust
│  NME-SC (default) или AHC   │   auto-K через normalized eigengap
│  cosine + threshold         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  Resegmentation pass        │   NEW. Pure Rust
│  - re-run segmenter         │   для overlap-фреймов assign 2-го
│  - assign 2nd speaker       │   спикера ближайшим cosine-кластером
└──────────────┬──────────────┘
               │
               ▼
        Vec<SpeakerTurn>
```

### 3.2 Изменения по стадиям

| Стадия | v0.5.2 | v1.0 | Импакт |
|---|---|---|---|
| Сегментация | `SileroVad` (бинарный VAD) | `PowersetSegmenter` (7-class) | −3…−5% DER на VoxConverse |
| Эмбеддинги | `FbankOnnxExtractor` (ResNet34 only) | trait `Embedder` + CAM++ / ResNet34 | размер + −1…−2% DER через overlap masking |
| Кластеризация | AHC default, spectral экспериментально | NME-SC default, AHC fallback | −0.5…−1% DER + auto-K |
| Resegmentation | нет | `OverlapResegmenter` opt-in pass | −1…−2% DER на VoxConverse |

Суммарный таргет (см. §2.1): 16.4% → ≤12.0% на VoxConverse, 24.5% → ≤19.0% на AMI; stretch — 10.5% / 17.5% соответственно. Размер моделей 8–10 МБ INT8 для Mobile.

### 3.3 Concurrency model (наследуется без изменений)

| Стадия | Параллелизм |
|---|---|
| Segmentation | Один ONNX session, sequentially по окнам (lightweight model) |
| Embedding | crossbeam-queue session pool, N сессий между K worker threads через rayon |
| Clustering | Sequential (faer eigen многопоточный внутри) |
| Resegmentation | Sequential, дёшево |

---

## 4. Структура src/ после v1.0

```
src/
├── lib.rs                     [refactored]   единая публичная поверхность v1.0
├── types.rs                   [refactored]   SampleRate, Confidence, SpeakerId, Profile, TimeRange
│
├── pipeline/                  [NEW dir]
│   ├── mod.rs                                Pipeline + Builder
│   ├── builder.rs                            PipelineBuilder
│   └── config.rs                             PipelineConfig
│
├── segmentation/              [NEW dir]
│   ├── mod.rs                                trait Segmenter
│   ├── powerset.rs                           PowersetSegmenter (ONNX wrap)
│   ├── decoder.rs                            powerset → frame labels
│   └── aggregator.rs                         sliding-window stitching + Hungarian
│
├── embedding/                 [NEW dir, replaces embedding.rs+ecapa.rs+onnx.rs]
│   ├── mod.rs                                trait Embedder + EmbedderPool
│   ├── cam_pp.rs                             CamPlusPlusExtractor
│   ├── resnet34.rs                           ResNet34Extractor (бывший FbankOnnxExtractor)
│   ├── overlap_mask.rs                       маскирование overlap-фреймов
│   └── pool.rs                               crossbeam-queue session pool
│
├── clustering/                [NEW dir, replaces ahc.rs+spectral.rs+cluster.rs+kmeans.rs]
│   ├── mod.rs                                trait Clusterer
│   ├── nme_sc.rs                             NME-SC (default)
│   ├── ahc.rs                                AHC (fallback)
│   └── eigengap.rs                           normalized maximum eigengap
│
├── resegmentation.rs          [NEW]          OverlapResegmenter
│
├── vad/                       [legacy]
│   └── silero.rs                             перенесён, #[deprecated(since="2.0")]
│
├── features.rs                [kept]         fbank + CMVN
├── der.rs                     [kept]
├── rttm.rs                    [kept]
├── overlap.rs                 [kept]
├── utils.rs                   [kept]
├── wav.rs                     [kept]
├── ffi.rs                     [updated]      адаптация к новому API
│
├── models/                    [NEW dir]
│   ├── mod.rs                                ModelRegistry, Manifest
│   ├── manifest.rs                           Manifest TOML/JSON parsing
│   └── download.rs                           ureq + checksum verify
│
└── bin/
    ├── polyvoice.rs           [refactored]   CLI с --profile
    └── polyvoice-bench.rs     [updated]      bench по профилям + INT8

DELETED:
├── pipeline.rs                                перенесён в pipeline/
├── ahc.rs / cluster.rs / kmeans.rs / spectral.rs  → clustering/
├── ecapa.rs / embedding.rs / onnx.rs            → embedding/
├── silero_vad.rs                                → vad/silero.rs (legacy)
├── vad.rs                                       → удалён, абсорбирован Segmenter trait
├── offline.rs                                   → поглощён Pipeline
├── online.rs                                    → kept как legacy stub до v1.1
```

---

## 5. Публичный API surface

### 5.1 Core types

```rust
// types.rs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    Mobile,    // CAM++ INT8 + powerset INT8. ~10 МБ. RAM ≤200 МБ.
    Balanced,  // ResNet34 INT8 + powerset INT8. ~30 МБ. RAM ≤400 МБ.
    Custom,    // пользователь подкладывает модели через with_*()
}

impl Profile {
    pub const fn embedding_dim(self) -> usize {
        match self {
            Profile::Mobile => 192,
            Profile::Balanced => 256,
            Profile::Custom => 0,
        }
    }
    pub const fn default_threshold(self) -> f32 {
        match self {
            Profile::Mobile => 0.55,
            Profile::Balanced => 0.45,
            Profile::Custom => 0.5,
        }
    }
}
```

### 5.2 PipelineConfig (заменяет DiarizationConfig + VadConfig)

```rust
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub profile: Profile,
    pub sample_rate: SampleRate,         // только 16 kHz; контракт

    // Segmentation
    pub seg_window_secs: f32,            // default 10.0
    pub seg_hop_secs: f32,               // default 0.5

    // Clustering
    pub clusterer: ClustererKind,        // NmeSc (default) | Ahc { threshold }
    pub max_speakers: u8,                // default 20
    pub min_cluster_size: usize,         // default 12

    // Resegmentation
    pub resegment_overlap: bool,         // default true
    pub resegment_min_overlap_secs: f32, // default 0.1

    // Post-processing
    pub min_speech_secs: f32,            // default 0.25
    pub max_gap_secs: f32,               // default 0.5

    // Performance
    pub embedder_pool_size: usize,       // default num_cpus().min(4)
    pub execution_provider: ExecutionProvider,
}

#[derive(Clone, Debug)]
pub enum ClustererKind {
    NmeSc,
    Ahc { threshold: f32 },
}

#[derive(Clone, Copy, Debug)]
pub enum ExecutionProvider {
    Cpu,
    CoreMl,        // macOS / iOS
    Nnapi,         // Android
    Cuda,          // Linux server
    XnnPack,       // INT8 ARM acceleration
}

impl ExecutionProvider {
    pub fn auto() -> Self { /* per-platform default */ }
}
```

### 5.3 Trait surface

```rust
pub trait Segmenter: Send + Sync {
    /// Segments audio into raw speaker-attributed segments.
    ///
    /// Requires: sample rate is 16 kHz, audio.len() >= MIN_AUDIO_SAMPLES (1600).
    /// Guarantees on Ok: all segments have valid `local_speaker_idx < max_local_speakers()`,
    /// timestamps are within [0, audio.len() / 16000], segments sorted by start.
    fn segment(&self, audio: &[f32]) -> Result<Vec<RawSegment>>;
    fn max_local_speakers(&self) -> usize;   // 3 для powerset-3.0
    fn supports_overlap(&self) -> bool;
}

pub struct RawSegment {
    pub time: TimeRange,
    pub local_speaker_idx: u8,   // локальная нумерация в окне
    pub is_overlap: bool,
    pub confidence: Confidence,
}

pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;       // 256 у ResNet34, 192 у CAM++
    /// Caller гарантирует, что overlap-frames замаскированы.
    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>>;
    fn embed_batch(&self, audios: &[&[f32]]) -> Result<Vec<Vec<f32>>>;
}

pub trait Clusterer: Send + Sync {
    /// Embeddings уже L2-нормированы.
    /// Returns compact 0..K numbering без пропусков.
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<GlobalSpeakerId>>;
}
```

### 5.4 Pipeline + Builder

```rust
pub struct Pipeline { /* private */ }

impl Pipeline {
    pub fn builder() -> PipelineBuilder { ... }
    pub fn run(&self, audio: &[f32], sr: SampleRate) -> Result<DiarizationResult>;
}

// Профильный путь:
let pipeline = Pipeline::builder()
    .profile(Profile::Mobile)
    .with_models_from(ModelRegistry::default()?)
    .resegment_overlap(true)
    .build()?;

// Custom путь:
let pipeline = Pipeline::builder()
    .config(my_config)
    .with_segmenter(my_segmenter)
    .with_embedder(my_embedder)
    .with_clusterer(my_clusterer)
    .build()?;
```

`PipelineBuilder::build()` валидирует:
- `profile != Custom` → требует `with_models_from(...)`
- `profile == Custom` → требует явные `with_segmenter`, `with_embedder`, `with_clusterer`
- conflict между `with_segmenter` и non-Custom profile → `Err(ConfigError::Conflict)`

### 5.5 Удаляется (breaking)

| Удалено | Замена |
|---|---|
| `Pipeline::new(DiarizationConfig, VadConfig)` | `Pipeline::builder()` |
| `struct VadConfig` | поглощён `PipelineConfig` |
| `trait VoiceActivityDetector` | поглощён `Segmenter` |
| `fn segment_speech(...)` | внутренняя функция `vad/silero.rs` |
| `struct EnergyVad` | удалён |
| `struct DummyExtractor` | в `embedding/mock.rs`, `#[cfg(test)]` only |
| `struct OnnxEmbeddingExtractor` | заменён `ResNet34Extractor` через `Embedder` trait |
| `fn compute_fbank` | private; trait pattern only |
| `struct OfflineDiarizer` | поглощён `Pipeline` |
| `struct DiarizationConfig` | заменён `PipelineConfig` |

### 5.6 Сохраняется стабильным

`SampleRate`, `Confidence`, `SpeakerId`, `TimeRange`, `SpeakerTurn`, `DiarizationResult`, `der::compute_der`, `rttm::*`, `wav::read_wav`. `OnlineDiarizer` остаётся как deprecated stub с `#[deprecated(note = "redesigned in v1.1")]`.

---

## 6. Model registry

### 6.1 Manifest TOML (поставляется в crate)

> Значения `sha256 = "..."` ниже — placeholder-ы; заполняются на стадии публикации каждой модели через `scripts/validate-int8.sh` (для INT8) или фиксируются один раз для FP32 ссылок на upstream.

```toml
schema = "polyvoice-models-v1"

[profiles.mobile]
segmenter = "powerset_int8"
embedder  = "cam_pp_int8"

[profiles.balanced]
segmenter = "powerset_int8"
embedder  = "resnet34_int8"

[models.powerset_fp32]
url    = "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx"
sha256 = "..."
size   = 5_984_000
input_shape = [1, 1, 160_000]

[models.powerset_int8]
url    = "https://github.com/ekhodzitsky/polyvoice/releases/download/v1.0.0/powerset_int8.onnx"
sha256 = "..."
size   = 1_504_000
calibration = "voxconverse_dev_500_samples"

[models.cam_pp_int8]
url    = "https://github.com/ekhodzitsky/polyvoice/releases/download/v1.0.0/cam_pp_int8.onnx"
sha256 = "..."
size   = 2_048_000

[models.resnet34_int8]
url    = "https://github.com/ekhodzitsky/polyvoice/releases/download/v1.0.0/resnet34_int8.onnx"
sha256 = "..."
size   = 6_400_000
```

### 6.2 ModelRegistry API

```rust
pub struct ModelRegistry { /* private */ }

impl ModelRegistry {
    pub fn default() -> Result<Self>;                           // встроенный manifest, default cache
    pub fn with_cache_dir(p: impl AsRef<Path>) -> Result<Self>;
    pub fn with_manifest(m: Manifest) -> Self;

    /// Скачивает модель в cache_dir если её там нет, верифицирует SHA-256.
    /// Idempotent: повторный вызов сразу возвращает кэшированный path.
    pub fn ensure(&self, model_id: &str) -> Result<PathBuf>;
    pub fn ensure_for_profile(&self, profile: Profile) -> Result<ProfileModels>;
}

pub struct ProfileModels {
    pub segmenter_path: PathBuf,
    pub embedder_path: PathBuf,
}
```

### 6.3 Хостинг моделей

- **INT8 модели** — GitHub Releases этого репо (контролируемая поверхность, версионирование вместе с crate).
- **FP32 модели** — ссылки на upstream (sherpa-onnx releases, WeSpeaker model zoo). Веса не дублируем.
- **Default cache** — `~/.cache/polyvoice/models/` (per-user); конфигурируемо через `with_cache_dir`.

### 6.4 CLI

```
polyvoice download-models --profile mobile           # ~10 МБ bundle
polyvoice download-models --profile balanced         # ~30 МБ bundle
polyvoice download-models --all
polyvoice download-models --profile mobile --output ./models/   # offline bundle
```

---

## 7. Data flow + бюджеты ресурсов

### 7.1 Latency на 60-секундном аудио (Mobile profile)

| Стадия | Apple M2 single-core | ARM Cortex-A78 single-core |
|---|---|---|
| Segmentation (102 окна × ~17 ms) | ~140 мс | ~600 мс |
| Aggregator (Hungarian matching) | ~20 мс | ~50 мс |
| Embedding (50–100 ONNX calls через pool) | ~80 мс | ~320 мс |
| Clustering (NME-SC, N=100) | ~10 мс | ~30 мс |
| Resegmentation (опционально) | ~60 мс | ~200 мс |
| Post-processing | <5 мс | <10 мс |
| **Итого** | ~315 мс (190x RT) | ~1.2 сек (50x RT) |

Cortex-A53 (бюджетный Android): ~6 сек на 60 сек аудио — ≈10× real-time, комфортно для batch use case. Для streaming на A53 необходимо специальное профилирование в v1.1.

### 7.2 Бюджет памяти на пиковом моменте (1 минута, Mobile)

| Источник | Размер |
|---|---|
| Audio buffer Vec\<f32\> (60s × 16k × 4) | 3.84 МБ |
| Powerset session (INT8) | ~1.5 МБ |
| CAM++ session (INT8) | ~2 МБ |
| Powerset window outputs (102 × 589 × 7 × 4) | ~1.6 МБ |
| Embeddings (200 segs × 192 × 4) | ~150 КБ |
| Affinity matrix (100×100) | 40 КБ |
| Workspace для k-means / eigen | <500 КБ |
| ORT + threadpool overhead | ~30–50 МБ |
| **Peak RAM** | **≈40–60 МБ** |

Для часовых файлов используется chunked WAV reader (если в `wav.rs` его ещё нет — добавить как часть M1, см. §10), audio буфер не материализуется целиком. Это small extension, не влияющее на основной pipeline-дизайн.

### 7.3 Граничные случаи

| Случай | Поведение |
|---|---|
| Аудио ≤10 сек (короче окна) | Single window inference + zero-pad до 10s |
| Полная тишина | Powerset выдаёт class 0 → DiarizationResult::empty() |
| Один спикер | NME-SC находит K=1 через eigengap, все сегменты → SpeakerId(0) |
| Сегмент короче min_speech | Дропается на post-processing, не до кластеризации |
| Полностью overlap | Powerset выдаёт классы 4–6, обе личности извлекаются; resegmenter раздаёт второго |
| > 3 одновременных спикеров | Powerset-3.0 натренирован на ≤3; свыше — теряет одного. Документируем как known limitation |
| Аудио > 1 час | Streaming reading через chunked WAV reader (добавляется в M1); aggregator stitches across chunks |
| ORT inference fails | `Err(SegmentationError::InferenceFailed)`, никаких partial results |
| K_estimate > max_speakers | Cap на max_speakers + warn log |

---

## 8. Error handling и контракты

### 8.1 Принципы (наследуются из AGENTS.md)

1. Все публичные функции возвращают `Result<T, E>` с конкретным error type. `anyhow` остаётся только в bin/CLI.
2. `unwrap`/`expect`/`panic!` запрещены в библиотечном коде (enforced by `cargo kimi check` + clippy `unwrap_used = "deny"`). Исключения — `#[cfg(test)]`.
3. Каждая публичная функция несёт **краткий** doc comment в стиле "Requires: ... / Guarantees on Ok: ...". Формальная Hoare-нотация остаётся только в `docs/FORMALISM.md`.
4. Невыполнимые состояния — типами через newtype (SampleRate, Confidence, SpeakerId).

### 8.2 Иерархия ошибок

```rust
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("audio I/O error: {0}")]
    Audio(#[from] AudioError),
    #[error("model loading or inference error: {0}")]
    Model(#[from] ModelError),
    #[error("segmentation stage failed: {0}")]
    Segmentation(#[from] SegmentationError),
    #[error("embedding stage failed: {0}")]
    Embedding(#[from] EmbeddingError),
    #[error("clustering stage failed: {0}")]
    Clustering(#[from] ClusteringError),
    #[error("invalid configuration: {0}")]
    Config(#[from] ConfigError),
    #[error("model registry error: {0}")]
    Registry(#[from] RegistryError),
}

pub type Result<T> = std::result::Result<T, Error>;
```

Каждая sub-error содержит достаточно типизированного контекста для диагностики (`window_idx`, `actual_shape`, `expected/actual checksum`, etc.) без `String` payload-ов кроме реально string-ных полей.

### 8.3 Graceful degradation

| Условие | Поведение | Лог |
|---|---|---|
| Сегмент короче min_speech_secs | Дроп без ошибки | trace |
| K_estimate > max_speakers | Cap | warn |
| Resegmenter не нашёл 2-го спикера выше threshold | Single-speaker turn | debug |
| ORT inference NaN/Inf в одном окне | Zero-fill окно если возможно, иначе Err | warn → error |
| Powerset uncertainty (max softmax < 0.5) | Продолжаем, low-confidence flag | trace |

### 8.4 FFI policy

Все FFI entry points обёрнуты в `std::panic::catch_unwind`. Любая panic → `POLYVOICE_ERR_INTERNAL`. Ошибки кодируются типизированными `polyvoice_status_t` enum значениями. Сообщение об ошибке через `polyvoice_last_error_message()` (thread-local).

```c
typedef enum {
    POLYVOICE_OK = 0,
    POLYVOICE_ERR_INVALID_ARG = 1,
    POLYVOICE_ERR_AUDIO_TOO_SHORT = 2,
    POLYVOICE_ERR_MODEL_LOAD = 10,
    POLYVOICE_ERR_INFERENCE = 11,
    POLYVOICE_ERR_OUT_OF_MEMORY = 20,
    POLYVOICE_ERR_REGISTRY = 30,
    POLYVOICE_ERR_INTERNAL = 99,
} polyvoice_status_t;
```

### 8.5 Panic policy

| Локация | Panic OK? |
|---|---|
| Library code (src/) | ❌ запрещено вне `#[cfg(test)]` |
| Bin code (src/bin/*.rs) | ✅ через `anyhow + ?` (CLI fail-fast) |
| Test code | ✅ |
| Build scripts | ✅ |
| FFI boundary | ❌, `catch_unwind` обёртка |

---

## 9. Testing & verification

### 9.1 CI gate (must pass per merge)

1. `cargo test --lib --features onnx`
2. `cargo test --test '*' --features onnx,cli`
3. `cargo test --doc`
4. proptest ≥10000 случаев на свойство
5. `cargo clippy --all-targets -- -D warnings`
6. `cargo fmt --check`
7. `cargo kimi check` (no unwrap/expect/panic + contracts)
8. `cargo audit`
9. `cargo deny`
10. `cargo semver-checks` (публичный API не сломан внутри минора)
11. Miri (nightly) на selected tests
12. Loom для session pool
13. cargo-fuzz quick run (60s × 4 targets)
14. **DER regression** на 20-файловом VoxConverse-smoke set
15. Cross-platform compile: linux-x86, linux-aarch64, macos-aarch64, android-aarch64

### 9.2 Nightly workflow

- Full DER на VoxConverse-test (232 файла, 43.5 ч)
- Full DER на AMI-test (16 митингов, 9 ч)
- Long cargo-fuzz (5 мин × 4 targets)
- RAM-peak benchmark на 1ч аудио для каждого профиля

### 9.3 DER regression baseline

`tests/der_baseline.json` с pinned baseline + tolerance:

```json
{
  "voxconverse_smoke": {
    "files": 20,
    "profile": "balanced",
    "der": 14.8,
    "tolerance": 1.0,
    "model_versions": {
      "powerset_int8": "sha256:abc...",
      "resnet34_int8": "sha256:def..."
    }
  }
}
```

Update baseline только через PR с явным `[der-baseline-update]` лейблом и review двух мейнтейнеров через CODEOWNERS.

### 9.4 INT8 calibration validation

Перед публикацией каждой INT8-модели запускается `scripts/validate-int8.sh`:

```
calibration: voxconverse_dev_500_samples (random seed 42)
fp32 → int8 metrics:
  segmenter:
    DER on hold-out  : 11.2% → 11.5%  (Δ +0.3, budget ≤ +0.5)
    output divergence: max KL = 0.018  (budget ≤ 0.05)
  embedder (cam_pp):
    EER on VoxCeleb1 : 1.41% → 1.58%  (Δ +0.17, budget ≤ +0.30)
    cosine vs fp32   : mean 0.998, p1 0.991  (budget ≥ 0.99)
```

Без отчёта в budget — модель не публикуется.

### 9.5 Property tests (расширение существующих)

- `aggregator::stitch`: single window stitch == decode (no-op aggregation); total durations bounded
- `embedding::*::embed`: output dim == self.dim() ∧ L2 norm == 1
- `clustering::*::cluster`: identical embeddings → один cluster; compact 0..K numbering
- `nme_sc::eigengap`: stability на shuffled embedding sequences

### 9.6 Miri sweep (новые таргеты)

```
tests/miri_aggregator.rs       — zero/single/many windows, max_speakers
tests/miri_clustering.rs       — single embedding, eigengap decision
tests/miri_resegmentation.rs   — no-overlap, full-overlap
```

ONNX-зависимое всё ещё под `#[cfg(not(miri))]` — `ort` не Miri-friendly.

### 9.7 Loom (расширение `tests/loom_pool.rs`)

Покрытие `EmbedderPool` concurrent embed на 4 нитях через crossbeam-queue.

### 9.8 cargo-fuzz (4 существующих + 3 новых)

```
fuzz_powerset_decoder.rs       — random logits → decoder, no panic
fuzz_aggregator_stitch.rs      — random window outputs
fuzz_nme_sc_cluster.rs         — random embeddings
```

### 9.9 CI matrix

```yaml
matrix:
  - { os: ubuntu-22.04, target: x86_64-unknown-linux-gnu, features: "onnx,cli" }
  - { os: ubuntu-22.04, target: aarch64-unknown-linux-gnu, features: "onnx,xnnpack" }
  - { os: macos-14, target: aarch64-apple-darwin, features: "onnx,coreml" }
  - { os: ubuntu-22.04, target: aarch64-linux-android, features: "onnx,nnapi" }   # via cargo-ndk
  - { os: ubuntu-22.04, target: x86_64-unknown-linux-gnu, features: "onnx,cuda" }
  - { os: ubuntu-22.04, miri: true }
  - { os: ubuntu-22.04, loom: true }
  - { os: ubuntu-22.04, target: wasm32-unknown-unknown, features: "" }            # smoke compile
```

### 9.10 Release gate (для v1.0.0 specifically)

`scripts/release-gate.sh` блокирует тэг при failure любого:

| Check | Threshold |
|---|---|
| DER VoxConverse Mobile | ≤ 12.5% |
| DER VoxConverse Balanced | ≤ 11.5% |
| DER AMI Mobile | ≤ 19.5% |
| DER AMI Balanced | ≤ 18.5% |
| Mobile model bundle | ≤ 10 МБ |
| Balanced model bundle | ≤ 35 МБ |
| Peak RSS на 1ч аудио (Mobile) | ≤ 250 МБ |
| RT-фактор на M2 single-core (Mobile) | ≥ 15x |
| RT-фактор на A78 (через QEMU) (Mobile) | ≥ 3x |
| Все CI matrix зелёные | required |
| `cargo semver-checks` vs v0.5.x | breaking confirmed (manual ack) |
| Doc coverage | 100% публичных items |

---

## 10. Milestone plan

### 10.1 Зависимости

```
M0 → M1 → M5 → ┐
M0 → M2 → M5 → │
M0 → M3 → ─── ┤
M0 → M4 → ─── ┤
              ▼
              M6 → M7 → M8 → M9
```

| ID | Название | Длительность | Блокирует | Deliverables |
|---|---|---|---|---|
| **M0** | Plumbing & registry | 1 нед | M1, M2, M5 | Cargo features (`coreml`/`nnapi`/`xnnpack`/`profile-*`), `ModelRegistry` skeleton с manifest TOML, `polyvoice download-models --profile` (без INT8), CI matrix (без Android), `release-gate.sh` stub |
| **M1** | Powerset segmenter | 2–3 нед | M5, M7 | `segmentation::PowersetSegmenter`, `decoder.rs`, `aggregator.rs` с Hungarian. ONNX FP32 inference на VoxConverse smoke. property tests + Miri-friendly aggregator tests |
| **M2** | Embedder trait + CAM++ | 1–2 нед | M5, M7 | `Embedder` trait, `CamPlusPlusExtractor`, рефакторинг → `ResNet34Extractor`, `overlap_mask`. Pool на новом trait |
| **M3** | NME-SC clusterer | 1 нед | M7 | `NmeScClusterer` с auto-K через normalized eigengap, проверка на synthetic + smoke. AHC помечен как fallback |
| **M4** | Overlap resegmenter | 1 нед | M7 | `OverlapResegmenter` pass, не ломает single-speaker / silence; ±0.5% DER на VoxConverse-smoke |
| **M5** | INT8 quantization | 2 нед | M7, M9 | INT8 артефакты для powerset + cam_pp + resnet34, `scripts/quantize-models.sh`, `scripts/validate-int8.sh`, calibration reports опубликованы. INT8 модели в GitHub Releases (pre-release tag) |
| **M6** | Pipeline + Profile API | 1–2 нед | M7 | `Pipeline::builder()`, `Profile::Mobile/Balanced/Custom`, `PipelineConfig`. Удаление старого `DiarizationConfig`/`VadConfig`. Все integration tests прошли |
| **M7** | CLI + Python + FFI | 1 нед | M8 | `polyvoice diarize --profile`, `polyvoice bench`, обновлённые Python bindings (с `.pyi` через `pyo3-stub-gen`), `polyvoice.h` v1.0, миграционная заметка в README |
| **M8** | Android + multi-platform CI | 2 нед | M9 | cargo-ndk build для aarch64-linux-android, NNAPI EP интеграция, Android RT-bench (через QEMU), aarch64-linux-gnu CI зелёный, macOS CoreML EP зелёный |
| **M9** | Release polish & v1.0.0 GA | 1 нед | — | `release-gate.sh` все check-ы зелёные, README/CHANGELOG/migrating-from-0.5 готовы, blog post draft, синхронная публикация crates.io + PyPI + GitHub Release |

**Итого: ~13–15 недель calendar work** (≈3.5 месяца). Параллелизация M1 + M2 + M3 + M4 короче на ARM-multitasking при наличии второго человека.

### 10.2 Release candidates

| RC | Когда | Что в нём |
|---|---|---|
| 0.6.0-alpha.1 | После M2+M3 | Новый pipeline, FP32-only, без resegmentation, без INT8. Внутренняя проверка DER |
| 0.6.0-beta.1 | После M5 | Полный pipeline + INT8. Early adopters пробуют |
| 0.6.0-rc.1 | После M7 | API заморожен, документация финал, осталось multi-platform CI |
| **1.0.0** | После M9 | Release gate green |

### 10.3 Risks с mitigation

| Риск | Вероятность | Mitigation |
|---|---|---|
| Hungarian matching реализован неправильно → DER не падает | средняя | Pin pyannote reference Python implementation, golden test против её output на 5 файлов |
| INT8 квантизация даёт >1% DER hit | средняя | Бюджет +0.5%; fallback на FP16 для embedder, INT8 только для segmenter |
| sherpa-onnx-pyannote-segmentation-3-0 имеет лицензионные ограничения | низкая | Проверить лицензию на M0; иметь fallback (own export через optimum-onnx) |
| Android NDK cross + ort оказался сложнее ожидаемого | средняя | M8 буфер 2 недели, fallback — отложить Android в v1.0.1 |
| NME-SC eigengap нестабилен на коротких аудио | низкая | Property test stability + fallback на AHC при N < 5 |
| WeSpeaker CAM++ ONNX не оптимизирован для INT8 | низкая | Проверить в M5 первой неделе; fallback FP16 для CAM++ |
| Upstream sherpa-onnx меняет формат весов | низкая | Pin specific release tag в manifest, fork в наш Releases на v1.0 если нужно |

---

## 11. Migration guide v0.5 → v1.0 (черновик)

### 11.1 Rust API

**Было (v0.5):**
```rust
let extractor = FbankOnnxExtractor::new(Path::new("models/wespeaker_resnet34.onnx"), 256, 4)?;
let mut vad = SileroVad::new(Path::new("models/silero_vad.onnx"), 512)?;
let pipeline = Pipeline::new(DiarizationConfig::default(), VadConfig::default());
let result = pipeline.run(&samples, &extractor, &mut vad)?;
```

**Стало (v1.0):**
```rust
let registry = ModelRegistry::default()?;
let pipeline = Pipeline::builder()
    .profile(Profile::Balanced)              // или Profile::Mobile
    .with_models_from(registry)
    .build()?;
let result = pipeline.run(&samples, sr)?;
```

### 11.2 Python API

**Было:** `polyvoice.Pipeline("models/")`
**Стало:** `polyvoice.Pipeline.mobile()` или `polyvoice.Pipeline.balanced()`

### 11.3 CLI

**Было:** `polyvoice diarize meeting.wav`
**Стало:** `polyvoice diarize meeting.wav --profile mobile` (default остаётся `balanced` для совместимости поведения).

### 11.4 Что сломается

- `OfflineDiarizer` удалён — использовать `Pipeline`
- `DiarizationConfig`, `VadConfig` удалены — использовать `PipelineConfig`
- `OnlineDiarizer` остаётся, но deprecated — переезжает в v1.1

---

## 12. References

### Research findings

- pyannote.audio 3.1 internals — [HuggingFace config.yaml](https://huggingface.co/ivrit-ai/pyannote-speaker-diarization-3.1/blame/main/config.yaml), [pyannote/segmentation-3.0 model card](https://huggingface.co/pyannote/segmentation-3.0)
- Powerset multi-class loss — Plaquet & Bredin, INTERSPEECH 2023, [arXiv:2310.13025](https://arxiv.org/html/2310.13025v1)
- DiariZen (BUT FIT) — [GitHub](https://github.com/BUTSpeechFIT/DiariZen)
- Sortformer (NVIDIA) — [arXiv:2409.06656](https://arxiv.org/abs/2409.06656)
- NME-SC — Park et al., [arXiv:2003.02405](https://arxiv.org/abs/2003.02405), [reference impl](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering)
- VBx (BUT) — [GitHub](https://github.com/BUTSpeechFIT/VBx) (out of scope для v1.0)
- 2025 Diarization benchmark — [arXiv:2509.26177](https://arxiv.org/html/2509.26177v1)

### Models

- sherpa-onnx-pyannote-segmentation-3-0 — [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)
- WeSpeaker model zoo (CAM++, ResNet34, ResNet152, ERes2Net) — [pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- 3D-Speaker (ERes2NetV2) — [GitHub](https://github.com/modelscope/3D-Speaker)

### Rust ecosystem

- speakrs — [crates.io](https://crates.io/crates/speakrs)
- pyannote-rs — [crates.io](https://crates.io/crates/pyannote-rs)
- sherpa-rs — [crates.io](https://crates.io/crates/sherpa-rs)

### Internal docs

- `AGENTS.md` — Rust correctness rules (cargo-kimi)
- `docs/FORMALISM.md` — formal Hoare triple notation (для теории, не для daily docstrings)
- `docs/PIPELINE.md` — development process Spec → Type → Implement → Verify
