//! Streaming Sortformer E2E diarizer.
//!
//! Ports the NeMo / parakeet-rs streaming loop onto polyvoice's
//! [`InferenceRuntime`]: FIFO + speaker-cache state tensors are passed as
//! ordinary named inputs/outputs between chunk calls (no ort imports here).

use super::config::{
    EMB_DIM, FRAME_DURATION_SECS, MAX_SPEAKERS, N_MELS, SAMPLE_RATE, SUBSAMPLING, SortformerConfig,
    SortformerError,
};
use super::features::SortformerFeatures;
use crate::onnx::{InferenceRuntime, InferenceTensor, NamedTensor};
use crate::types::{SpeakerId, SpeakerTurn, TimeRange};
use std::collections::HashMap;
use std::path::Path;

// Cache compression constants (NeMo).
const SPKCACHE_SIL_FRAMES_PER_SPK: usize = 3;
const PRED_SCORE_THRESHOLD: f32 = 0.25;
const STRONG_BOOST_RATE: f32 = 0.75;
const WEAK_BOOST_RATE: f32 = 1.5;
const MIN_POS_SCORES_RATE: f32 = 0.5;
const SIL_THRESHOLD: f32 = 0.2;
const MAX_INDEX: usize = 99_999;

const OUT_PREDS: &str = "spkcache_fifo_chunk_preds";
const OUT_EMBS: &str = "chunk_pre_encode_embs";

/// End-to-end Streaming Sortformer diarizer.
///
/// Generic over [`InferenceRuntime`] so unit tests can inject a mock without
/// loading the ~470 MB ONNX weights.
pub struct SortformerDiarizer<R: InferenceRuntime> {
    session: R,
    config: SortformerConfig,
    features: SortformerFeatures,
    // Streaming state (NeMo-compatible).
    /// Speaker cache embeddings, row-major `[frames * EMB_DIM]`, time = frames.
    spkcache: Vec<f32>,
    spkcache_frames: usize,
    /// Optional cache predictions `[frames * MAX_SPEAKERS]`.
    spkcache_preds: Option<Vec<f32>>,
    /// FIFO embeddings.
    fifo: Vec<f32>,
    fifo_frames: usize,
    /// FIFO predictions.
    fifo_preds: Vec<f32>,
    mean_sil_emb: Vec<f32>,
    n_sil_frames: usize,
    /// Buffered samples for [`Self::feed`].
    audio_buffer: Vec<f32>,
    elapsed_samples: usize,
}

impl SortformerDiarizer<crate::onnx::OrtSession> {
    /// Load from an ONNX model path with default config.
    pub fn from_path(model_path: impl AsRef<Path>) -> Result<Self, SortformerError> {
        Self::from_path_with_config(model_path, SortformerConfig::default())
    }

    /// Load from path with explicit config (validates max_speakers first).
    pub fn from_path_with_config(
        model_path: impl AsRef<Path>,
        mut config: SortformerConfig,
    ) -> Result<Self, SortformerError> {
        config.validate()?;
        let session = crate::onnx::build_session_with_ep(
            model_path.as_ref(),
            crate::onnx::ExecutionProvider::Cpu,
            None,
        )
        .map_err(|e| SortformerError::Load(e.to_string()))?;

        // Override geometry from ONNX metadata when present.
        if let Ok(props) = session.custom_metadata_props() {
            if let Some(v) = props.get("chunk_len").and_then(|s| s.parse().ok()) {
                config.chunk_len = v;
            }
            if let Some(v) = props.get("fifo_len").and_then(|s| s.parse().ok()) {
                config.fifo_len = v;
            }
            if let Some(v) = props.get("spkcache_len").and_then(|s| s.parse().ok()) {
                config.spkcache_len = v;
            }
            if let Some(v) = props.get("right_context").and_then(|s| s.parse().ok()) {
                config.right_context = v;
            }
        }

        Ok(Self::from_runtime(session, config))
    }
}

impl<R: InferenceRuntime> SortformerDiarizer<R> {
    /// Wrap an existing inference runtime (used by tests with mocks).
    pub fn from_runtime(session: R, config: SortformerConfig) -> Self {
        let mut this = Self {
            session,
            config,
            features: SortformerFeatures::new(),
            spkcache: Vec::new(),
            spkcache_frames: 0,
            spkcache_preds: None,
            fifo: Vec::new(),
            fifo_frames: 0,
            fifo_preds: Vec::new(),
            mean_sil_emb: vec![0.0; EMB_DIM],
            n_sil_frames: 0,
            audio_buffer: Vec::new(),
            elapsed_samples: 0,
        };
        this.reset();
        this
    }

    /// Access config.
    pub fn config(&self) -> &SortformerConfig {
        &self.config
    }

    /// Nominal streaming latency in seconds.
    pub fn latency_secs(&self) -> f32 {
        self.config.latency_secs()
    }

    /// Reset FIFO / speaker-cache / audio buffer for a new stream.
    pub fn reset(&mut self) {
        self.spkcache.clear();
        self.spkcache_frames = 0;
        self.spkcache_preds = None;
        self.fifo.clear();
        self.fifo_frames = 0;
        self.fifo_preds.clear();
        self.mean_sil_emb.fill(0.0);
        self.n_sil_frames = 0;
        self.audio_buffer.clear();
        self.elapsed_samples = 0;
    }

    /// Offline diarization of a full 16 kHz mono buffer. Resets state first.
    pub fn diarize(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<SpeakerTurn>, SortformerError> {
        self.reset();
        if audio_16k_mono.is_empty() {
            return Ok(Vec::new());
        }
        let preds = self.predict_frames(audio_16k_mono)?;
        let filtered = self.median_filter(&preds);
        let mut turns = self.binarize(&filtered);
        clip_turns_to_audio(&mut turns, audio_16k_mono.len());
        Ok(turns)
    }

    /// Streaming chunk: preserves internal state across calls.
    ///
    /// Segment times are relative to this chunk (start at 0).
    pub fn diarize_chunk(
        &mut self,
        audio_16k_mono: &[f32],
    ) -> Result<Vec<SpeakerTurn>, SortformerError> {
        if audio_16k_mono.is_empty() {
            return Ok(Vec::new());
        }
        let preds = self.predict_frames(audio_16k_mono)?;
        let filtered = self.median_filter(&preds);
        let mut turns = self.binarize(&filtered);
        clip_turns_to_audio(&mut turns, audio_16k_mono.len());
        Ok(turns)
    }

    /// Buffered streaming: accumulate samples, emit turns with absolute times.
    pub fn feed(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<SpeakerTurn>, SortformerError> {
        self.audio_buffer.extend_from_slice(audio_16k_mono);
        let feed_samples =
            (self.config.chunk_len + self.config.right_context) * SUBSAMPLING * HOP_LENGTH;
        let stride_samples = self.config.chunk_len * SUBSAMPLING * HOP_LENGTH;

        let mut all = Vec::new();
        while self.audio_buffer.len() >= feed_samples {
            let window: Vec<f32> = self.audio_buffer[..feed_samples].to_vec();
            let (mel, n_frames) = self.features.extract_log_mel(&window)?;
            let feed_size = (self.config.chunk_len + self.config.right_context) * SUBSAMPLING;
            let current_len = n_frames.min(feed_size);
            let chunk_feat = pad_or_slice_mel(&mel, n_frames, feed_size);
            let chunk_preds = self.streaming_update(&chunk_feat, current_len)?;
            let filtered = self.median_filter(&chunk_preds);
            let sample_offset = self.elapsed_samples as f64 / SAMPLE_RATE as f64;
            let chunk_dur =
                (self.config.chunk_len * SUBSAMPLING * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
            let mut turns = self.binarize(&filtered);
            for t in &mut turns {
                t.time.start += sample_offset;
                t.time.end = (t.time.end + sample_offset).min(sample_offset + chunk_dur);
            }
            turns.retain(|t| t.time.end > t.time.start);
            all.extend(turns);

            self.audio_buffer.drain(..stride_samples);
            self.elapsed_samples += stride_samples;
        }
        Ok(all)
    }

    /// Flush remaining buffered audio at end of stream.
    pub fn flush(&mut self) -> Result<Vec<SpeakerTurn>, SortformerError> {
        if self.audio_buffer.is_empty() {
            return Ok(Vec::new());
        }
        let remaining = std::mem::take(&mut self.audio_buffer);
        let (mel, n_frames) = self.features.extract_log_mel(&remaining)?;
        let feed_size = (self.config.chunk_len + self.config.right_context) * SUBSAMPLING;
        let current_len = n_frames.min(feed_size);
        let chunk_feat = pad_or_slice_mel(&mel, n_frames, feed_size);
        let chunk_preds = self.streaming_update(&chunk_feat, current_len)?;
        let filtered = self.median_filter(&chunk_preds);
        let sample_offset = self.elapsed_samples as f64 / SAMPLE_RATE as f64;
        let remaining_secs = remaining.len() as f64 / SAMPLE_RATE as f64;
        let mut turns = self.binarize(&filtered);
        for t in &mut turns {
            t.time.start += sample_offset;
            t.time.end = (t.time.end + sample_offset).min(sample_offset + remaining_secs);
        }
        turns.retain(|t| t.time.end > t.time.start);
        self.elapsed_samples += remaining.len();
        Ok(turns)
    }

    /// Full-feature streaming prediction over mel frames of an audio buffer.
    fn predict_frames(&mut self, audio: &[f32]) -> Result<Vec<f32>, SortformerError> {
        let (mel, total_frames) = self.features.extract_log_mel(audio)?;
        if total_frames == 0 {
            return Ok(Vec::new());
        }
        let chunk_stride = self.config.chunk_len * SUBSAMPLING;
        let feed_size = (self.config.chunk_len + self.config.right_context) * SUBSAMPLING;
        let num_chunks = total_frames.div_ceil(chunk_stride);
        let mut all = Vec::new();

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_stride;
            let end = (start + feed_size).min(total_frames);
            let current_len = end - start;
            let mut chunk_mel = vec![0.0f32; feed_size * N_MELS];
            for t in 0..current_len {
                let src = (start + t) * N_MELS;
                let dst = t * N_MELS;
                chunk_mel[dst..dst + N_MELS].copy_from_slice(&mel[src..src + N_MELS]);
            }
            let preds = self.streaming_update(&chunk_mel, current_len)?;
            all.extend(preds);
        }
        Ok(all)
    }

    /// One NeMo `streaming_update` step.
    ///
    /// `chunk_mel` is row-major `[feed_size, N_MELS]` (zero-padded if needed).
    /// `current_len` is the number of valid mel frames before padding.
    /// Returns flat predictions `[keep_frames * MAX_SPEAKERS]`.
    fn streaming_update(
        &mut self,
        chunk_mel: &[f32],
        current_len: usize,
    ) -> Result<Vec<f32>, SortformerError> {
        let feed_size = (self.config.chunk_len + self.config.right_context) * SUBSAMPLING;
        if chunk_mel.len() != feed_size * N_MELS {
            return Err(SortformerError::Shape(format!(
                "chunk mel len {} != feed_size*N_MELS {}",
                chunk_mel.len(),
                feed_size * N_MELS
            )));
        }

        let spkcache_len = self.spkcache_frames;
        let fifo_len = self.fifo_frames;

        let chunk = InferenceTensor::f32(vec![1, feed_size, N_MELS], chunk_mel.to_vec());
        let chunk_lengths = InferenceTensor::i64(vec![1], vec![current_len as i64]);
        let spkcache = InferenceTensor::f32(
            vec![1, spkcache_len, EMB_DIM],
            if spkcache_len == 0 {
                Vec::new()
            } else {
                self.spkcache.clone()
            },
        );
        let spkcache_lengths = InferenceTensor::i64(vec![1], vec![spkcache_len as i64]);
        let fifo = InferenceTensor::f32(
            vec![1, fifo_len, EMB_DIM],
            if fifo_len == 0 {
                Vec::new()
            } else {
                self.fifo.clone()
            },
        );
        let fifo_lengths = InferenceTensor::i64(vec![1], vec![fifo_len as i64]);

        let outputs = self
            .session
            .run(&[
                NamedTensor::new("chunk", &chunk),
                NamedTensor::new("chunk_lengths", &chunk_lengths),
                NamedTensor::new("spkcache", &spkcache),
                NamedTensor::new("spkcache_lengths", &spkcache_lengths),
                NamedTensor::new("fifo", &fifo),
                NamedTensor::new("fifo_lengths", &fifo_lengths),
            ])
            .map_err(|e| SortformerError::Inference(e.to_string()))?;

        let by_name = map_outputs(self.session.output_names(), outputs)?;

        let preds_t = by_name
            .get(OUT_PREDS)
            .ok_or_else(|| SortformerError::MissingOutput {
                name: OUT_PREDS,
                available: by_name.keys().cloned().collect(),
            })?;
        let embs_t = by_name
            .get(OUT_EMBS)
            .ok_or_else(|| SortformerError::MissingOutput {
                name: OUT_EMBS,
                available: by_name.keys().cloned().collect(),
            })?;

        let preds = preds_t
            .as_f32_slice()
            .map_err(|e| SortformerError::Inference(e.to_string()))?;
        let new_embs = embs_t
            .as_f32_slice()
            .map_err(|e| SortformerError::Inference(e.to_string()))?;

        // preds shape: [1, spkcache_len + fifo_len + chunk_frames, 4]
        // embs shape:  [1, chunk_emb_frames, 512]
        let preds_frames = preds_shape_frames(preds_t)?;
        let emb_frames = embs_shape_frames(embs_t)?;

        let valid_frames = current_len.div_ceil(SUBSAMPLING);
        let keep = self.config.chunk_len.min(valid_frames).min(emb_frames);

        // Slice chunk predictions after cache+fifo prefix.
        let prefix = spkcache_len + fifo_len;
        if prefix + keep > preds_frames {
            return Err(SortformerError::Shape(format!(
                "preds frames {preds_frames} < prefix {prefix} + keep {keep}"
            )));
        }
        let mut chunk_preds = vec![0.0f32; keep * MAX_SPEAKERS];
        for t in 0..keep {
            let src = (prefix + t) * MAX_SPEAKERS;
            let dst = t * MAX_SPEAKERS;
            chunk_preds[dst..dst + MAX_SPEAKERS].copy_from_slice(&preds[src..src + MAX_SPEAKERS]);
        }

        // FIFO predictions for current fifo frames (recomputed).
        let mut fifo_preds_now = vec![0.0f32; fifo_len * MAX_SPEAKERS];
        if fifo_len > 0 {
            for t in 0..fifo_len {
                let src = (spkcache_len + t) * MAX_SPEAKERS;
                let dst = t * MAX_SPEAKERS;
                fifo_preds_now[dst..dst + MAX_SPEAKERS]
                    .copy_from_slice(&preds[src..src + MAX_SPEAKERS]);
            }
        }

        // Chunk embeddings.
        let mut chunk_embs = vec![0.0f32; keep * EMB_DIM];
        for t in 0..keep {
            let src = t * EMB_DIM;
            let dst = t * EMB_DIM;
            chunk_embs[dst..dst + EMB_DIM].copy_from_slice(&new_embs[src..src + EMB_DIM]);
        }

        // Append chunk to FIFO.
        self.fifo.extend_from_slice(&chunk_embs);
        self.fifo_frames += keep;
        if fifo_len > 0 {
            // Replace fifo_preds with [old_fifo_preds | chunk_preds]
            self.fifo_preds = fifo_preds_now;
            self.fifo_preds.extend_from_slice(&chunk_preds);
        } else {
            self.fifo_preds = chunk_preds.clone();
        }

        // Pop FIFO → speaker cache when over limit.
        if self.fifo_frames > self.config.fifo_len {
            let mut pop_out_len = self.config.chunk_len;
            pop_out_len =
                pop_out_len.max(valid_frames.saturating_sub(self.config.fifo_len) + fifo_len);
            pop_out_len = pop_out_len.min(self.fifo_frames);

            let pop_embs = self.fifo[..pop_out_len * EMB_DIM].to_vec();
            let pop_preds = self.fifo_preds[..pop_out_len * MAX_SPEAKERS].to_vec();

            self.update_silence_profile(&pop_embs, &pop_preds, pop_out_len);

            self.fifo.drain(..pop_out_len * EMB_DIM);
            self.fifo_preds.drain(..pop_out_len * MAX_SPEAKERS);
            self.fifo_frames -= pop_out_len;

            // Append to cache.
            self.spkcache.extend_from_slice(&pop_embs);
            self.spkcache_frames += pop_out_len;
            if let Some(ref mut cp) = self.spkcache_preds {
                cp.extend_from_slice(&pop_preds);
            }

            if self.spkcache_frames > self.config.spkcache_len {
                if self.spkcache_preds.is_none() {
                    // Seed cache preds from the current run's cache prefix.
                    let mut initial = vec![0.0f32; spkcache_len * MAX_SPEAKERS];
                    for t in 0..spkcache_len {
                        let src = t * MAX_SPEAKERS;
                        initial[src..src + MAX_SPEAKERS]
                            .copy_from_slice(&preds[src..src + MAX_SPEAKERS]);
                    }
                    initial.extend_from_slice(&pop_preds);
                    self.spkcache_preds = Some(initial);
                }
                self.compress_spkcache();
            }
        }

        Ok(chunk_preds)
    }

    fn update_silence_profile(&mut self, embs: &[f32], preds: &[f32], n_frames: usize) {
        for t in 0..n_frames {
            let sum: f32 = (0..MAX_SPEAKERS).map(|s| preds[t * MAX_SPEAKERS + s]).sum();
            if sum < SIL_THRESHOLD {
                let emb = &embs[t * EMB_DIM..(t + 1) * EMB_DIM];
                let old_n = self.n_sil_frames as f32;
                self.n_sil_frames += 1;
                let new_n = self.n_sil_frames as f32;
                for i in 0..EMB_DIM {
                    self.mean_sil_emb[i] = (self.mean_sil_emb[i] * old_n + emb[i]) / new_n;
                }
            }
        }
    }

    fn compress_spkcache(&mut self) {
        let Some(cache_preds) = self.spkcache_preds.clone() else {
            return;
        };
        let n_frames = self.spkcache_frames;
        let per_spk = self.config.spkcache_len / MAX_SPEAKERS;
        if per_spk <= SPKCACHE_SIL_FRAMES_PER_SPK {
            // Truncate.
            let keep = self.config.spkcache_len.min(n_frames);
            self.spkcache.truncate(keep * EMB_DIM);
            self.spkcache_frames = keep;
            if let Some(ref mut p) = self.spkcache_preds {
                p.truncate(keep * MAX_SPEAKERS);
            }
            return;
        }
        let spkcache_len_per_spk = per_spk - SPKCACHE_SIL_FRAMES_PER_SPK;
        let strong_boost = (spkcache_len_per_spk as f32 * STRONG_BOOST_RATE) as usize;
        let weak_boost = (spkcache_len_per_spk as f32 * WEAK_BOOST_RATE) as usize;
        let min_pos = (spkcache_len_per_spk as f32 * MIN_POS_SCORES_RATE) as usize;

        let mut scores = get_log_pred_scores(&cache_preds, n_frames);
        scores = disable_low_scores(&cache_preds, scores, n_frames, min_pos);
        scores = boost_topk_scores(scores, n_frames, strong_boost, 2.0);
        scores = boost_topk_scores(scores, n_frames, weak_boost, 1.0);

        // Silence-frame placeholders.
        let mut padded =
            vec![f32::NEG_INFINITY; (n_frames + SPKCACHE_SIL_FRAMES_PER_SPK) * MAX_SPEAKERS];
        padded[..n_frames * MAX_SPEAKERS].copy_from_slice(&scores);
        for i in n_frames..n_frames + SPKCACHE_SIL_FRAMES_PER_SPK {
            for j in 0..MAX_SPEAKERS {
                padded[i * MAX_SPEAKERS + j] = f32::INFINITY;
            }
        }

        let (indices, disabled) = get_topk_indices(
            &padded,
            n_frames + SPKCACHE_SIL_FRAMES_PER_SPK,
            n_frames,
            self.config.spkcache_len,
        );

        let (new_embs, new_preds) =
            self.gather_spkcache(&indices, &disabled, self.config.spkcache_len);
        self.spkcache = new_embs;
        self.spkcache_frames = self.config.spkcache_len;
        self.spkcache_preds = Some(new_preds);
    }

    fn gather_spkcache(
        &self,
        indices: &[usize],
        disabled: &[bool],
        out_len: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut new_embs = vec![0.0f32; out_len * EMB_DIM];
        let mut new_preds = vec![0.0f32; out_len * MAX_SPEAKERS];
        let cache_preds = self
            .spkcache_preds
            .as_ref()
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        for (i, (&idx, &is_dis)) in indices.iter().zip(disabled.iter()).enumerate() {
            if i >= out_len {
                break;
            }
            if is_dis {
                new_embs[i * EMB_DIM..(i + 1) * EMB_DIM].copy_from_slice(&self.mean_sil_emb);
            } else if idx < self.spkcache_frames {
                new_embs[i * EMB_DIM..(i + 1) * EMB_DIM]
                    .copy_from_slice(&self.spkcache[idx * EMB_DIM..(idx + 1) * EMB_DIM]);
                if !cache_preds.is_empty() {
                    new_preds[i * MAX_SPEAKERS..(i + 1) * MAX_SPEAKERS].copy_from_slice(
                        &cache_preds[idx * MAX_SPEAKERS..(idx + 1) * MAX_SPEAKERS],
                    );
                }
            }
        }
        (new_embs, new_preds)
    }

    fn median_filter(&self, preds: &[f32]) -> Vec<f32> {
        let window = self.config.post.median_window;
        if window <= 1 || preds.is_empty() {
            return preds.to_vec();
        }
        let n_frames = preds.len() / MAX_SPEAKERS;
        let half = window / 2;
        let mut filtered = preds.to_vec();
        for spk in 0..MAX_SPEAKERS {
            for t in 0..n_frames {
                let start = t.saturating_sub(half);
                let end = (t + half + 1).min(n_frames);
                let mut values: Vec<f32> = (start..end)
                    .map(|i| preds[i * MAX_SPEAKERS + spk])
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                filtered[t * MAX_SPEAKERS + spk] = values[values.len() / 2];
            }
        }
        filtered
    }

    fn binarize(&self, preds: &[f32]) -> Vec<SpeakerTurn> {
        let n_frames = preds.len() / MAX_SPEAKERS;
        if n_frames == 0 {
            return Vec::new();
        }
        let post = &self.config.post;
        let max_spk = self.config.max_speakers.min(MAX_SPEAKERS);
        let mut segments = Vec::new();

        for spk in 0..max_spk {
            let mut in_seg = false;
            let mut seg_start = 0usize;
            let mut temp = Vec::new();

            for t in 0..n_frames {
                let p = preds[t * MAX_SPEAKERS + spk];
                if p >= post.onset && !in_seg {
                    in_seg = true;
                    seg_start = t;
                } else if p < post.offset && in_seg {
                    in_seg = false;
                    let start = (seg_start as f64 * FRAME_DURATION_SECS as f64
                        - post.pad_onset as f64)
                        .max(0.0);
                    let end = t as f64 * FRAME_DURATION_SECS as f64 + post.pad_offset as f64;
                    if end - start >= post.min_duration_on as f64 {
                        temp.push(SpeakerTurn {
                            speaker: SpeakerId(spk as u32),
                            time: TimeRange { start, end },
                            text: None,
                            stable: true,
                        });
                    }
                }
            }
            if in_seg {
                let start = (seg_start as f64 * FRAME_DURATION_SECS as f64 - post.pad_onset as f64)
                    .max(0.0);
                let end = n_frames as f64 * FRAME_DURATION_SECS as f64 + post.pad_offset as f64;
                if end - start >= post.min_duration_on as f64 {
                    temp.push(SpeakerTurn {
                        speaker: SpeakerId(spk as u32),
                        time: TimeRange { start, end },
                        text: None,
                        stable: true,
                    });
                }
            }

            // Merge close gaps.
            if temp.len() > 1 {
                let mut merged: Vec<SpeakerTurn> = Vec::with_capacity(temp.len());
                for seg in temp {
                    match merged.last_mut() {
                        Some(last)
                            if seg.time.start - last.time.end < post.min_duration_off as f64 =>
                        {
                            last.time.end = seg.time.end;
                        }
                        _ => merged.push(seg),
                    }
                }
                segments.extend(merged);
            } else {
                segments.extend(temp);
            }
        }

        segments.sort_by(|a, b| {
            a.time
                .start
                .partial_cmp(&b.time.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        segments
    }
}

const HOP_LENGTH: usize = 160;

fn clip_turns_to_audio(turns: &mut Vec<SpeakerTurn>, n_samples: usize) {
    let dur = n_samples as f64 / SAMPLE_RATE as f64;
    for t in turns.iter_mut() {
        t.time.end = t.time.end.min(dur);
    }
    turns.retain(|t| t.time.end > t.time.start);
}

fn pad_or_slice_mel(mel: &[f32], n_frames: usize, feed_size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; feed_size * N_MELS];
    let copy_frames = n_frames.min(feed_size);
    if copy_frames > 0 {
        out[..copy_frames * N_MELS].copy_from_slice(&mel[..copy_frames * N_MELS]);
    }
    out
}

fn map_outputs(
    names: &[String],
    outputs: Vec<InferenceTensor>,
) -> Result<HashMap<String, InferenceTensor>, SortformerError> {
    if !names.is_empty() && names.len() == outputs.len() {
        return Ok(names.iter().cloned().zip(outputs).collect());
    }
    // Fallback: known positional order for the community export when names
    // are unavailable (mock runtimes). Prefer name-based when present.
    let mut map = HashMap::new();
    let mut it = outputs.into_iter();
    if let Some(preds) = it.next() {
        map.insert(OUT_PREDS.to_owned(), preds);
    }
    if let Some(embs) = it.next() {
        map.insert(OUT_EMBS.to_owned(), embs);
    }
    Ok(map)
}

fn preds_shape_frames(t: &InferenceTensor) -> Result<usize, SortformerError> {
    // [1, T, 4] or [T, 4]
    match t.shape.as_slice() {
        [_, frames, spk] if *spk == MAX_SPEAKERS => Ok(*frames),
        [frames, spk] if *spk == MAX_SPEAKERS => Ok(*frames),
        other => Err(SortformerError::Shape(format!(
            "unexpected preds shape {other:?}"
        ))),
    }
}

fn embs_shape_frames(t: &InferenceTensor) -> Result<usize, SortformerError> {
    match t.shape.as_slice() {
        [_, frames, dim] if *dim == EMB_DIM => Ok(*frames),
        [frames, dim] if *dim == EMB_DIM => Ok(*frames),
        other => Err(SortformerError::Shape(format!(
            "unexpected embs shape {other:?}"
        ))),
    }
}

fn get_log_pred_scores(preds: &[f32], n_frames: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; n_frames * MAX_SPEAKERS];
    for t in 0..n_frames {
        let mut log_1_probs_sum = 0.0f32;
        for s in 0..MAX_SPEAKERS {
            let p = preds[t * MAX_SPEAKERS + s].max(PRED_SCORE_THRESHOLD);
            log_1_probs_sum += (1.0 - p).max(PRED_SCORE_THRESHOLD).ln();
        }
        for s in 0..MAX_SPEAKERS {
            let p = preds[t * MAX_SPEAKERS + s].max(PRED_SCORE_THRESHOLD);
            let log_p = p.ln();
            let log_1_p = (1.0 - p).max(PRED_SCORE_THRESHOLD).ln();
            scores[t * MAX_SPEAKERS + s] = log_p - log_1_p + log_1_probs_sum - 0.5f32.ln();
        }
    }
    scores
}

fn disable_low_scores(
    preds: &[f32],
    mut scores: Vec<f32>,
    n_frames: usize,
    min_pos_scores_per_spk: usize,
) -> Vec<f32> {
    let mut pos_count = [0usize; MAX_SPEAKERS];
    for t in 0..n_frames {
        for s in 0..MAX_SPEAKERS {
            if scores[t * MAX_SPEAKERS + s] > 0.0 {
                pos_count[s] += 1;
            }
        }
    }
    for t in 0..n_frames {
        for s in 0..MAX_SPEAKERS {
            let is_speech = preds[t * MAX_SPEAKERS + s] > 0.5;
            if !is_speech {
                scores[t * MAX_SPEAKERS + s] = f32::NEG_INFINITY;
            } else {
                let is_pos = scores[t * MAX_SPEAKERS + s] > 0.0;
                if !is_pos && pos_count[s] >= min_pos_scores_per_spk {
                    scores[t * MAX_SPEAKERS + s] = f32::NEG_INFINITY;
                }
            }
        }
    }
    scores
}

fn boost_topk_scores(
    mut scores: Vec<f32>,
    n_frames: usize,
    n_boost_per_spk: usize,
    scale_factor: f32,
) -> Vec<f32> {
    for s in 0..MAX_SPEAKERS {
        let mut col: Vec<(usize, f32)> = (0..n_frames)
            .map(|t| (t, scores[t * MAX_SPEAKERS + s]))
            .collect();
        col.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for item in col.iter().take(n_boost_per_spk.min(col.len())) {
            let t = item.0;
            if scores[t * MAX_SPEAKERS + s] != f32::NEG_INFINITY {
                scores[t * MAX_SPEAKERS + s] -= scale_factor * 0.5f32.ln();
            }
        }
    }
    scores
}

fn get_topk_indices(
    scores: &[f32],
    n_frames: usize,
    n_frames_no_sil: usize,
    spkcache_len: usize,
) -> (Vec<usize>, Vec<bool>) {
    let mut flat: Vec<(usize, f32)> = Vec::with_capacity(n_frames * MAX_SPEAKERS);
    for s in 0..MAX_SPEAKERS {
        for t in 0..n_frames {
            flat.push((s * n_frames + t, scores[t * MAX_SPEAKERS + s]));
        }
    }
    flat.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut topk_flat: Vec<usize> = flat
        .iter()
        .take(spkcache_len)
        .map(|(idx, score)| {
            if *score == f32::NEG_INFINITY {
                MAX_INDEX
            } else {
                *idx
            }
        })
        .collect();
    while topk_flat.len() < spkcache_len {
        topk_flat.push(MAX_INDEX);
    }
    topk_flat.sort_unstable();

    let mut is_disabled = vec![false; spkcache_len];
    let mut frame_indices = vec![0usize; spkcache_len];
    for (i, &flat_idx) in topk_flat.iter().enumerate() {
        if flat_idx == MAX_INDEX {
            is_disabled[i] = true;
            frame_indices[i] = 0;
        } else {
            let frame_idx = flat_idx % n_frames;
            if frame_idx >= n_frames_no_sil {
                is_disabled[i] = true;
                frame_indices[i] = 0;
            } else {
                frame_indices[i] = frame_idx;
            }
        }
    }
    (frame_indices, is_disabled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::{InferenceError, InferenceTensor, NamedTensor, TensorData};

    /// Mock that returns zeros shaped like Sortformer outputs.
    struct ZeroMock {
        input_names: Vec<String>,
        output_names: Vec<String>,
        chunk_len: usize,
    }

    impl ZeroMock {
        fn new(chunk_len: usize) -> Self {
            Self {
                input_names: vec![
                    "chunk".into(),
                    "chunk_lengths".into(),
                    "spkcache".into(),
                    "spkcache_lengths".into(),
                    "fifo".into(),
                    "fifo_lengths".into(),
                ],
                output_names: vec![OUT_PREDS.into(), OUT_EMBS.into()],
                chunk_len,
            }
        }
    }

    impl InferenceRuntime for ZeroMock {
        fn input_names(&self) -> &[String] {
            &self.input_names
        }
        fn output_names(&self) -> &[String] {
            &self.output_names
        }
        fn run(
            &mut self,
            inputs: &[NamedTensor<'_>],
        ) -> Result<Vec<InferenceTensor>, InferenceError> {
            // Derive spkcache/fifo lengths from inputs so shapes stay consistent.
            let mut spk = 0usize;
            let mut fifo = 0usize;
            for nt in inputs {
                match nt.name {
                    "spkcache_lengths" => {
                        if let TensorData::I64(v) = &nt.tensor.data {
                            spk = *v.first().unwrap_or(&0) as usize;
                        }
                    }
                    "fifo_lengths" => {
                        if let TensorData::I64(v) = &nt.tensor.data {
                            fifo = *v.first().unwrap_or(&0) as usize;
                        }
                    }
                    _ => {}
                }
            }
            // Emit chunk_len model frames of predictions after prefix.
            let total = spk + fifo + self.chunk_len;
            let preds = InferenceTensor::f32(
                vec![1, total, MAX_SPEAKERS],
                vec![0.0; total * MAX_SPEAKERS],
            );
            let embs = InferenceTensor::f32(
                vec![1, self.chunk_len, EMB_DIM],
                vec![0.0; self.chunk_len * EMB_DIM],
            );
            Ok(vec![preds, embs])
        }
        fn run_ordered(
            &mut self,
            _inputs: &[&InferenceTensor],
        ) -> Result<Vec<InferenceTensor>, InferenceError> {
            Err(InferenceError::Run("not used".into()))
        }
    }

    #[test]
    fn mock_diarize_empty_audio() {
        let cfg = SortformerConfig::default();
        let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
        let turns = d.diarize(&[]).unwrap();
        assert!(turns.is_empty());
    }

    #[test]
    fn mock_diarize_silence_yields_no_turns() {
        // Zero mock → all probs 0 → no onsets.
        let cfg = SortformerConfig::default();
        let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
        let audio = vec![0.0f32; SAMPLE_RATE as usize]; // 1 s
        let turns = d.diarize(&audio).unwrap();
        assert!(turns.is_empty(), "silence must not produce speakers");
    }

    #[test]
    fn mock_chunk_preserves_state_across_calls() {
        let cfg = SortformerConfig {
            chunk_len: 4,
            fifo_len: 4,
            spkcache_len: 8,
            right_context: 1,
            ..Default::default()
        };
        let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
        // ~1 s of audio
        let audio = vec![0.01f32; SAMPLE_RATE as usize];
        let _ = d.diarize_chunk(&audio).unwrap();
        // After one chunk FIFO may have grown; second call should not panic.
        let _ = d.diarize_chunk(&audio).unwrap();
        assert!(d.fifo_frames <= d.config.fifo_len || d.spkcache_frames > 0);
    }

    #[test]
    fn from_runtime_rejects_via_config_validate() {
        let err = SortformerConfig::default()
            .with_max_speakers(8)
            .expect_err("cap");
        assert!(matches!(
            err,
            SortformerError::MaxSpeakersExceeded { requested: 8 }
        ));
    }
}
