//! Online (streaming) speaker diarization.

use crate::cluster::SpeakerCluster;
use crate::embedding::EmbeddingExtractor;
use crate::types::{DiarizationConfig, Segment, SpeakerId, TimeRange, WordAlignment};

/// Per-connection state for streaming diarization.
pub struct OnlineDiarizer {
    config: DiarizationConfig,
    cluster: SpeakerCluster,
    /// Accumulated raw samples since last extraction.
    audio_buffer: Vec<f32>,
    /// Speakers extracted per hop, aligned to audio_buffer position.
    embedding_buffer: Vec<(usize, SpeakerId, f32)>, // (end_sample, speaker, confidence)
    /// Current best-guess speaker for the latest audio.
    current_speaker: Option<SpeakerId>,
    /// Running timestamp offset in samples.
    total_samples: usize,
}

impl OnlineDiarizer {
    /// Create a new streaming diarizer.
    ///
    /// ```rust
    /// use polyvoice::{OnlineDiarizer, DiarizationConfig};
    /// let diarizer = OnlineDiarizer::new(DiarizationConfig::default());
    /// assert_eq!(diarizer.num_speakers(), 0);
    /// ```
    pub fn new(config: DiarizationConfig) -> Self {
        Self {
            cluster: SpeakerCluster::new(config),
            audio_buffer: Vec::new(),
            embedding_buffer: Vec::new(),
            current_speaker: None,
            total_samples: 0,
            config,
        }
    }

    /// Feed audio samples into the streaming diarizer.
    ///
    /// Returns newly completed segments whenever enough audio has accumulated
    /// for a full analysis window.
    ///
    /// ```rust
    /// use polyvoice::{OnlineDiarizer, DiarizationConfig, DummyExtractor};
    /// let mut diarizer = OnlineDiarizer::new(DiarizationConfig::default());
    /// let extractor = DummyExtractor::new(256);
    /// let samples = vec![0.0f32; 16000]; // 1 second
    /// let segments = diarizer.feed(&samples, &extractor).unwrap();
    /// // With default 1.5s window, 1s may not produce a segment yet.
    /// ```
    pub fn feed<E: EmbeddingExtractor>(
        &mut self,
        samples: &[f32],
        extractor: &E,
    ) -> Result<Vec<Segment>, crate::embedding::EmbeddingError> {
        self.audio_buffer.extend_from_slice(samples);
        let mut new_segments = Vec::new();
        let window = self.config.window_samples();
        let hop = self.config.hop_samples();

        while self.audio_buffer.len() >= window {
            let window_samples: Vec<f32> = self.audio_buffer[..window].to_vec();
            let embedding = extractor.extract(&window_samples, &self.config)?;
            let (speaker, confidence) = self.cluster.assign(&embedding);
            self.current_speaker = Some(speaker);

            let segment_end = self.total_samples + window;
            self.embedding_buffer.push((segment_end, speaker, confidence));

            // Drain hop samples from the front of the buffer (sliding window).
            self.audio_buffer.drain(..hop);
            self.total_samples += hop;

            new_segments.push(Segment {
                time: TimeRange {
                    start: (self.total_samples.saturating_sub(window) as f64)
                        / self.config.sample_rate.get() as f64,
                    end: (self.total_samples as f64) / self.config.sample_rate.get() as f64,
                },
                speaker: Some(speaker),
                confidence: Some(confidence),
            });
        }

        Ok(new_segments)
    }

    /// Align transcript words to speakers based on diarization state.
    ///
    /// Each word's `speaker` field is updated to the most likely speaker
    /// at the word's midpoint timestamp.
    ///
    /// ```rust
    /// use polyvoice::{OnlineDiarizer, DiarizationConfig, DummyExtractor, WordAlignment, TimeRange};
    /// let mut diarizer = OnlineDiarizer::new(DiarizationConfig::default());
    /// let extractor = DummyExtractor::new(256);
    /// let samples = vec![0.0f32; 16000 * 3];
    /// let _ = diarizer.feed(&samples, &extractor).unwrap();
    ///
    /// let mut words = vec![
    ///     WordAlignment { word: "hello".into(), time: TimeRange { start: 0.5, end: 1.0 }, speaker: None, confidence: 0.0 },
    /// ];
    /// diarizer.align_words(&mut words);
    /// assert!(words[0].speaker.is_some());
    /// ```
    pub fn align_words(&self, words: &mut [WordAlignment]) {
        for word in words.iter_mut() {
            let mid_sample = ((word.time.start + word.time.end) / 2.0
                * self.config.sample_rate.get() as f64) as usize;

            // Find the first window whose end_sample >= mid_sample.
            let speaker = self
                .embedding_buffer
                .iter()
                .find(|(end, _, _)| *end >= mid_sample)
                .map(|(_, spk, _)| *spk)
                .or(self.current_speaker);

            word.speaker = speaker;
        }
    }

    /// Return the most recently assigned speaker, if any.
    ///
    /// ```rust
    /// use polyvoice::{OnlineDiarizer, DiarizationConfig};
    /// let diarizer = OnlineDiarizer::new(DiarizationConfig::default());
    /// assert!(diarizer.current_speaker().is_none());
    /// ```
    pub fn current_speaker(&self) -> Option<SpeakerId> {
        self.current_speaker
    }

    /// Return the number of distinct speakers detected so far.
    ///
    /// ```rust
    /// use polyvoice::{OnlineDiarizer, DiarizationConfig};
    /// let diarizer = OnlineDiarizer::new(DiarizationConfig::default());
    /// assert_eq!(diarizer.num_speakers(), 0);
    /// ```
    pub fn num_speakers(&self) -> usize {
        self.cluster.num_speakers()
    }

    /// Flush any remaining audio and return the final segment.
    ///
    /// Pads the trailing audio with zeros to fill the analysis window.
    /// Returns `Ok(None)` if the buffer is empty.
    ///
    /// ```rust
    /// use polyvoice::{OnlineDiarizer, DiarizationConfig, DummyExtractor};
    /// let mut diarizer = OnlineDiarizer::new(DiarizationConfig::default());
    /// let extractor = DummyExtractor::new(256);
    /// let samples = vec![0.0f32; 16000];
    /// let _ = diarizer.feed(&samples, &extractor).unwrap();
    /// let final_seg = diarizer.flush(&extractor).unwrap();
    /// // May be Some or None depending on buffer state.
    /// ```
    pub fn flush<E: EmbeddingExtractor>(
        &mut self,
        extractor: &E,
    ) -> Result<Option<Segment>, crate::embedding::EmbeddingError> {
        if self.audio_buffer.is_empty() {
            return Ok(None);
        }
        let window = self.config.window_samples();
        let mut padded = vec![0.0f32; window];
        let copy_len = self.audio_buffer.len().min(window);
        padded[..copy_len].copy_from_slice(&self.audio_buffer[..copy_len]);
        let embedding = extractor.extract(&padded, &self.config)?;
        let (speaker, confidence) = self.cluster.assign(&embedding);
        self.current_speaker = Some(speaker);
        self.total_samples += self.audio_buffer.len();
        self.embedding_buffer.push((self.total_samples, speaker, confidence));
        self.audio_buffer.clear();

        Ok(Some(Segment {
            time: TimeRange {
                start: (self.total_samples.saturating_sub(window) as f64)
                    / self.config.sample_rate.get() as f64,
                end: (self.total_samples as f64) / self.config.sample_rate.get() as f64,
            },
            speaker: Some(speaker),
            confidence: Some(confidence),
        }))
    }
}
