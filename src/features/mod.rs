//! Log-mel filterbank (fbank) feature extraction for speaker embeddings.
//!
//! Typical parameters for ECAPA-TDNN (16 kHz):
//! - `n_fft = 512`
//! - `win_length = 400` (25 ms)
//! - `hop_length = 160` (10 ms)
//! - `n_mels = 80`
//! - `f_min = 20.0`, `f_max = 7600.0`
//! - `pre_emphasis = 0.97`

use realfft::RealFftPlanner;
use thiserror::Error;

/// Error during fbank computation.
#[derive(Error, Debug, Clone)]
pub enum FbankError {
    #[error("fft failed: {0}")]
    Fft(String),
    #[error("invalid shape: {0}")]
    Shape(String),
    #[error("invalid fbank config: {0}")]
    InvalidConfig(String),
}

/// Configuration for log-mel filterbank extraction.
#[derive(Debug, Clone, Copy)]
pub struct FbankConfig {
    /// Expected sample rate in Hz.
    pub sample_rate: u32,
    /// FFT size.
    pub n_fft: usize,
    /// Window length in samples.
    pub win_length: usize,
    /// Hop length in samples.
    pub hop_length: usize,
    /// Number of mel bins.
    pub n_mels: usize,
    /// Lowest frequency (Hz).
    pub f_min: f32,
    /// Highest frequency (Hz).
    pub f_max: f32,
    /// Pre-emphasis coefficient.
    pub pre_emphasis: f32,
}

impl Default for FbankConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 80,
            f_min: 20.0,
            f_max: 7600.0,
            pre_emphasis: 0.97,
        }
    }
}

impl FbankConfig {
    /// { true }
    /// `pub fn validate(&self) -> Result<(), FbankError>`
    /// { true }
    /// Check the config for the constraints [`FbankExtractor`] relies on.
    ///
    /// Returns the first violation found as [`FbankError::InvalidConfig`].
    pub fn validate(&self) -> Result<(), FbankError> {
        if self.sample_rate == 0 {
            return Err(FbankError::InvalidConfig(
                "FbankConfig::sample_rate must be > 0".to_string(),
            ));
        }
        if self.n_fft == 0 {
            return Err(FbankError::InvalidConfig(
                "FbankConfig::n_fft must be > 0".to_string(),
            ));
        }
        if self.win_length == 0 {
            return Err(FbankError::InvalidConfig(
                "FbankConfig::win_length must be > 0".to_string(),
            ));
        }
        if self.hop_length == 0 {
            return Err(FbankError::InvalidConfig(
                "FbankConfig::hop_length must be > 0".to_string(),
            ));
        }
        if self.win_length > self.n_fft {
            return Err(FbankError::InvalidConfig(format!(
                "FbankConfig::win_length ({}) must be <= n_fft ({})",
                self.win_length, self.n_fft
            )));
        }
        if self.n_mels == 0 {
            return Err(FbankError::InvalidConfig(
                "FbankConfig::n_mels must be > 0".to_string(),
            ));
        }
        if !self.f_min.is_finite() || self.f_min < 0.0 {
            return Err(FbankError::InvalidConfig(
                "FbankConfig::f_min must be finite and non-negative".to_string(),
            ));
        }
        if !self.f_max.is_finite() || self.f_max <= self.f_min {
            return Err(FbankError::InvalidConfig(format!(
                "FbankConfig::f_max must be finite and greater than f_min ({})",
                self.f_min
            )));
        }
        Ok(())
    }
}

fn pre_emphasis(samples: &[f32], coeff: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    if let Some(&first) = samples.first() {
        out.push(first);
        for i in 1..samples.len() {
            out.push(samples[i] - coeff * samples[i - 1]);
        }
    }
    out
}

fn frame(samples: &[f32], win_length: usize, hop_length: usize) -> Vec<Vec<f32>> {
    let num_frames = if samples.len() >= win_length {
        1 + (samples.len() - win_length) / hop_length
    } else {
        0
    };
    let mut frames = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * hop_length;
        frames.push(samples[start..start + win_length].to_vec());
    }
    frames
}

fn hamming_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos())
        .collect()
}

/// Cached log-mel filterbank extractor.
///
/// Reuses the FFT planner, window, and mel-filterbank matrices across calls,
/// eliminating per-call allocation overhead.
pub struct FbankExtractor {
    pub config: FbankConfig,
    r2c: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
}

impl FbankExtractor {
    /// { true }
    /// pub fn new(config: FbankConfig) -> Self
    /// { true }
    /// Create a cached fbank extractor.
    ///
    /// The FFT planner, Hamming window, and mel-filterbank matrices are computed
    /// once and reused across subsequent [`extract`](Self::extract) calls.
    ///
    /// ```rust
    /// use polyvoice::features::{FbankExtractor, FbankConfig};
    /// let config = FbankConfig::default();
    /// let extractor = FbankExtractor::new(config);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `config` is invalid (see [`FbankConfig::validate`]).
    /// Use [`try_new`](Self::try_new) for a fallible alternative.
    #[allow(clippy::panic)] // Documented convenience over `try_new`.
    pub fn new(config: FbankConfig) -> Self {
        match Self::try_new(config) {
            Ok(extractor) => extractor,
            Err(e) => panic!("{e}"),
        }
    }

    /// { true }
    /// `pub fn try_new(config: FbankConfig) -> Result<Self, FbankError>`
    /// { true }
    /// Fallible constructor: validates `config` via [`FbankConfig::validate`]
    /// and returns [`FbankError::InvalidConfig`] instead of panicking.
    pub fn try_new(config: FbankConfig) -> Result<Self, FbankError> {
        config.validate()?;
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(config.n_fft);
        let window = hamming_window(config.win_length);
        let mel_filters = mel_filterbank(
            config.n_fft,
            config.n_mels,
            config.sample_rate,
            config.f_min,
            config.f_max,
        );
        Ok(Self {
            config,
            r2c,
            window,
            mel_filters,
        })
    }

    /// { true }
    /// `pub fn extract(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>, FbankError>`
    /// { ret.as_ref().map_or(true, |v| v.iter().all(|f| f.len() == self.config.n_mels)) }
    /// Extract log-mel filterbank features from audio samples.
    ///
    /// Returns an empty vector if `samples` is shorter than the window length.
    ///
    /// ```rust
    /// use polyvoice::features::{FbankExtractor, FbankConfig};
    /// let config = FbankConfig::default();
    /// let extractor = FbankExtractor::new(config);
    /// let samples = vec![0.0f32; 16000 * 2]; // 2 seconds @ 16 kHz
    /// let fb = extractor.extract(&samples).unwrap();
    /// assert!(!fb.is_empty());
    /// assert!(fb.iter().all(|f| f.len() == config.n_mels));
    /// ```
    pub fn extract(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>, FbankError> {
        if samples.len() < self.config.win_length {
            return Ok(Vec::new());
        }

        let pre = pre_emphasis(samples, self.config.pre_emphasis);
        let frames = frame(&pre, self.config.win_length, self.config.hop_length);
        let mut spectrum = self.r2c.make_output_vec();
        let mut melspec = Vec::with_capacity(frames.len());
        let spectrum_len = spectrum.len();

        for fr in frames {
            let mut buf = vec![0.0f32; self.config.n_fft];
            for (i, &v) in fr.iter().enumerate() {
                buf[i] = v * self.window[i];
            }

            if buf.len() != self.config.n_fft {
                return Err(FbankError::Shape(format!(
                    "buffer len {} != n_fft {}",
                    buf.len(),
                    self.config.n_fft
                )));
            }
            if spectrum.len() != spectrum_len {
                return Err(FbankError::Shape(
                    "spectrum buffer resized unexpectedly".to_string(),
                ));
            }

            self.r2c
                .process(&mut buf, &mut spectrum)
                .map_err(|e| FbankError::Fft(e.to_string()))?;

            let power: Vec<f32> = spectrum.iter().map(|c| c.norm_sqr()).collect();

            let mut mel = vec![0.0f32; self.config.n_mels];
            for (i, filter) in self.mel_filters.iter().enumerate() {
                let sum = filter
                    .iter()
                    .zip(power.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                mel[i] = sum.max(1e-10).ln();
            }
            melspec.push(mel);
        }

        Ok(melspec)
    }
}

/// { true }
/// `pub fn apply_cmvn(frames: &[Vec<f32>]) -> Vec<Vec<f32>>`
/// { ret.len() == frames.len() }
/// Apply cepstral mean normalization (CMN) to fbank features.
///
/// Subtracts the per-bin mean across all frames. Required by WeSpeaker
/// models to normalize channel effects.
pub fn apply_cmvn(frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if frames.is_empty() {
        return Vec::new();
    }
    let n_bins = frames[0].len();
    let n_frames = frames.len() as f32;

    let mut means = vec![0.0f32; n_bins];
    for frame in frames {
        for (i, &v) in frame.iter().enumerate() {
            means[i] += v;
        }
    }
    for m in &mut means {
        *m /= n_frames;
    }

    frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .zip(means.iter())
                .map(|(&v, &m)| v - m)
                .collect()
        })
        .collect()
}

fn mel_filterbank(
    n_fft: usize,
    n_mels: usize,
    sample_rate: u32,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    let fft_freqs: Vec<f32> = (0..=n_fft / 2)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let mut filters = vec![vec![0.0f32; fft_freqs.len()]; n_mels];
    for (i, filter) in filters.iter_mut().enumerate() {
        let f_left = hz_points[i];
        let f_center = hz_points[i + 1];
        let f_right = hz_points[i + 2];
        for (j, &freq) in fft_freqs.iter().enumerate() {
            if freq >= f_left && freq <= f_center {
                let denom = f_center - f_left;
                if denom > 0.0 {
                    filter[j] = (freq - f_left) / denom;
                }
            } else if freq > f_center && freq <= f_right {
                let denom = f_right - f_center;
                if denom > 0.0 {
                    filter[j] = (f_right - freq) / denom;
                }
            }
        }
    }
    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_fbank_shape() {
        let config = FbankConfig::default();
        let samples = vec![0.0f32; 16000 * 2]; // 2 seconds
        let extractor = FbankExtractor::new(config);
        let fb = extractor.extract(&samples).unwrap();
        assert!(!fb.is_empty());
        assert!(fb.iter().all(|f| f.len() == config.n_mels));
    }

    #[test]
    fn test_fbank_short_audio() {
        let config = FbankConfig::default();
        let samples = vec![0.0f32; 100]; // less than win_length
        let extractor = FbankExtractor::new(config);
        let fb = extractor.extract(&samples).unwrap();
        assert!(fb.is_empty());
    }

    #[test]
    fn test_pre_emphasis() {
        let samples = vec![1.0f32, 2.0, 3.0];
        let pre = pre_emphasis(&samples, 0.97);
        assert!((pre[1] - (2.0 - 0.97 * 1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_hamming_window_sum() {
        let w = hamming_window(400);
        let sum: f32 = w.iter().sum();
        // Hamming window sum is approximately 200 (half of length * 0.5 average? No, average ~0.5)
        assert!(sum > 150.0 && sum < 250.0);
    }

    #[test]
    fn test_apply_cmvn() {
        let frames = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 4.0, 5.0],
            vec![5.0, 6.0, 7.0],
        ];
        let normalized = apply_cmvn(&frames);
        assert_eq!(normalized.len(), 3);
        assert!((normalized[0][0] - (-2.0)).abs() < 1e-5);
        assert!((normalized[1][0] - 0.0).abs() < 1e-5);
        assert!((normalized[2][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_cmvn_empty() {
        let frames: Vec<Vec<f32>> = vec![];
        let normalized = apply_cmvn(&frames);
        assert!(normalized.is_empty());
    }

    #[test]
    #[should_panic(expected = "FbankConfig::hop_length must be > 0")]
    fn fbank_extractor_rejects_zero_hop_length() {
        let config = FbankConfig {
            hop_length: 0,
            ..Default::default()
        };
        let _ = FbankExtractor::new(config);
    }

    #[test]
    #[should_panic(expected = "FbankConfig::win_length (500) must be <= n_fft (400)")]
    fn fbank_extractor_rejects_win_longer_than_n_fft() {
        let config = FbankConfig {
            n_fft: 400,
            win_length: 500,
            ..Default::default()
        };
        let _ = FbankExtractor::new(config);
    }

    #[test]
    fn fbank_config_validate_accepts_default() {
        assert!(FbankConfig::default().validate().is_ok());
    }

    #[test]
    fn fbank_config_validate_reports_first_violation() {
        let config = FbankConfig {
            hop_length: 0,
            n_mels: 0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        match err {
            FbankError::InvalidConfig(detail) => {
                assert!(
                    detail.contains("hop_length"),
                    "first violation is hop_length, got: {detail}"
                );
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn fbank_extractor_try_new_rejects_invalid_config() {
        let config = FbankConfig {
            sample_rate: 0,
            ..Default::default()
        };
        match FbankExtractor::try_new(config) {
            Ok(_) => panic!("expected Err for sample_rate == 0"),
            Err(e) => assert!(matches!(e, FbankError::InvalidConfig(_))),
        }
        assert!(FbankExtractor::try_new(FbankConfig::default()).is_ok());
    }

    #[test]
    fn fbank_config_validate_rejects_each_field() {
        let cases: Vec<FbankConfig> = vec![
            FbankConfig {
                n_fft: 0,
                ..Default::default()
            },
            FbankConfig {
                win_length: 0,
                ..Default::default()
            },
            FbankConfig {
                n_mels: 0,
                ..Default::default()
            },
            FbankConfig {
                f_min: f32::NAN,
                ..Default::default()
            },
            FbankConfig {
                f_min: -1.0,
                ..Default::default()
            },
            FbankConfig {
                f_max: f32::NAN,
                ..Default::default()
            },
            FbankConfig {
                f_min: 100.0,
                f_max: 100.0,
                ..Default::default()
            },
            FbankConfig {
                f_min: 200.0,
                f_max: 100.0,
                ..Default::default()
            },
        ];
        for c in cases {
            assert!(
                matches!(c.validate(), Err(FbankError::InvalidConfig(_))),
                "expected InvalidConfig for {c:?}"
            );
        }
    }

    #[test]
    fn fbank_error_display_messages() {
        assert_eq!(
            FbankError::Fft("boom".into()).to_string(),
            "fft failed: boom"
        );
        assert_eq!(
            FbankError::Shape("bad".into()).to_string(),
            "invalid shape: bad"
        );
        assert_eq!(
            FbankError::InvalidConfig("nope".into()).to_string(),
            "invalid fbank config: nope"
        );
    }

    #[test]
    fn test_pre_emphasis_empty() {
        assert!(pre_emphasis(&[], 0.97).is_empty());
    }

    #[test]
    fn frame_exact_window_multiple() {
        let samples = vec![1.0f32; 800];
        let frames = frame(&samples, 400, 160);
        assert_eq!(frames.len(), 1 + (800 - 400) / 160);
        assert!(frames.iter().all(|f| f.len() == 400));
        // Exactly one window's worth of samples yields exactly one frame.
        let frames = frame(&samples[..400], 400, 160);
        assert_eq!(frames.len(), 1);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn fbank_extract_sine_yields_finite_frames() {
        let config = FbankConfig::default();
        let extractor = FbankExtractor::new(config);
        let sr = config.sample_rate as f32;
        let samples: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin() * 0.5)
            .collect();
        let fb = extractor.extract(&samples).unwrap();
        assert_eq!(
            fb.len(),
            1 + (16000 - config.win_length) / config.hop_length
        );
        assert!(fb.iter().all(|frame| frame.iter().all(|v| v.is_finite())));
    }
}
