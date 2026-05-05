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

/// { samples.len() >= config.win_length }
/// `fn compute_fbank(samples: &[f32], config: &FbankConfig) -> Result<Vec<Vec<f32>>, FbankError>`
/// { ret.iter().all(|f| f.len() == config.n_mels) }
pub fn compute_fbank(
    samples: &[f32],
    config: &FbankConfig,
) -> Result<Vec<Vec<f32>>, FbankError> {
    if samples.len() < config.win_length {
        return Ok(Vec::new());
    }

    let pre = pre_emphasis(samples, config.pre_emphasis);
    let frames = frame(&pre, config.win_length, config.hop_length);
    let window = hamming_window(config.win_length);
    let mel_filters = mel_filterbank(config.n_fft, config.n_mels, config.sample_rate, config.f_min, config.f_max);

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(config.n_fft);
    let mut spectrum = r2c.make_output_vec();

    let mut melspec = Vec::with_capacity(frames.len());
    let spectrum_len = spectrum.len();

    for fr in frames {
        let mut buf = vec![0.0f32; config.n_fft];
        for (i, &v) in fr.iter().enumerate() {
            buf[i] = v * window[i];
        }

        if buf.len() != config.n_fft {
            return Err(FbankError::Shape(format!("buffer len {} != n_fft {}", buf.len(), config.n_fft)));
        }
        if spectrum.len() != spectrum_len {
            return Err(FbankError::Shape("spectrum buffer resized unexpectedly".to_string()));
        }

        r2c.process(&mut buf, &mut spectrum)
            .map_err(|e| FbankError::Fft(e.to_string()))?;

        let mut power = vec![0.0f32; config.n_fft / 2 + 1];
        for (i, c) in spectrum.iter().enumerate() {
            power[i] = c.norm_sqr();
        }

        let mut mel = vec![0.0f32; config.n_mels];
        for (i, filter) in mel_filters.iter().enumerate() {
            let sum = filter.iter().zip(power.iter()).map(|(a, b)| a * b).sum::<f32>();
            mel[i] = sum.max(1e-10).ln();
        }
        melspec.push(mel);
    }

    Ok(melspec)
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
        .map(|i| {
            0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos()
        })
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
    /// { config.n_fft > 0 && config.n_mels > 0 }
    /// `fn new(config: FbankConfig) -> Self`
    /// { ret.config == config }
    pub fn new(config: FbankConfig) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(config.n_fft);
        let window = hamming_window(config.win_length);
        let mel_filters =
            mel_filterbank(config.n_fft, config.n_mels, config.sample_rate, config.f_min, config.f_max);
        Self {
            config,
            r2c,
            window,
            mel_filters,
        }
    }

    /// { samples.len() >= self.config.win_length || ret.is_empty() }
    /// `fn extract(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>, FbankError>`
    /// { ret.iter().all(|f| f.len() == self.config.n_mels) }
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

            let mut power = vec![0.0f32; self.config.n_fft / 2 + 1];
            for (i, c) in spectrum.iter().enumerate() {
                power[i] = c.norm_sqr();
            }

            let mut mel = vec![0.0f32; self.config.n_mels];
            for (i, filter) in self.mel_filters.iter().enumerate() {
                let sum = filter.iter().zip(power.iter()).map(|(a, b)| a * b).sum::<f32>();
                mel[i] = sum.max(1e-10).ln();
            }
            melspec.push(mel);
        }

        Ok(melspec)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fbank_shape() {
        let config = FbankConfig::default();
        let samples = vec![0.0f32; 16000 * 2]; // 2 seconds
        let fb = compute_fbank(&samples, &config).unwrap();
        assert!(!fb.is_empty());
        assert!(fb.iter().all(|f| f.len() == config.n_mels));
    }

    #[test]
    fn test_fbank_short_audio() {
        let config = FbankConfig::default();
        let samples = vec![0.0f32; 100]; // less than win_length
        let fb = compute_fbank(&samples, &config).unwrap();
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
}
