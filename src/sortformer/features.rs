//! NeMo-compatible log-mel features for Streaming Sortformer.
//!
//! Geometry matches the community ONNX export used by parakeet-rs:
//! - `n_fft=512`, `win_length=400`, `hop_length=160`
//! - 128 Slaney-normalized mel bins
//! - pre-emphasis 0.97, log zero-guard `2^-24`
//! - center-padded STFT (n_fft/2 each side)
//!
//! This path is intentionally separate from [`crate::features::FbankExtractor`]
//! (WeSpeaker/ECAPA geometry) so embedding models keep their existing fbank.

use super::config::{N_MELS, SAMPLE_RATE, SortformerError};
use realfft::RealFftPlanner;
use std::f32::consts::PI;

const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8; // 2^-24

/// Cached mel filterbank + FFT plan for Sortformer feature extraction.
pub struct SortformerFeatures {
    mel_basis: Vec<Vec<f32>>, // [n_mels][freq_bins]
    window: Vec<f32>,         // zero-padded Hann of length N_FFT
}

impl SortformerFeatures {
    pub fn new() -> Self {
        let hann = hann_window(WIN_LENGTH);
        let win_offset = (N_FFT - WIN_LENGTH) / 2;
        let mut window = vec![0.0f32; N_FFT];
        window[win_offset..win_offset + WIN_LENGTH].copy_from_slice(&hann);
        Self {
            mel_basis: create_mel_filterbank_slaney(N_FFT, N_MELS, SAMPLE_RATE as usize),
            window,
        }
    }

    /// Extract log-mel features as a flat row-major buffer shaped
    /// `[1, time, N_MELS]` (batch axis implicit).
    ///
    /// Returns `(flat_data, time_frames)`.
    pub fn extract_log_mel(&self, audio: &[f32]) -> Result<(Vec<f32>, usize), SortformerError> {
        if audio.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let pre = apply_preemphasis(audio, PREEMPH);
        let power = stft_power(&pre, &self.window)?;
        let freq_bins = N_FFT / 2 + 1;
        let n_frames = power.len() / freq_bins;
        // mel: (n_mels, n_frames) then transpose to (time, n_mels)
        let mut flat = vec![0.0f32; n_frames * N_MELS];
        for t in 0..n_frames {
            let frame = &power[t * freq_bins..(t + 1) * freq_bins];
            for m in 0..N_MELS {
                let sum: f32 = self.mel_basis[m]
                    .iter()
                    .zip(frame.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                flat[t * N_MELS + m] = (sum + LOG_ZERO_GUARD).ln();
            }
        }
        Ok((flat, n_frames))
    }
}

impl Default for SortformerFeatures {
    fn default() -> Self {
        Self::new()
    }
}

fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(audio.len());
    out.push(audio[0]);
    for i in 1..audio.len() {
        out.push(audio[i] - coef * audio[i - 1]);
    }
    out
}

/// Periodic Hann window of length `n` (librosa fftbins=True: divide by N).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / n as f32).cos())
        .collect()
}

/// Power spectrogram, row-major `[frame][freq]` flattened.
fn stft_power(audio: &[f32], window: &[f32]) -> Result<Vec<f32>, SortformerError> {
    let pad = N_FFT / 2;
    let mut padded = vec![0.0f32; pad];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad, 0.0);

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(N_FFT);
    let freq_bins = N_FFT / 2 + 1;
    let num_frames = if padded.len() >= N_FFT {
        (padded.len() - N_FFT) / HOP_LENGTH + 1
    } else {
        0
    };

    let mut spectrogram = vec![0.0f32; num_frames * freq_bins];
    let mut input = vec![0.0f32; N_FFT];
    let mut output = r2c.make_output_vec();
    let mut scratch = r2c.make_scratch_vec();

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_LENGTH;
        for i in 0..N_FFT {
            input[i] = padded[start + i] * window[i];
        }
        r2c.process_with_scratch(&mut input, &mut output, &mut scratch)
            .map_err(|e| SortformerError::Features(format!("FFT failed: {e}")))?;
        let base = frame_idx * freq_bins;
        for k in 0..freq_bins {
            spectrogram[base + k] = output[k].norm_sqr();
        }
    }
    Ok(spectrogram)
}

// Slaney mel scale (librosa default).
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.068_751_777_420_949_12;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

fn create_mel_filterbank_slaney(n_fft: usize, n_mels: usize, sample_rate: usize) -> Vec<Vec<f32>> {
    let freq_bins = n_fft / 2 + 1;
    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(fmax);
    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();
    let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();

    let mut filterbank = vec![vec![0.0f32; freq_bins]; n_mels];
    for i in 0..n_mels {
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - freq) / fdiff[i + 1];
            filterbank[i][k] = 0.0f64.max(lower.min(upper)) as f32;
        }
        let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
        for slot in &mut filterbank[i] {
            *slot *= enorm as f32;
        }
    }
    filterbank
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_audio_yields_empty_features() {
        let f = SortformerFeatures::new();
        let (data, n) = f.extract_log_mel(&[]).unwrap();
        assert!(data.is_empty());
        assert_eq!(n, 0);
    }

    #[test]
    fn one_second_silence_has_expected_shape() {
        let f = SortformerFeatures::new();
        let audio = vec![0.0f32; SAMPLE_RATE as usize];
        let (data, n_frames) = f.extract_log_mel(&audio).unwrap();
        assert!(n_frames > 0);
        assert_eq!(data.len(), n_frames * N_MELS);
        // Log of guard value is finite.
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stft_concentrates_power_near_1khz() {
        // 1 kHz sine @ 16 kHz for 1 s → bin 1000*512/16000 = 32.
        let n = SAMPLE_RATE as usize;
        let audio: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let hann = hann_window(WIN_LENGTH);
        let win_offset = (N_FFT - WIN_LENGTH) / 2;
        let mut window = vec![0.0f32; N_FFT];
        window[win_offset..win_offset + WIN_LENGTH].copy_from_slice(&hann);
        let power = stft_power(&audio, &window).unwrap();
        let freq_bins = N_FFT / 2 + 1;
        let num_frames = power.len() / freq_bins;
        let expected = 32usize;
        let mut hits = 0;
        for t in 2..num_frames.saturating_sub(2) {
            let frame = &power[t * freq_bins..(t + 1) * freq_bins];
            let max_bin = frame
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            if max_bin == expected {
                hits += 1;
            }
        }
        let interior = num_frames.saturating_sub(4);
        assert!(
            hits > interior / 2,
            "expected bin {expected} dominant, hits={hits}/{interior}"
        );
    }
}
