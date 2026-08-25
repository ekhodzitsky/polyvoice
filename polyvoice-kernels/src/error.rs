//! Load / shape errors for the ResNet34 kernels.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error("failed to read {path}: {detail}")]
    Io { path: PathBuf, detail: String },

    #[error("not a usable polyvoice kernel model: {detail}")]
    Model { detail: String },

    #[error("weight {name} missing or has shape {got:?}, expected {expected:?}")]
    Weight {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("fbank must be T×{expected_mels} (got T={n_frames}, mels={n_mels})")]
    FbankShape {
        n_frames: usize,
        n_mels: usize,
        expected_mels: usize,
    },

    #[error("need at least 1 fbank frame")]
    EmptyFbank,

    #[error("waveform batch must be N×1×T (got n={n}, t={t}, len={len})")]
    WaveformShape { n: usize, t: usize, len: usize },

    #[error("waveform too short for SincNet: T={t} (need >= {min_t})")]
    WaveformTooShort { t: usize, min_t: usize },
}
