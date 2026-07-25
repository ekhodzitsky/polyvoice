#![allow(deprecated)] // legacy embedding API; see polyvoice::embedder
//! ONNX-based speaker embedding extractor with a session pool.
//!
//! # Runtime boundary
//!
//! All `ort::` imports live in the private `ort_session` module. The optional
//! tract backend lives in private `tract_session` (feature `backend-tract`).
//! Neural stages outside this module must depend only on [`InferenceRuntime`] /
//! [`RuntimeSession`] and must **not** import `ort::` or `tract_onnx` directly.
//!
//! Default backend is always ort. Select tract with env
//! `POLYVOICE_INFERENCE_BACKEND=tract` (requires `backend-tract`) or
//! [`InferenceBackend::force`].

use crate::embedding::{EmbeddingError, EmbeddingExtractor};
use crate::types::DiarizationConfig;
use crate::utils::l2_normalize;
use std::path::Path;

mod factory;
mod ort_session;
#[cfg(all(test, feature = "backend-tract"))]
mod parity;
mod runtime;
#[cfg(feature = "backend-tract")]
mod tract_session;

pub use factory::{InferenceBackend, RuntimeSession};
pub use ort_session::OrtSession;
pub use runtime::{InferenceError, InferenceRuntime, InferenceTensor, NamedTensor, TensorData};
#[cfg(feature = "backend-tract")]
pub use tract_session::TractSession;

/// Minimum plausible size for an ONNX file (header only).
pub const ONNX_MIN_HEADER_BYTES: usize = 64;

/// Which ONNX Runtime execution provider to request for a session.
///
/// Canonical home is here (the module that owns session creation) so the
/// low-level constructors can name it without depending on `pipeline_v2`;
/// `pipeline_v2::config` re-exports it, so existing imports keep compiling.
///
/// EP is **ort-specific config** — it is not part of [`InferenceRuntime`].
/// Stages pass it only at session construction via [`build_session_with_ep`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExecutionProvider {
    Cpu,
    CoreMl,
    Nnapi,
    Cuda,
    XnnPack,
}

impl ExecutionProvider {
    /// Best default for the current target: CoreML on Apple Silicon, XNNPACK on
    /// aarch64 Linux, plain CPU elsewhere. Unwired providers fall back to CPU
    /// with a warning at session-build time.
    pub fn auto() -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        return Self::CoreMl;
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        return Self::XnnPack;
        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(target_os = "linux", target_arch = "aarch64"),
        )))]
        return Self::Cpu;
    }
}

/// Build an inference session for `model_path` with the requested execution
/// provider. This is the ONE place embedding/segmentation sessions are
/// constructed: it validates the ONNX header BEFORE the backend ever parses
/// the file (the validate-before-build invariant), then registers the EP.
///
/// Returns [`RuntimeSession`] — ort by default, or tract when the
/// `backend-tract` feature is enabled and selected via
/// [`InferenceBackend`] / `POLYVOICE_INFERENCE_BACKEND=tract`. Callers must
/// depend only on [`InferenceRuntime`], not on underlying `ort` / tract types.
///
/// `intra_threads`: `Some(n)` pins the session's intra-op thread count for ort
/// (the fbank embedder uses 1 because it parallelises across a session pool).
/// Ignored by tract.
///
/// EP behavior (ort only): `Cpu` registers nothing. `CoreMl` registers CoreML
/// when the build carries the `coreml` feature on macOS aarch64, else warns
/// and runs on CPU. `Nnapi`/`Cuda`/`XnnPack` are not wired yet — they warn
/// and run on CPU. EP registration failure is deliberately not an error:
/// ort's built-in CPU fallback keeps inference correct. tract always uses
/// pure-Rust CPU and ignores EP.
pub fn build_session_with_ep(
    model_path: &Path,
    ep: ExecutionProvider,
    intra_threads: Option<usize>,
) -> anyhow::Result<RuntimeSession> {
    RuntimeSession::from_path(model_path, ep, intra_threads)
}

/// Read ONNX `metadata_props` (custom metadata key/value pairs) from `path`.
///
/// Opens a short-lived CPU session solely to query model metadata, then drops
/// it. Used by `models::metadata::load_model_config` so stage adapters can take
/// geometry / license / adapter_type from the model itself (sherpa-onnx pattern).
///
/// Returns an empty map when the model has no custom props (not an error).
pub fn read_model_metadata_props(
    path: &Path,
) -> Result<std::collections::HashMap<String, String>, String> {
    let session = OrtSession::from_path(path, ExecutionProvider::Cpu, Some(1))
        .map_err(|e| format!("open for metadata: {e}"))?;
    session.custom_metadata_props()
}

/// Error raised when an ONNX file fails structural header validation.
#[derive(thiserror::Error, Debug)]
#[error("ONNX header validation failed for {path}: {detail}")]
pub struct OnnxValidationError {
    pub path: std::path::PathBuf,
    pub detail: String,
}

/// { true }
/// `pub fn validate_onnx_header(path: &Path) -> Result<(), OnnxValidationError>`
/// { true }
/// Validate that `path` points to a file with a plausible ONNX header.
///
/// Checks (in order):
/// 1. File exists and is at least [`ONNX_MIN_HEADER_BYTES`] bytes.
/// 2. The first 64 bytes can be read.
/// 3. Either:
///    - The first 16 bytes contain the ASCII substring `"ONNX"`, **or**
///    - The first byte is `0x08` (protobuf tag for field 1, wire-type varint),
///      indicating a valid ONNX ModelProto protobuf header.
///
/// This is intentionally lightweight — it runs **before** any runtime session
/// creation so that garbage or truncated files never reach the backend parser
/// (mitigates DOS-003).
pub fn validate_onnx_header(path: &Path) -> Result<(), OnnxValidationError> {
    let metadata = std::fs::metadata(path).map_err(|e| OnnxValidationError {
        path: path.to_path_buf(),
        detail: format!("cannot read metadata: {e}"),
    })?;

    if metadata.len() < ONNX_MIN_HEADER_BYTES as u64 {
        return Err(OnnxValidationError {
            path: path.to_path_buf(),
            detail: format!(
                "file too small ({} bytes, need at least {ONNX_MIN_HEADER_BYTES})",
                metadata.len()
            ),
        });
    }

    let mut file = std::fs::File::open(path).map_err(|e| OnnxValidationError {
        path: path.to_path_buf(),
        detail: format!("cannot open file: {e}"),
    })?;

    let mut header = [0u8; ONNX_MIN_HEADER_BYTES];
    let n = std::io::Read::read(&mut file, &mut header).map_err(|e| OnnxValidationError {
        path: path.to_path_buf(),
        detail: format!("cannot read header: {e}"),
    })?;

    if n < ONNX_MIN_HEADER_BYTES {
        return Err(OnnxValidationError {
            path: path.to_path_buf(),
            detail: format!("short read ({n} bytes, need at least {ONNX_MIN_HEADER_BYTES})"),
        });
    }

    // Check 1: "ONNX" magic in the first 16 bytes.
    let has_onnx_magic = header[..16].windows(4).any(|w| w == b"ONNX");

    // Check 2: plausible protobuf header for ONNX ModelProto.
    // Field 1 = ir_version, wire type 0 (varint) → tag byte 0x08.
    let has_protobuf_header = header[0] == 0x08;

    if !has_onnx_magic && !has_protobuf_header {
        return Err(OnnxValidationError {
            path: path.to_path_buf(),
            detail: "ONNX magic bytes not found and file does not start with a valid ONNX protobuf header".to_string(),
        });
    }

    Ok(())
}

/// A pooled ONNX session for speaker embedding extraction.
///
/// Wraps [`RuntimeSession`] in a blocking object pool so concurrent extractors
/// can reuse sessions (checkout waits; Drop returns).
/// Raw-waveform pooled ONNX embedder (no fbank). **Unused** by production
/// adapters — prefer [`crate::ecapa::FbankOnnxExtractor`] or
/// [`crate::embedder::ResNet34Adapter`].
#[deprecated(
    since = "0.7.0",
    note = "unused by production paths; use embedder::ResNet34Adapter / ecapa::FbankOnnxExtractor (Embedder)"
)]
pub struct OnnxEmbeddingExtractor {
    pool: crate::utils::ObjectPool<RuntimeSession>,
    embedding_dim: usize,
    window_samples: usize,
}

impl OnnxEmbeddingExtractor {
    /// { pool_size > 0 }
    /// `fn new(model_path: &Path, embedding_dim: usize, window_samples: usize, pool_size: usize, ep: ExecutionProvider) -> Result<Self, anyhow::Error>`
    /// { true }
    pub fn new(
        model_path: &Path,
        embedding_dim: usize,
        window_samples: usize,
        pool_size: usize,
        ep: ExecutionProvider,
    ) -> anyhow::Result<Self> {
        if pool_size == 0 {
            anyhow::bail!("pool_size must be > 0");
        }
        let mut sessions = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            let session = build_session_with_ep(model_path, ep, None)
                .map_err(|e| EmbeddingError::InferenceFailed(format!("session {i}: {e}")))?;
            sessions.push(session);
        }
        Ok(Self {
            pool: crate::utils::ObjectPool::new(sessions),
            embedding_dim,
            window_samples,
        })
    }
}

impl EmbeddingExtractor for OnnxEmbeddingExtractor {
    fn extract(
        &self,
        samples: &[f32],
        _config: &DiarizationConfig,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let mut session = self.pool.checkout();

        if samples.len() != self.window_samples {
            return Err(EmbeddingError::InvalidInput {
                expected: self.window_samples,
                got: samples.len(),
            });
        }

        let input = InferenceTensor::f32(vec![1, self.window_samples], samples.to_vec());
        let outputs = session
            .run_ordered(&[&input])
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        let first = outputs.into_iter().next().ok_or_else(|| {
            EmbeddingError::InferenceFailed("ONNX model produced no outputs".to_string())
        })?;
        let data = first
            .into_f32()
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        let data_len = data.len();
        if data_len != self.embedding_dim {
            return Err(EmbeddingError::InferenceFailed(format!(
                "expected embedding dim {}, got {}",
                self.embedding_dim, data_len
            )));
        }
        let mut embedding = data;
        l2_normalize(&mut embedding);

        Ok(embedding)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn valid_onnx_file_passes_validation() {
        let path = std::path::Path::new("models/silero_vad.onnx");
        if !path.exists() {
            // Skip if model is missing (e.g. CI without models).
            return;
        }
        assert!(validate_onnx_header(path).is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn random_64_bytes_fails_validation() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[0xAB; 64]).unwrap();
        let result = validate_onnx_header(tmp.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ONNX magic") || msg.contains("protobuf header"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn empty_file_fails_validation() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let result = validate_onnx_header(tmp.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("too small"),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn file_with_onnx_magic_passes() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut data = vec![0u8; 64];
        data[4..8].copy_from_slice(b"ONNX");
        tmp.write_all(&data).unwrap();
        assert!(validate_onnx_header(tmp.path()).is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn file_with_protobuf_header_passes() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        let mut data = vec![0u8; 64];
        data[0] = 0x08; // protobuf tag for field 1, varint
        data[1] = 0x08; // ir_version = 8
        tmp.write_all(&data).unwrap();
        assert!(validate_onnx_header(tmp.path()).is_ok());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn build_session_with_ep_rejects_garbage_before_ort() {
        // Validation must run first: garbage never reaches the ort parser.
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[0xAB; 64]).unwrap();
        let err = build_session_with_ep(tmp.path(), ExecutionProvider::Cpu, None)
            .expect_err("garbage must fail header validation");
        assert!(err.to_string().contains("ONNX header validation failed"));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn build_session_with_ep_cpu_and_unwired_ep_build_ok() {
        let path = std::path::Path::new("models/silero_vad.onnx");
        if !path.exists() {
            // Skip if the model is missing (e.g. CI without models).
            return;
        }
        // Pin ort: silero does not load on tract today, and env/force must not
        // flip this smoke test off the default backend.
        InferenceBackend::force(Some(InferenceBackend::Ort));
        let built = build_session_with_ep(path, ExecutionProvider::Cpu, None);
        assert!(
            built.is_ok(),
            "ort session build failed: {:?}",
            built.err().map(|e| e.to_string())
        );
        assert!(build_session_with_ep(path, ExecutionProvider::Cpu, Some(1)).is_ok());
        // Unwired providers warn and fall back to CPU — never panic or error.
        assert!(build_session_with_ep(path, ExecutionProvider::Cuda, None).is_ok());
        assert!(build_session_with_ep(path, ExecutionProvider::auto(), None).is_ok());
        InferenceBackend::force(None);
    }
}
