#![allow(deprecated)] // legacy embedding API; see polyvoice::embedder
//! ONNX-based speaker embedding extractor with a session pool.

use crate::embedding::{EmbeddingError, EmbeddingExtractor};
use crate::types::DiarizationConfig;
use crate::utils::l2_normalize;
use std::path::Path;

/// Minimum plausible size for an ONNX file (header only).
pub const ONNX_MIN_HEADER_BYTES: usize = 64;

/// Which ONNX Runtime execution provider to request for a session.
///
/// Canonical home is here (the module that owns session creation) so the
/// low-level constructors can name it without depending on `pipeline_v2`;
/// `pipeline_v2::config` re-exports it, so existing imports keep compiling.
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

/// Build an `ort` session for `model_path` with the requested execution
/// provider. This is the ONE place embedding/segmentation sessions are
/// constructed: it validates the ONNX header BEFORE ort ever parses the file
/// (the validate-before-build invariant), then registers the EP.
///
/// `intra_threads`: `Some(n)` pins the session's intra-op thread count (the
/// fbank embedder uses 1 because it parallelises across a session pool).
///
/// EP behavior: `Cpu` registers nothing. `CoreMl` registers CoreML when the
/// build carries the `coreml` feature on macOS aarch64, else warns and runs on
/// CPU. `Nnapi`/`Cuda`/`XnnPack` are not wired yet — they warn and run on CPU.
/// EP registration failure is deliberately not an error: ort's built-in CPU
/// fallback keeps inference correct.
pub fn build_session_with_ep(
    model_path: &Path,
    ep: ExecutionProvider,
    intra_threads: Option<usize>,
) -> anyhow::Result<ort::session::Session> {
    validate_onnx_header(model_path)?;
    // ort::Error is not Send+Sync, so it cannot ride `?` into anyhow — stringify.
    let mut builder =
        ort::session::Session::builder().map_err(|e| anyhow::anyhow!("session builder: {e}"))?;
    if let Some(n) = intra_threads {
        builder = builder
            .with_intra_threads(n)
            .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?;
    }
    match ep {
        ExecutionProvider::Cpu => {}
        ExecutionProvider::CoreMl => {
            #[cfg(all(feature = "coreml", target_os = "macos", target_arch = "aarch64"))]
            {
                let coreml = ort::execution_providers::CoreMLExecutionProvider::default();
                builder = builder
                    .with_execution_providers([coreml.build()])
                    .map_err(|e| anyhow::anyhow!("coreml ep: {e}"))?;
            }
            #[cfg(not(all(feature = "coreml", target_os = "macos", target_arch = "aarch64")))]
            tracing::warn!(
                "execution provider CoreMl is not compiled in (needs the `coreml` feature on \
                 macOS aarch64) — falling back to CPU"
            );
        }
        ExecutionProvider::XnnPack => {
            #[cfg(feature = "xnnpack")]
            {
                let xnnpack = ort::execution_providers::XNNPACKExecutionProvider::default();
                builder = builder
                    .with_execution_providers([xnnpack.build()])
                    .map_err(|e| anyhow::anyhow!("xnnpack ep: {e}"))?;
            }
            #[cfg(not(feature = "xnnpack"))]
            tracing::warn!(
                "execution provider XnnPack is not compiled in (needs the `xnnpack` feature) —                  falling back to CPU"
            );
        }
        ExecutionProvider::Nnapi | ExecutionProvider::Cuda => {
            tracing::warn!("execution provider {ep:?} is not wired yet — falling back to CPU");
        }
    }
    builder
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("commit_from_file: {e}"))
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
/// This is intentionally lightweight — it runs **before** any `ort::Session`
/// creation so that garbage or truncated files never reach the C++ ONNX
/// Runtime parser (mitigates DOS-003).
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
/// Wraps `ort::session::Session` in a [`crossbeam_queue::ArrayQueue`]
/// so that multiple threads can extract embeddings concurrently without lock contention.
#[cfg(feature = "onnx")]
#[deprecated(
    since = "0.7.0",
    note = "use the v1.0 Embedder trait in polyvoice::embedder"
)]
pub struct OnnxEmbeddingExtractor {
    pool: crossbeam_queue::ArrayQueue<ort::session::Session>,
    embedding_dim: usize,
    window_samples: usize,
}

#[cfg(feature = "onnx")]
impl OnnxEmbeddingExtractor {
    /// { pool_size > 0 }
    /// `fn new(model_path: &Path, embedding_dim: usize, window_samples: usize, pool_size: usize, ep: ExecutionProvider) -> Result<Self, anyhow::Error>`
    /// { ret.pool.len() == pool_size }
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
        let pool = crossbeam_queue::ArrayQueue::new(pool_size);
        for i in 0..pool_size {
            let session = build_session_with_ep(model_path, ep, None)
                .map_err(|e| EmbeddingError::InferenceFailed(format!("session {i}: {e}")))?;
            pool.push(session)
                .map_err(|_| anyhow::anyhow!("failed to push session into pool"))?;
        }
        Ok(Self {
            pool,
            embedding_dim,
            window_samples,
        })
    }

    fn checkout(&self) -> Option<PooledSession<'_>> {
        self.pool.pop().map(|s| PooledSession {
            session: Some(s),
            pool: &self.pool,
        })
    }
}

#[cfg(feature = "onnx")]
impl EmbeddingExtractor for OnnxEmbeddingExtractor {
    fn extract(
        &self,
        samples: &[f32],
        _config: &DiarizationConfig,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let mut guard = self.checkout().ok_or_else(|| {
            EmbeddingError::InferenceFailed("ONNX session pool exhausted".to_string())
        })?;

        if samples.len() != self.window_samples {
            return Err(EmbeddingError::InvalidInput {
                expected: self.window_samples,
                got: samples.len(),
            });
        }

        let input_tensor =
            ort::value::TensorRef::from_array_view(([1_usize, self.window_samples], samples))
                .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        let session = guard
            .session
            .as_mut()
            .ok_or_else(|| EmbeddingError::InferenceFailed("session not available".to_string()))?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        if outputs.iter().next().is_none() {
            return Err(EmbeddingError::InferenceFailed(
                "ONNX model produced no outputs".to_string(),
            ));
        }
        let (_, data) = &outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))?;

        let data_len = data.len();
        if data_len != self.embedding_dim {
            return Err(EmbeddingError::InferenceFailed(format!(
                "expected embedding dim {}, got {}",
                self.embedding_dim, data_len
            )));
        }
        let mut embedding = vec![0.0f32; self.embedding_dim];
        embedding.copy_from_slice(data);
        l2_normalize(&mut embedding);

        Ok(embedding)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(feature = "onnx")]
struct PooledSession<'a> {
    session: Option<ort::session::Session>,
    pool: &'a crossbeam_queue::ArrayQueue<ort::session::Session>,
}

#[cfg(feature = "onnx")]
impl Drop for PooledSession<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            let _ = self.pool.push(session);
        }
    }
}

/// Stub when the `onnx` feature is disabled.
#[cfg(not(feature = "onnx"))]
#[deprecated(
    since = "0.7.0",
    note = "use the v1.0 Embedder trait in polyvoice::embedder"
)]
pub struct OnnxEmbeddingExtractor;

#[cfg(not(feature = "onnx"))]
impl OnnxEmbeddingExtractor {
    /// { false } // Always fails because onnx feature is disabled.
    /// `fn new(_model_path: &Path, _embedding_dim: usize, _window_samples: usize, _pool_size: usize) -> Result<Self, anyhow::Error>`
    /// { false }
    pub fn new(
        _model_path: &Path,
        _embedding_dim: usize,
        _window_samples: usize,
        _pool_size: usize,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("the `onnx` feature is not enabled")
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
        assert!(build_session_with_ep(path, ExecutionProvider::Cpu, None).is_ok());
        assert!(build_session_with_ep(path, ExecutionProvider::Cpu, Some(1)).is_ok());
        // Unwired providers warn and fall back to CPU — never panic or error.
        assert!(build_session_with_ep(path, ExecutionProvider::Cuda, None).is_ok());
        assert!(build_session_with_ep(path, ExecutionProvider::auto(), None).is_ok());
    }
}
