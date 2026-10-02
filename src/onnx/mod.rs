//! ONNX-file inference through tract, with a session pool.
//!
//! # Runtime boundary
//!
//! The tract backend lives in private `tract_session` (feature `backend-tract`).
//! Neural stages outside this module must depend only on [`InferenceRuntime`] /
//! [`RuntimeSession`] and must **not** import `tract_onnx` directly.
//!
//! This module compiles only with `infer`, which requires `backend-tract`.
//! The product CLI does not enable it. Select tract with env
//! `POLYVOICE_INFERENCE_BACKEND=tract` or [`InferenceBackend::force`].

use std::path::Path;

#[cfg(not(feature = "backend-tract"))]
compile_error!("feature `infer` requires `backend-tract`");

mod factory;
mod runtime;
#[cfg(feature = "backend-tract")]
mod tract_session;

pub use factory::{InferenceBackend, RuntimeSession};
pub use runtime::{InferenceError, InferenceRuntime, InferenceTensor, NamedTensor, TensorData};
#[cfg(feature = "backend-tract")]
pub use tract_session::TractSession;

pub use crate::types::execution_provider::ExecutionProvider;

/// Minimum plausible size for an ONNX file (header only).
pub const ONNX_MIN_HEADER_BYTES: usize = 64;

/// Build an inference session for `model_path`.
///
/// This is the one place embedding and segmentation sessions are constructed:
/// it validates the ONNX header before tract parses the file. Tract always
/// runs on CPU and ignores [`ExecutionProvider`]. `intra_threads` is accepted
/// for call-site parity and ignored by tract.
///
/// Callers must depend only on [`InferenceRuntime`], not on tract types.
pub fn build_session_with_ep(
    model_path: &Path,
    ep: ExecutionProvider,
    intra_threads: Option<usize>,
) -> Result<RuntimeSession, OnnxError> {
    RuntimeSession::from_path(model_path, ep, intra_threads)
}

/// Resolve the ONNX session-pool size for segmenter / embedder.
///
/// Order: `POLYVOICE_SESSION_POOL_SIZE` env (if positive) → `configured` → 1.
/// Used so operators can tune CPU fan-out without a rebuild; defaults stay
/// DER-identical (same math, different scheduling only).
pub fn resolve_session_pool_size(configured: usize) -> usize {
    std::env::var("POLYVOICE_SESSION_POOL_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(configured.max(1))
        .max(1)
}

/// Intra-op threads per pooled session: share cores across the pool so N
/// sessions do not request N×cores workers.
///
/// Override with `POLYVOICE_INTRA_THREADS` (positive integer) for host tuning.
pub fn resolve_intra_threads(pool_size: usize) -> usize {
    if let Some(n) = std::env::var("POLYVOICE_INTRA_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    let pool = pool_size.max(1);
    std::thread::available_parallelism()
        .map(|n| (n.get() / pool).max(1))
        .unwrap_or(1)
}

/// Read ONNX `metadata_props` (custom metadata key/value pairs) from `path`.
///
/// Tract-only builds validate the header and return an empty map — adapters
/// then take geometry from the manifest / defaults
/// (`models::metadata::load_model_config`).
///
/// Returns an empty map when the model has no custom props (not an error).
pub fn read_model_metadata_props(
    path: &Path,
) -> Result<std::collections::HashMap<String, String>, OnnxError> {
    validate_onnx_header(path)?;
    Ok(std::collections::HashMap::new())
}

/// Errors from ONNX session construction and model metadata reads.
///
/// Replaces `anyhow::Error` in this module's public constructors so callers
/// can classify load failures without substring matching. Backend error types
/// (tract errors) are not always `Send + Sync`, so their details are carried as
/// strings.
#[derive(Clone, thiserror::Error, Debug)]
pub enum OnnxError {
    /// Structural header validation failed before the backend parsed the file.
    #[error(transparent)]
    Validation(#[from] OnnxValidationError),

    /// The backend failed to build an inference session from the model file
    /// (protobuf parse, graph optimization, or EP wiring).
    #[error("failed to build inference session for {path}: {detail}")]
    SessionBuild {
        path: std::path::PathBuf,
        detail: String,
    },

    /// Reading custom `metadata_props` from the model failed.
    #[error("failed to read ONNX metadata_props: {detail}")]
    Metadata { detail: String },
}

/// Error raised when an ONNX file fails structural header validation.
#[derive(Clone, thiserror::Error, Debug)]
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
    fn resolve_session_pool_size_is_at_least_one() {
        // Ambient POLYVOICE_SESSION_POOL_SIZE may be set; still never zero.
        assert!(resolve_session_pool_size(0) >= 1);
        assert!(resolve_session_pool_size(4) >= 1);
    }

    #[test]
    fn resolve_intra_threads_is_at_least_one() {
        assert!(resolve_intra_threads(1) >= 1);
        assert!(resolve_intra_threads(4) >= 1);
    }

    #[test]
    fn execution_provider_auto_is_cpu() {
        let auto = ExecutionProvider::auto();
        assert_eq!(auto, ExecutionProvider::Cpu);
        let copied = auto;
        assert_eq!(copied, auto);
        assert!(!format!("{auto:?}").is_empty());
    }

    #[test]
    fn execution_provider_is_available_matches_wiring() {
        assert!(ExecutionProvider::Cpu.is_available());
        assert!(!ExecutionProvider::Nnapi.is_available());
        assert!(!ExecutionProvider::Cuda.is_available());
        // CoreMl / XnnPack were ort-only EPs; the core crate has no ort.
        assert!(!ExecutionProvider::CoreMl.is_available());
        assert!(!ExecutionProvider::XnnPack.is_available());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn read_model_metadata_props_real_model() {
        let path = std::path::Path::new("models/silero_vad.onnx");
        if !path.exists() {
            return;
        }
        // Silero carries no custom props; the read itself must succeed.
        let props = read_model_metadata_props(path).unwrap();
        assert!(props.keys().all(|k| !k.is_empty()));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn read_model_metadata_props_rejects_missing_file() {
        let err =
            read_model_metadata_props(std::path::Path::new("models/definitely_not_a_model.onnx"))
                .expect_err("missing file must fail validation");
        assert!(
            matches!(err, OnnxError::Validation(_)),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn read_model_metadata_props_rejects_garbage() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[0xAB; 64]).unwrap();
        let err =
            read_model_metadata_props(tmp.path()).expect_err("garbage must fail header validation");
        assert!(
            matches!(err, OnnxError::Validation(_)),
            "unexpected error: {err}"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn validate_onnx_header_missing_file() {
        let err = validate_onnx_header(std::path::Path::new("models/no_such_file.onnx"))
            .expect_err("missing file must fail");
        assert!(err.detail.contains("cannot read metadata"));
        assert!(err.to_string().contains("no_such_file.onnx"));
    }

    #[test]
    fn onnx_error_display_variants() {
        let build = OnnxError::SessionBuild {
            path: std::path::PathBuf::from("m.onnx"),
            detail: "parse failed".to_string(),
        };
        assert_eq!(
            build.to_string(),
            "failed to build inference session for m.onnx: parse failed"
        );
        let meta = OnnxError::Metadata {
            detail: "no meta".to_string(),
        };
        assert_eq!(
            meta.to_string(),
            "failed to read ONNX metadata_props: no meta"
        );
        let validation = OnnxError::Validation(OnnxValidationError {
            path: std::path::PathBuf::from("bad.onnx"),
            detail: "too small".to_string(),
        });
        assert_eq!(
            validation.to_string(),
            "ONNX header validation failed for bad.onnx: too small"
        );
        // Clone derive round-trips.
        let cloned = validation.clone();
        assert_eq!(cloned.to_string(), validation.to_string());
    }
}
