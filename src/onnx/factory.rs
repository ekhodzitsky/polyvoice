//! Backend selection and unified session type for [`InferenceRuntime`].
//!
//! Default is always ort. The pure-Rust tract backend is available only when
//! the `backend-tract` feature is enabled and selected via
//! [`InferenceBackend`] / env `POLYVOICE_INFERENCE_BACKEND=tract`.

use super::ExecutionProvider;
use super::OnnxError;
use super::ort_session::OrtSession;
use super::runtime::{InferenceError, InferenceRuntime, InferenceTensor, NamedTensor};
use std::cell::Cell;
use std::path::Path;

const BACKEND_AUTO: u8 = 0;
const BACKEND_ORT: u8 = 1;
#[cfg(feature = "backend-tract")]
const BACKEND_TRACT: u8 = 2;

// Thread-local override for tests / programmatic selection.
// BACKEND_AUTO means "read env / default to ort".
// Thread-local so parallel tests cannot race each other.
thread_local! {
    static BACKEND_FORCE: Cell<u8> = const { Cell::new(BACKEND_AUTO) };
}

/// Which concrete inference backend to construct.
///
/// EP (`ExecutionProvider`) remains ort-only configuration. tract ignores EP
/// and always runs on pure-Rust CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceBackend {
    /// ONNX Runtime via the `ort` crate (default).
    Ort,
    /// Pure-Rust tract (requires the `backend-tract` cargo feature).
    #[cfg(feature = "backend-tract")]
    Tract,
}

impl InferenceBackend {
    /// Resolve the backend: forced override → env → ort default.
    ///
    /// Env var: `POLYVOICE_INFERENCE_BACKEND` = `ort` | `tract` (case-insensitive).
    /// Unknown values and missing feature fall back to ort with a warning.
    pub fn resolve() -> Self {
        let forced = BACKEND_FORCE.with(Cell::get);
        match forced {
            BACKEND_ORT => return Self::Ort,
            #[cfg(feature = "backend-tract")]
            BACKEND_TRACT => return Self::Tract,
            _ => {}
        }
        match std::env::var("POLYVOICE_INFERENCE_BACKEND") {
            Ok(v) => {
                let lower = v.to_ascii_lowercase();
                match lower.as_str() {
                    "ort" | "onnxruntime" | "onnx-runtime" => Self::Ort,
                    "tract" => {
                        #[cfg(feature = "backend-tract")]
                        {
                            Self::Tract
                        }
                        #[cfg(not(feature = "backend-tract"))]
                        {
                            tracing::warn!(
                                "POLYVOICE_INFERENCE_BACKEND=tract but the `backend-tract` \
                                 feature is not enabled — falling back to ort"
                            );
                            Self::Ort
                        }
                    }
                    other => {
                        tracing::warn!("unknown POLYVOICE_INFERENCE_BACKEND={other:?}; using ort");
                        Self::Ort
                    }
                }
            }
            Err(_) => Self::Ort,
        }
    }

    /// Force a backend for subsequent [`build_session_with_ep`](super::build_session_with_ep)
    /// calls on **this thread**. Pass `None` to clear the override (env / default apply again).
    ///
    /// Intended for tests and spike harnesses — not a stable multi-tenant API.
    /// Thread-local so parallel test threads do not interfere.
    pub fn force(backend: Option<Self>) {
        let code = match backend {
            None => BACKEND_AUTO,
            Some(Self::Ort) => BACKEND_ORT,
            #[cfg(feature = "backend-tract")]
            Some(Self::Tract) => BACKEND_TRACT,
        };
        BACKEND_FORCE.with(|c| c.set(code));
    }
}

/// Concrete session holding either ort or (optionally) tract.
///
/// Stages store this type and only call [`InferenceRuntime`] methods so the
/// default remains ort without stage code branching on backend.
#[derive(Debug)]
pub enum RuntimeSession {
    Ort(OrtSession),
    #[cfg(feature = "backend-tract")]
    Tract(super::tract_session::TractSession),
}

impl RuntimeSession {
    /// Build a session for `model_path` using the resolved [`InferenceBackend`].
    ///
    /// Validates the ONNX header before the backend parses the file (both
    /// paths). `ep` and `intra_threads` apply to ort; tract ignores EP.
    pub fn from_path(
        model_path: &Path,
        ep: ExecutionProvider,
        intra_threads: Option<usize>,
    ) -> Result<Self, OnnxError> {
        match InferenceBackend::resolve() {
            InferenceBackend::Ort => Ok(Self::Ort(OrtSession::from_path(
                model_path,
                ep,
                intra_threads,
            )?)),
            #[cfg(feature = "backend-tract")]
            InferenceBackend::Tract => Ok(Self::Tract(
                super::tract_session::TractSession::from_path(model_path, intra_threads)?,
            )),
        }
    }

    /// Which backend this session is using.
    pub fn backend(&self) -> InferenceBackend {
        match self {
            Self::Ort(_) => InferenceBackend::Ort,
            #[cfg(feature = "backend-tract")]
            Self::Tract(_) => InferenceBackend::Tract,
        }
    }
}

impl InferenceRuntime for RuntimeSession {
    fn input_names(&self) -> &[String] {
        match self {
            Self::Ort(s) => s.input_names(),
            #[cfg(feature = "backend-tract")]
            Self::Tract(s) => s.input_names(),
        }
    }

    fn run(&mut self, inputs: &[NamedTensor<'_>]) -> Result<Vec<InferenceTensor>, InferenceError> {
        match self {
            Self::Ort(s) => s.run(inputs),
            #[cfg(feature = "backend-tract")]
            Self::Tract(s) => s.run(inputs),
        }
    }

    fn run_ordered(
        &mut self,
        inputs: &[&InferenceTensor],
    ) -> Result<Vec<InferenceTensor>, InferenceError> {
        match self {
            Self::Ort(s) => s.run_ordered(inputs),
            #[cfg(feature = "backend-tract")]
            Self::Tract(s) => s.run_ordered(inputs),
        }
    }
}
