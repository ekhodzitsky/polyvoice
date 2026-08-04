//! Runtime-agnostic inference surface.
//!
//! Neural stages (Silero VAD, powerset segmenter, embedders) must depend only
//! on this module — never on a concrete ONNX Runtime binding. The default
//! implementation is ort via [`crate::onnx::RuntimeSession`]; a pure-Rust
//! tract backend implements the same trait behind the `backend-tract` feature.

use std::fmt;

/// Element storage for an inference tensor.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// Owned tensor passed into / returned from [`InferenceRuntime::run`].
///
/// Shape is row-major; `data` length must equal the product of `shape`
/// (empty shape = scalar, length 1).
#[derive(Clone, Debug, PartialEq)]
pub struct InferenceTensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

impl InferenceTensor {
    /// Build an `f32` tensor. Does not check that `data.len()` matches `shape`.
    pub fn f32(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Self {
        Self {
            shape: shape.into(),
            data: TensorData::F32(data),
        }
    }

    /// Build an `i64` tensor. Does not check that `data.len()` matches `shape`.
    pub fn i64(shape: impl Into<Vec<usize>>, data: Vec<i64>) -> Self {
        Self {
            shape: shape.into(),
            data: TensorData::I64(data),
        }
    }

    /// Scalar `i64` (0-d), matching ONNX scalar inputs such as Silero `sr`.
    pub fn i64_scalar(value: i64) -> Self {
        Self::i64(Vec::new(), vec![value])
    }

    /// Borrow `f32` data, or error if this tensor is not `f32`.
    pub fn as_f32_slice(&self) -> Result<&[f32], InferenceError> {
        match &self.data {
            TensorData::F32(v) => Ok(v.as_slice()),
            TensorData::I64(_) => Err(InferenceError::TypeMismatch {
                expected: "f32",
                actual: "i64",
            }),
        }
    }

    /// Take ownership of `f32` data, or error if this tensor is not `f32`.
    pub fn into_f32(self) -> Result<Vec<f32>, InferenceError> {
        match self.data {
            TensorData::F32(v) => Ok(v),
            TensorData::I64(_) => Err(InferenceError::TypeMismatch {
                expected: "f32",
                actual: "i64",
            }),
        }
    }
}

/// Named input binding for [`InferenceRuntime::run`].
#[derive(Clone, Debug)]
pub struct NamedTensor<'a> {
    pub name: &'a str,
    pub tensor: &'a InferenceTensor,
}

impl<'a> NamedTensor<'a> {
    pub fn new(name: &'a str, tensor: &'a InferenceTensor) -> Self {
        Self { name, tensor }
    }
}

/// Errors from the runtime-agnostic inference surface.
#[derive(Debug)]
pub enum InferenceError {
    /// Session load / builder failure (path, EP, etc.).
    Load(String),
    /// `run` failed inside the backend.
    Run(String),
    /// Output/input tensor had an unexpected element type.
    TypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    /// Model produced fewer outputs than the caller required.
    MissingOutput { index: usize, available: usize },
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(msg) => write!(f, "inference session load failed: {msg}"),
            Self::Run(msg) => write!(f, "inference run failed: {msg}"),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "tensor type mismatch: expected {expected}, got {actual}")
            }
            Self::MissingOutput { index, available } => {
                write!(
                    f,
                    "missing output index {index} (model produced {available} outputs)"
                )
            }
        }
    }
}

impl std::error::Error for InferenceError {}

/// Minimal pluggable inference session.
///
/// Surface is intentionally small: named / ordered tensor run, plus input
/// metadata for models that expose dynamic input names. Stateful models
/// (Silero LSTM) pass state as ordinary named I/O tensors — there is no
/// separate "stateful run" API.
///
/// Implementors must be [`Send`] so sessions can live in a shared object pool.
pub trait InferenceRuntime: Send {
    /// Model input names in declaration order.
    fn input_names(&self) -> &[String];

    /// Model output names in declaration order (matches [`Self::run`] vector order).
    /// Default is empty when a backend does not expose names.
    fn output_names(&self) -> &[String] {
        &[]
    }

    /// First input name, if any — convenience for single-input models.
    fn primary_input_name(&self) -> Option<&str> {
        self.input_names().first().map(String::as_str)
    }

    /// Run with named inputs (Silero, powerset, any multi-input graph).
    fn run(&mut self, inputs: &[NamedTensor<'_>]) -> Result<Vec<InferenceTensor>, InferenceError>;

    /// Run with positional inputs in model input order (embedders).
    fn run_ordered(
        &mut self,
        inputs: &[&InferenceTensor],
    ) -> Result<Vec<InferenceTensor>, InferenceError>;
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Cheap mock used to unit-test stages without a native runtime.
    pub struct MockRuntime {
        names: Vec<String>,
        /// When set, `run` / `run_ordered` return these outputs (cloned).
        pub outputs: Vec<InferenceTensor>,
        pub last_named: Vec<(String, InferenceTensor)>,
        pub last_ordered: Vec<InferenceTensor>,
    }

    impl MockRuntime {
        pub fn new(input_names: &[&str], outputs: Vec<InferenceTensor>) -> Self {
            Self {
                names: input_names.iter().map(|s| (*s).to_owned()).collect(),
                outputs,
                last_named: Vec::new(),
                last_ordered: Vec::new(),
            }
        }
    }

    impl InferenceRuntime for MockRuntime {
        fn input_names(&self) -> &[String] {
            &self.names
        }

        fn run(
            &mut self,
            inputs: &[NamedTensor<'_>],
        ) -> Result<Vec<InferenceTensor>, InferenceError> {
            self.last_named = inputs
                .iter()
                .map(|n| (n.name.to_owned(), n.tensor.clone()))
                .collect();
            Ok(self.outputs.clone())
        }

        fn run_ordered(
            &mut self,
            inputs: &[&InferenceTensor],
        ) -> Result<Vec<InferenceTensor>, InferenceError> {
            self.last_ordered = inputs.iter().map(|t| (*t).clone()).collect();
            Ok(self.outputs.clone())
        }
    }

    #[test]
    fn mock_runtime_round_trip_named() {
        let out = InferenceTensor::f32(vec![1], vec![0.9]);
        let mut rt = MockRuntime::new(&["input"], vec![out.clone()]);
        let inp = InferenceTensor::f32(vec![1, 4], vec![0.0; 4]);
        let got = rt
            .run(&[NamedTensor::new("input", &inp)])
            .expect("mock run");
        assert_eq!(got, vec![out]);
        assert_eq!(rt.last_named[0].0, "input");
    }

    #[test]
    fn tensor_f32_accessors() {
        let t = InferenceTensor::f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.as_f32_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        let i = InferenceTensor::i64_scalar(16_000);
        assert!(i.as_f32_slice().is_err());
    }

    #[test]
    fn tensor_into_f32() {
        let t = InferenceTensor::f32(vec![2], vec![1.0, 2.0]);
        assert_eq!(t.into_f32().unwrap(), vec![1.0, 2.0]);
        let i = InferenceTensor::i64_scalar(16_000);
        let err = i.into_f32().unwrap_err();
        assert!(err.to_string().contains("expected f32, got i64"));
    }

    #[test]
    fn tensor_type_mismatch_display() {
        let i = InferenceTensor::i64(vec![2], vec![1, 2]);
        let err = i.as_f32_slice().unwrap_err();
        assert_eq!(
            err.to_string(),
            "tensor type mismatch: expected f32, got i64"
        );
    }

    #[test]
    fn tensor_scalar_shape_is_empty() {
        let s = InferenceTensor::i64_scalar(8_000);
        assert!(s.shape.is_empty());
        assert_eq!(s.data, TensorData::I64(vec![8_000]));
    }

    #[test]
    fn inference_error_display() {
        let e = InferenceError::MissingOutput {
            index: 1,
            available: 1,
        };
        assert!(e.to_string().contains("missing output"));
    }

    #[test]
    fn inference_error_display_all_variants() {
        let load = InferenceError::Load("bad path".to_string());
        assert_eq!(load.to_string(), "inference session load failed: bad path");
        let run = InferenceError::Run("boom".to_string());
        assert_eq!(run.to_string(), "inference run failed: boom");
        let tm = InferenceError::TypeMismatch {
            expected: "f32",
            actual: "i64",
        };
        assert_eq!(
            tm.to_string(),
            "tensor type mismatch: expected f32, got i64"
        );
        let missing = InferenceError::MissingOutput {
            index: 2,
            available: 1,
        };
        assert_eq!(
            missing.to_string(),
            "missing output index 2 (model produced 1 outputs)"
        );
        // All variants implement std::error::Error.
        for e in [load, run, tm, missing] {
            let _: &dyn std::error::Error = &e;
            assert!(std::error::Error::source(&e).is_none());
        }
    }

    #[test]
    fn mock_runtime_round_trip_ordered() {
        let out = InferenceTensor::f32(vec![1], vec![0.5]);
        let mut rt = MockRuntime::new(&["feats"], vec![out.clone()]);
        let inp = InferenceTensor::f32(vec![1, 80], vec![0.0; 80]);
        let got = rt.run_ordered(&[&inp]).expect("mock run_ordered");
        assert_eq!(got, vec![out]);
        assert_eq!(rt.last_ordered.len(), 1);
        assert_eq!(rt.last_ordered[0], inp);
    }

    #[test]
    fn default_output_names_is_empty() {
        // MockRuntime does not override `output_names`, so the trait default
        // (empty slice) applies.
        let rt = MockRuntime::new(&["input"], vec![]);
        assert!(rt.output_names().is_empty());
    }

    #[test]
    fn primary_input_name_first_or_none() {
        let rt = MockRuntime::new(&["a", "b"], vec![]);
        assert_eq!(rt.primary_input_name(), Some("a"));
        let empty = MockRuntime::new(&[], vec![]);
        assert_eq!(empty.primary_input_name(), None);
    }
}
