#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]

//! pyo3 bindings for the polyvoice Pipeline v2.

use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{Pipeline as RustPipeline, PipelineError};
use polyvoice::types::{Profile, SampleRate};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python-facing `Pipeline` wrapper (v2).
#[pyclass]
pub struct Pipeline {
    pipeline: RustPipeline,
}

#[pymethods]
impl Pipeline {
    /// Build a Mobile-profile Pipeline.
    #[staticmethod]
    #[pyo3(signature = (models_cache=None))]
    fn mobile(models_cache: Option<&str>) -> PyResult<Self> {
        Self::build_profile(Profile::Mobile, models_cache)
    }

    /// Build a Balanced-profile Pipeline.
    #[staticmethod]
    #[pyo3(signature = (models_cache=None))]
    fn balanced(models_cache: Option<&str>) -> PyResult<Self> {
        Self::build_profile(Profile::Balanced, models_cache)
    }

    /// Run diarization on an iterable of f32 samples.
    fn run<'py>(
        &self,
        py: Python<'py>,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sr = SampleRate::new(sample_rate).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid sample rate {sample_rate} (expected 8000..=192000)"
            ))
        })?;

        let result = py.detach(|| self.pipeline.run(&samples, sr)).map_err(|e| match e {
            PipelineError::UnsupportedSampleRate { actual } => {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported sample rate {actual} (expected 16000)"
                ))
            }
            PipelineError::ModelLoad { .. } |
            PipelineError::Registry(_) |
            PipelineError::Segmentation(_) |
            PipelineError::Embedding(_) => {
                pyo3::exceptions::PyOSError::new_err(format!("model/inference error: {e}"))
            }
            _ => pyo3::exceptions::PyRuntimeError::new_err(format!("pipeline.run: {e}")),
        })?;

        let dict = PyDict::new(py);
        dict.set_item("num_speakers", result.num_speakers)?;

        let turns: Vec<_> = result
            .turns
            .iter()
            .map(|t| {
                let d = PyDict::new(py);
                d.set_item("start", t.time.start)?;
                d.set_item("end", t.time.end)?;
                d.set_item("speaker", t.speaker.0)?;
                Ok(d)
            })
            .collect::<PyResult<Vec<_>>>()?;
        dict.set_item("turns", turns)?;

        Ok(dict)
    }
}

impl Pipeline {
    fn build_profile(profile: Profile, models_cache: Option<&str>) -> PyResult<Self> {
        let registry = match models_cache {
            Some(path) => ModelRegistry::with_cache_dir(path),
            None => ModelRegistry::default(),
        }
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("model registry: {e}")))?;

        let pipeline = RustPipeline::builder()
            .profile(profile)
            .with_models_from(registry)
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("pipeline build: {e}"))
            })?;

        Ok(Self { pipeline })
    }
}

#[pymodule]
fn _polyvoice(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pipeline>()?;
    Ok(())
}
