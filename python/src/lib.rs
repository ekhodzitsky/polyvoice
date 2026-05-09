//! pyo3 bindings for the legacy v0.5.2 Pipeline.

use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::Pipeline as RustPipeline;
use polyvoice::types::{DiarizationConfig, Profile, SampleRate};
use polyvoice::vad::VadConfig;
use polyvoice::FbankOnnxExtractor;
use polyvoice::SileroVad;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python-facing `Pipeline` wrapper.
#[pyclass]
pub struct Pipeline {
    pipeline: RustPipeline,
    extractor: FbankOnnxExtractor,
    vad: SileroVad,
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
        &mut self,
        py: Python<'py>,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let _sr = SampleRate::new(sample_rate)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!(
                "invalid sample rate {sample_rate} (expected 8000..=192000)"
            )))?;
        let result = self
            .pipeline
            .run(&samples, &self.extractor, &mut self.vad)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("pipeline.run: {e}")))?;
        let dict = PyDict::new(py);
        dict.set_item("num_speakers", result.num_speakers)?;
        let turns: Vec<_> = result
            .turns
            .iter()
            .map(|t| {
                let d = PyDict::new(py);
                d.set_item("start", t.time.start).unwrap();
                d.set_item("end", t.time.end).unwrap();
                d.set_item("speaker", t.speaker.0).unwrap();
                d
            })
            .collect();
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
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("registry: {e}")))?;
        let models = registry
            .ensure_for_profile(profile)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("models: {e}")))?;
        let extractor = FbankOnnxExtractor::new(
            &models.embedder_path,
            profile.embedding_dim(),
            1,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("extractor: {e}")))?;
        let vad = SileroVad::new(&models.segmenter_path, 512)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vad: {e}")))?;
        let pipeline = RustPipeline::new(DiarizationConfig::default(), VadConfig::default());
        Ok(Self {
            pipeline,
            extractor,
            vad,
        })
    }
}

#[pymodule]
fn _polyvoice(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pipeline>()?;
    Ok(())
}
