#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]

//! pyo3 bindings for the polyvoice Pipeline v2.

use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{Pipeline as RustPipeline, PipelineError};
use polyvoice::types::{DiarizationResult as RustDiarizationResult, Profile, SampleRate};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Typed diarization result — the canonical `DiarizationResult` v1, projectable
/// to JSON/RTTM/SRT/VTT/TXT with field-for-field parity with the CLI.
#[pyclass]
pub struct DiarizationResult {
    inner: RustDiarizationResult,
}

#[pymethods]
impl DiarizationResult {
    /// Parse a canonical diarization-result-v1 JSON document.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner: RustDiarizationResult = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid diarization-result JSON: {e}"
            ))
        })?;
        Ok(Self { inner })
    }

    /// Serialize to canonical JSON (same shape as `polyvoice diarize --format json`).
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("to_json: {e}")))
    }

    /// Render as RTTM. `file_id` fills the RTTM file-id column; RTTM is
    /// whitespace-delimited, so the id must be non-empty and whitespace-free.
    #[pyo3(signature = (file_id="audio"))]
    fn to_rttm(&self, file_id: &str) -> PyResult<String> {
        if file_id.is_empty() || file_id.contains(char::is_whitespace) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "file_id must be non-empty and contain no whitespace (RTTM is \
                 whitespace-delimited), got {file_id:?}"
            )));
        }
        let mut buf = Vec::new();
        polyvoice::rttm::write_rttm(&mut buf, file_id, &self.inner.turns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("to_rttm: {e}")))?;
        utf8(buf)
    }

    /// Render as SubRip (SRT) subtitles.
    fn to_srt(&self) -> PyResult<String> {
        let mut buf = Vec::new();
        polyvoice::format::write_srt(&mut buf, &self.inner.turns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("to_srt: {e}")))?;
        utf8(buf)
    }

    /// Render as WebVTT (`<v SPEAKER_NN>text</v>` voice spans when text is present).
    fn to_vtt(&self) -> PyResult<String> {
        let mut buf = Vec::new();
        polyvoice::format::write_vtt(&mut buf, &self.inner.turns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("to_vtt: {e}")))?;
        utf8(buf)
    }

    /// Render as a readable transcript (`[start - end] SPEAKER_NN: text`).
    fn to_txt(&self) -> PyResult<String> {
        let mut buf = Vec::new();
        polyvoice::format::write_txt(&mut buf, &self.inner.turns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("to_txt: {e}")))?;
        utf8(buf)
    }

    /// The full canonical result as a plain dict (equivalent to `json.loads(self.to_json())`).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json = self.to_json()?;
        py.import("json")?.call_method1("loads", (json,))
    }

    #[getter]
    fn num_speakers(&self) -> usize {
        self.inner.num_speakers
    }

    #[getter]
    fn schema_version(&self) -> &str {
        &self.inner.schema_version
    }

    /// Turns as dicts: `{"speaker", "start", "end", "text"}` (text is None
    /// unless an ASR pass populated it).
    #[getter]
    fn turns<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .turns
            .iter()
            .map(|t| {
                let d = PyDict::new(py);
                d.set_item("speaker", t.speaker.0)?;
                d.set_item("start", t.time.start)?;
                d.set_item("end", t.time.end)?;
                d.set_item("text", t.text.as_deref())?;
                Ok(d)
            })
            .collect()
    }

    /// Per-speaker rollup as dicts: `{"id", "label", "total_speech_s", "turn_count"}`.
    #[getter]
    fn speakers<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .speakers
            .iter()
            .map(|s| {
                let d = PyDict::new(py);
                d.set_item("id", s.id)?;
                d.set_item("label", &s.label)?;
                d.set_item("total_speech_s", s.total_speech_s)?;
                d.set_item("turn_count", s.turn_count)?;
                Ok(d)
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DiarizationResult(num_speakers={}, turns={})",
            self.inner.num_speakers,
            self.inner.turns.len()
        )
    }
}

fn utf8(buf: Vec<u8>) -> PyResult<String> {
    String::from_utf8(buf)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("invalid UTF-8: {e}")))
}

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
        let result = self.run_core(py, samples, sample_rate)?;

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

    /// Run diarization and return the typed [`DiarizationResult`] (projectable
    /// to JSON/RTTM/SRT/VTT/TXT).
    fn run_result(
        &self,
        py: Python<'_>,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> PyResult<DiarizationResult> {
        let inner = self.run_core(py, samples, sample_rate)?;
        Ok(DiarizationResult { inner })
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

    fn run_core(
        &self,
        py: Python<'_>,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> PyResult<RustDiarizationResult> {
        let sr = SampleRate::new(sample_rate).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid sample rate {sample_rate} (expected 8000..=192000)"
            ))
        })?;

        py.detach(|| self.pipeline.run(&samples, sr)).map_err(|e| match e {
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
        })
    }
}

#[pymodule]
fn _polyvoice(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pipeline>()?;
    m.add_class::<DiarizationResult>()?;
    Ok(())
}
