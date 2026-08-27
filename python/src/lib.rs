#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]

//! pyo3 bindings for the polyvoice Pipeline v2.

use polyvoice::models::ModelRegistry;
use polyvoice::pipeline_v2::{
    ClustererKind, Pipeline as RustPipeline, PipelineConfig, PipelineError,
};
use polyvoice::types::{DiarizationResult as RustDiarizationResult, Profile, SampleRate};
use std::path::{Component, Path, PathBuf};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Reject cache paths that contain `..` components (parity with CLI/FFI).
fn reject_path_traversal(path: &str) -> PyResult<()> {
    if Path::new(path)
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "path traversal rejected",
        ));
    }
    Ok(())
}

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
    ///
    /// `clusterer` is `"vbx"` (default, matching the CLI) or `"ahc"`. VBx
    /// resolves its PLDA params via `vbx_plda_dir`, then the
    /// `POLYVOICE_VBX_PLDA_DIR` env var, then a registry download.
    #[staticmethod]
    #[pyo3(signature = (models_cache=None, clusterer=None, vbx_plda_dir=None))]
    fn mobile(
        models_cache: Option<&str>,
        clusterer: Option<&str>,
        vbx_plda_dir: Option<&str>,
    ) -> PyResult<Self> {
        Self::build_profile(Profile::Mobile, models_cache, clusterer, vbx_plda_dir)
    }

    /// Build a Balanced-profile Pipeline.
    ///
    /// Same kwargs as [`Pipeline::mobile`]. Defaults to VBx (matching the
    /// CLI); pass `clusterer="ahc"` for the cosine-AHC backend.
    #[staticmethod]
    #[pyo3(signature = (models_cache=None, clusterer=None, vbx_plda_dir=None))]
    fn balanced(
        models_cache: Option<&str>,
        clusterer: Option<&str>,
        vbx_plda_dir: Option<&str>,
    ) -> PyResult<Self> {
        Self::build_profile(Profile::Balanced, models_cache, clusterer, vbx_plda_dir)
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
    fn build_profile(
        profile: Profile,
        models_cache: Option<&str>,
        clusterer: Option<&str>,
        vbx_plda_dir: Option<&str>,
    ) -> PyResult<Self> {
        if let Some(path) = models_cache {
            reject_path_traversal(path)?;
        }
        if let Some(path) = vbx_plda_dir {
            reject_path_traversal(path)?;
        }
        let registry = match models_cache {
            Some(path) => ModelRegistry::with_cache_dir(path),
            None => ModelRegistry::default(),
        }
        .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("model registry: {e}")))?;

        let mut cfg = PipelineConfig {
            profile,
            ..PipelineConfig::default()
        };
        match clusterer {
            // VBx is the default (matches the CLI); the builder resolves PLDA
            // via vbx_plda_dir → POLYVOICE_VBX_PLDA_DIR → registry download.
            Some("vbx") | None => {
                cfg.clusterer = ClustererKind::Vbx;
                cfg.vbx_plda_dir = vbx_plda_dir.map(PathBuf::from);
            }
            Some("ahc") => {
                cfg.clusterer = ClustererKind::Ahc {
                    threshold: polyvoice::DEFAULT_AHC_THRESHOLD,
                };
            }
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown clusterer '{other}' (expected 'vbx' or 'ahc')"
                )));
            }
        }

        let pipeline = RustPipeline::builder()
            .config(cfg)
            .with_models_from(registry)
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "pipeline build: {e} (for VBx set vbx_plda_dir= / POLYVOICE_VBX_PLDA_DIR, allow registry PLDA download, or clusterer='ahc')"
                ))
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
            PipelineError::AudioTooLong {
                actual_samples,
                max_samples,
            } => pyo3::exceptions::PyValueError::new_err(format!(
                "audio too long: {actual_samples} samples exceeds max {max_samples} (~1 hour at 16 kHz)"
            )),
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
