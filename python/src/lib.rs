use pyo3::prelude::*;
use std::path::{Path, PathBuf};

/// A speaker turn with speaker ID and time boundaries.
#[pyclass(frozen, from_py_object)]
#[derive(Clone)]
struct Turn {
    #[pyo3(get)]
    speaker: String,
    #[pyo3(get)]
    start: f64,
    #[pyo3(get)]
    end: f64,
}

#[pymethods]
impl Turn {
    fn __repr__(&self) -> String {
        format!(
            "Turn(speaker='{}', start={:.2}, end={:.2})",
            self.speaker, self.start, self.end
        )
    }

    #[getter]
    fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Speaker diarization pipeline.
///
/// Usage:
///     pipeline = Pipeline("models/")
///     turns = pipeline("meeting.wav")
///     for turn in turns:
///         print(f"{turn.speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
#[pyclass]
struct Pipeline {
    extractor: polyvoice::FbankOnnxExtractor,
    vad_path: PathBuf,
    threshold: f32,
    max_speakers: usize,
}

#[pymethods]
impl Pipeline {
    #[new]
    #[pyo3(signature = (model_dir, threshold=0.5, max_speakers=64))]
    fn new(model_dir: &str, threshold: f32, max_speakers: usize) -> PyResult<Self> {
        let model_dir = Path::new(model_dir);
        let wespeaker_path = model_dir.join("wespeaker_resnet34.onnx");
        let vad_path = model_dir.join("silero_vad.onnx");

        if !wespeaker_path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("WeSpeaker model not found: {}", wespeaker_path.display()),
            ));
        }
        if !vad_path.exists() {
            return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                format!("Silero VAD model not found: {}", vad_path.display()),
            ));
        }

        let extractor = polyvoice::FbankOnnxExtractor::new(&wespeaker_path, 256, 4)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            extractor,
            vad_path,
            threshold,
            max_speakers,
        })
    }

    /// Run diarization on a WAV file path or f32 sample list.
    #[pyo3(signature = (audio, sample_rate=16000))]
    fn __call__(&self, audio: AudioInput, sample_rate: u32) -> PyResult<Vec<Turn>> {
        let samples = match audio {
            AudioInput::Path(path) => {
                let (s, _sr) = polyvoice::wav::read_wav(Path::new(&path))
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
                s
            }
            AudioInput::Samples(s) => s,
        };

        let mut vad = polyvoice::SileroVad::new(&self.vad_path, 512)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let config = polyvoice::DiarizationConfig {
            threshold: self.threshold,
            max_speakers: self.max_speakers,
            sample_rate: polyvoice::SampleRate::new(sample_rate).ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid sample rate")
            })?,
            ..Default::default()
        };
        let vad_config = polyvoice::VadConfig::default();
        let pipeline = polyvoice::Pipeline::new(config, vad_config);

        let result = pipeline
            .run(&samples, &self.extractor, &mut vad)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(result
            .turns
            .into_iter()
            .map(|t| Turn {
                speaker: format!("{}", t.speaker),
                start: t.time.start,
                end: t.time.end,
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "Pipeline(threshold={}, max_speakers={})",
            self.threshold, self.max_speakers
        )
    }
}

#[derive(FromPyObject)]
enum AudioInput {
    Path(String),
    Samples(Vec<f32>),
}

#[pymodule]
fn _polyvoice(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Pipeline>()?;
    m.add_class::<Turn>()?;
    Ok(())
}
