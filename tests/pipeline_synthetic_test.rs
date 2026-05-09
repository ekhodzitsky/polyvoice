//! M6a — synthetic-data integration tests for `polyvoice::pipeline`.
//!
//! Pure-CPU; no ONNX. Covers builder validation paths, end-to-end
//! Custom-profile run, and overlap-resegmentation toggling.

#![cfg(all(
    feature = "pipeline",
    feature = "onnx",
    feature = "segmentation",
    feature = "embedder",
    feature = "clusterer",
    feature = "resegmentation",
))]

use polyvoice::clusterer::{Clusterer, ClustererError};
use polyvoice::embedder::{Embedder, EmbedderError};
use polyvoice::pipeline_v1::{ClustererKind, ConfigError, Pipeline, PipelineConfig, PipelineError};
use polyvoice::segmentation::{RawSegment, SegmentationError, Segmenter};
use polyvoice::types::{Confidence, Profile, SampleRate, TimeRange};

// ----- Local mocks (re-defined here because the crate-internal mocks are #[cfg(test)]) -----

struct ConstSegmenter(Vec<RawSegment>);
impl Segmenter for ConstSegmenter {
    fn segment(&self, _audio: &[f32]) -> Result<Vec<RawSegment>, SegmentationError> {
        Ok(self.0.clone())
    }
    fn max_local_speakers(&self) -> usize {
        3
    }
    fn supports_overlap(&self) -> bool {
        true
    }
}

type AxisPicker = Box<dyn Fn(&[f32]) -> usize + Send + Sync>;

struct AxisEmbedder {
    dim: usize,
    axis_picker: AxisPicker,
}
impl Embedder for AxisEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        let axis = (self.axis_picker)(audio).min(self.dim - 1);
        let mut v = vec![0.0_f32; self.dim];
        v[axis] = 1.0;
        Ok(v)
    }
}

struct PerSampleClusterer {
    labels: Vec<usize>,
}
impl Clusterer for PerSampleClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        if embeddings.len() != self.labels.len() {
            return Err(ClustererError::AlgorithmFailed {
                detail: format!(
                    "labels {} vs embeddings {}",
                    self.labels.len(),
                    embeddings.len()
                ),
            });
        }
        Ok(self.labels.clone())
    }
    fn max_clusters(&self) -> usize {
        16
    }
}

fn raw(start: f64, end: f64, spk: u8, overlap: bool) -> RawSegment {
    RawSegment {
        time: TimeRange { start, end },
        local_speaker_idx: spk,
        is_overlap: overlap,
        confidence: Confidence::new(0.9).unwrap(),
    }
}

fn axis_picker_constant(axis: usize) -> AxisPicker {
    Box::new(move |_| axis)
}

// ----- Tests -----

#[test]
fn builder_validation_mobile_missing_registry() {
    let result = Pipeline::builder().profile(Profile::Mobile).build();
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected MissingRegistry for Mobile without registry"),
    };
    assert!(matches!(
        err,
        ConfigError::MissingRegistry {
            profile: Profile::Mobile
        }
    ));
}

#[test]
fn builder_validation_custom_missing_components() {
    let result = Pipeline::builder().profile(Profile::Custom).build();
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected MissingCustomComponent for Custom without components"),
    };
    assert!(matches!(err, ConfigError::MissingCustomComponent { .. }));
}

#[test]
fn pipeline_run_unsupported_sample_rate_errors() {
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(vec![raw(0.0, 1.0, 0, false)])))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![0] }))
        .build()
        .expect("custom build");
    let err = p
        .run(&vec![0.0_f32; 8000], SampleRate::new(8000).unwrap())
        .unwrap_err();
    assert!(matches!(
        err,
        PipelineError::UnsupportedSampleRate { actual: 8000 }
    ));
}

#[test]
fn pipeline_run_silence_returns_empty() {
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(Vec::new())))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![] }))
        .build()
        .expect("custom build");
    let r = p
        .run(&vec![0.0_f32; 16000], SampleRate::new(16000).unwrap())
        .unwrap();
    assert!(r.turns.is_empty());
    assert_eq!(r.num_speakers, 0);
}

#[test]
fn pipeline_run_two_speakers_through_custom_profile() {
    let segs = vec![raw(0.0, 1.0, 0, false), raw(2.0, 3.0, 1, false)];
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        clusterer: ClustererKind::NmeSc,
        resegment_overlap: false,
        min_speech_secs: 0.0,
        max_gap_secs: 0.0,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(segs)))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![0, 1] }))
        .build()
        .expect("custom build");
    let r = p
        .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
        .unwrap();
    assert_eq!(r.num_speakers, 2);
    assert_eq!(r.turns.len(), 2);
    let speakers: Vec<u32> = r.turns.iter().map(|t| t.speaker.0).collect();
    assert!(speakers.contains(&0));
    assert!(speakers.contains(&1));
}

#[test]
fn pipeline_run_returns_sorted_turns() {
    let segs = vec![raw(2.0, 3.0, 0, false), raw(0.0, 1.0, 0, false)];
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        resegment_overlap: false,
        min_speech_secs: 0.0,
        max_gap_secs: 0.0,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(segs)))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![0, 0] }))
        .build()
        .expect("custom build");
    let r = p
        .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
        .unwrap();
    for w in r.turns.windows(2) {
        assert!(w[0].time.start <= w[1].time.start);
    }
}

#[test]
fn pipeline_resegment_overlap_disabled_no_secondaries() {
    let segs = vec![
        raw(0.0, 1.0, 0, true),
        raw(0.0, 1.0, 1, true),
        raw(2.0, 3.0, 0, false),
    ];
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        resegment_overlap: false,
        min_speech_secs: 0.0,
        max_gap_secs: 0.0,
        ..PipelineConfig::default()
    };
    let p = Pipeline::builder()
        .config(cfg)
        .with_segmenter(Box::new(ConstSegmenter(segs)))
        .with_embedder(Box::new(AxisEmbedder {
            dim: 8,
            axis_picker: axis_picker_constant(0),
        }))
        .with_clusterer(Box::new(PerSampleClusterer { labels: vec![0] }))
        .build()
        .expect("custom build");
    let r = p
        .run(&vec![0.0_f32; 16000 * 4], SampleRate::new(16000).unwrap())
        .unwrap();
    assert_eq!(r.num_speakers, 1);
}
