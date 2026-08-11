use super::*;
use crate::types::{DiarizationResult, SpeakerId, SpeakerTurn, TimeRange};

fn sample_result() -> DiarizationResult {
    let turns = vec![SpeakerTurn {
        speaker: SpeakerId(0),
        time: TimeRange {
            start: 0.5,
            end: 2.0,
        },
        text: Some("hello".to_owned()),
        stable: true,
    }];
    DiarizationResult::new(vec![], turns, 1)
}

#[test]
fn render_result_rejects_unknown_format() {
    let result = sample_result();
    assert_eq!(
        render_result(&result, 42),
        Err(PolyvoiceStatus::InvalidArg as c_int)
    );
    assert_eq!(
        render_result(&result, -1),
        Err(PolyvoiceStatus::InvalidArg as c_int)
    );
}

#[test]
fn render_result_covers_every_format() {
    let result = sample_result();
    for (format, marker) in [
        (PolyvoiceFormat::Json as c_int, "num_speakers"),
        (PolyvoiceFormat::Rttm as c_int, "SPEAKER audio 1"),
        (PolyvoiceFormat::Srt as c_int, "00:00:00,500"),
        (PolyvoiceFormat::Vtt as c_int, "WEBVTT"),
        (PolyvoiceFormat::Txt as c_int, "SPEAKER_00: hello"),
    ] {
        let rendered = render_result(&result, format).unwrap();
        assert!(
            rendered.contains(marker),
            "format {format} missing marker {marker:?}: {rendered}"
        );
    }
}

#[test]
fn cache_dir_rejects_parent_dir_traversal() {
    for path in ["../evil", "models/../../evil", ".."] {
        assert_eq!(
            validate_cache_dir(path),
            Err(PolyvoiceStatus::InvalidArg as c_int),
            "traversal path must be rejected: {path}"
        );
    }
}

#[test]
fn cache_dir_accepts_absolute_and_relative_paths() {
    for path in ["/opt/polyvoice/models", "models/cache", "."] {
        assert!(
            validate_cache_dir(path).is_ok(),
            "legitimate cache dir must be accepted: {path}"
        );
    }
}

// ---------------------------------------------------------------------
// Entry-point tests: create/destroy lifecycle, run/run_format argument
// validation, and error-code mapping. Pipelines are built from the
// pipeline_v2 mock components so no ONNX models or network are needed.
// ---------------------------------------------------------------------

use crate::pipeline_v2::mocks::{MockClusterer, MockEmbedder, MockSegmenter, raw_segment};
use crate::resegmentation::OverlapResegmenter;
use std::ptr;

const INVALID_ARG: c_int = PolyvoiceStatus::InvalidArg as c_int;
const AUDIO_TOO_LONG: c_int = PolyvoiceStatus::AudioTooLong as c_int;
const INFERENCE: c_int = PolyvoiceStatus::Inference as c_int;
const REGISTRY: c_int = PolyvoiceStatus::Registry as c_int;

/// A pipeline handle backed by constant-output mock components: no model
/// files, no registry, no network.
fn mock_handle(segments: Vec<crate::segmentation::RawSegment>) -> *mut PolyvoicePipeline {
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        resegment_overlap: false,
        min_speech_secs: 0.0,
        max_gap_secs: 0.0,
        ..PipelineConfig::default()
    };
    let pipeline = Pipeline::from_components(
        cfg,
        Box::new(MockSegmenter { segments }),
        Box::new(MockEmbedder::default()),
        Box::new(MockClusterer::default()),
        Box::new(OverlapResegmenter::default()),
    );
    Box::into_raw(Box::new(PolyvoicePipeline { inner: pipeline }))
}

/// Embedder that always fails, to drive the pipeline-error mapping.
struct FailingEmbedder;

impl crate::embedder::Embedder for FailingEmbedder {
    fn dim(&self) -> usize {
        192
    }

    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, crate::embedder::EmbedderError> {
        Err(crate::embedder::EmbedderError::AudioTooShort {
            actual_secs: 0.0,
            min_secs: 0.01,
        })
    }
}

fn failing_handle() -> *mut PolyvoicePipeline {
    let cfg = PipelineConfig {
        profile: Profile::Custom,
        resegment_overlap: false,
        min_speech_secs: 0.0,
        max_gap_secs: 0.0,
        ..PipelineConfig::default()
    };
    let pipeline = Pipeline::from_components(
        cfg,
        Box::new(MockSegmenter {
            segments: vec![raw_segment(0.0, 1.0, 0, false)],
        }),
        Box::new(FailingEmbedder),
        Box::new(MockClusterer::default()),
        Box::new(OverlapResegmenter::default()),
    );
    Box::into_raw(Box::new(PolyvoicePipeline { inner: pipeline }))
}

#[test]
fn create_null_out_handle_returns_invalid_arg() {
    // SAFETY: null out_handle is the condition under test.
    let rc = unsafe { polyvoice_pipeline_create(0, ptr::null(), ptr::null_mut()) };
    assert_eq!(rc, INVALID_ARG);
}

#[test]
fn create_invalid_profile_returns_invalid_arg() {
    for profile in [-1, 2, 42] {
        let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
        // SAFETY: handle is non-null; invalid profile is the condition under test.
        let rc = unsafe { polyvoice_pipeline_create(profile, ptr::null(), &mut handle) };
        assert_eq!(rc, INVALID_ARG, "profile {profile} must be rejected");
        assert!(handle.is_null(), "handle must stay null on error");
    }
}

#[test]
fn create_rejects_non_utf8_cache_dir() {
    let dir = CString::new([0xFFu8, 0xFE]).unwrap();
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    // SAFETY: handle is non-null; dir is a valid nul-terminated (non-UTF-8) string.
    let rc = unsafe { polyvoice_pipeline_create(1, dir.as_ptr(), &mut handle) };
    assert_eq!(rc, INVALID_ARG);
    assert!(handle.is_null());
}

#[test]
fn create_rejects_parent_dir_traversal() {
    let dir = CString::new("../evil").unwrap();
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    // SAFETY: handle is non-null; dir is a valid nul-terminated string. The
    // traversal check runs before any registry or model access.
    let rc = unsafe { polyvoice_pipeline_create(0, dir.as_ptr(), &mut handle) };
    assert_eq!(rc, INVALID_ARG);
    assert!(handle.is_null());
}

#[test]
fn create_unwritable_cache_dir_maps_to_registry() {
    // A cache dir nested under a regular file cannot be created, so the
    // registry constructor fails before any download is attempted.
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("blocker");
    std::fs::write(&file, b"not a directory").unwrap();
    let bad = file.join("child");
    let dir = CString::new(bad.to_str().unwrap()).unwrap();
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    // SAFETY: handle is non-null; dir is a valid nul-terminated string.
    let rc = unsafe { polyvoice_pipeline_create(0, dir.as_ptr(), &mut handle) };
    assert_eq!(rc, REGISTRY);
    assert!(handle.is_null());
}

#[test]
fn run_null_out_params_return_invalid_arg() {
    let handle = mock_handle(vec![]);
    let samples = [0.0f32; 16000];
    let mut out: *mut c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: handle is valid, samples is valid; each null out-param is the
    // condition under test.
    let rc = unsafe {
        polyvoice_pipeline_run(
            handle,
            samples.as_ptr(),
            samples.len(),
            16000,
            ptr::null_mut(),
            &mut out_len,
        )
    };
    assert_eq!(rc, INVALID_ARG);
    // SAFETY: same contract as above.
    let rc = unsafe {
        polyvoice_pipeline_run(
            handle,
            samples.as_ptr(),
            samples.len(),
            16000,
            &mut out,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, INVALID_ARG);
    assert!(out.is_null());
    // SAFETY: handle was created by mock_handle and is destroyed exactly once.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn run_null_pipeline_returns_invalid_arg() {
    let samples = [0.0f32; 16000];
    let mut out: *mut c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: null pipeline is the condition under test.
    let rc = unsafe {
        polyvoice_pipeline_run(
            ptr::null_mut(),
            samples.as_ptr(),
            samples.len(),
            16000,
            &mut out,
            &mut out_len,
        )
    };
    assert_eq!(rc, INVALID_ARG);
    assert!(out.is_null());
}

#[test]
fn run_null_samples_returns_invalid_arg() {
    let handle = mock_handle(vec![]);
    let mut out: *mut c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: handle is valid; null samples is the condition under test.
    let rc =
        unsafe { polyvoice_pipeline_run(handle, ptr::null(), 0, 16000, &mut out, &mut out_len) };
    assert_eq!(rc, INVALID_ARG);
    assert!(out.is_null());
    // SAFETY: handle was created by mock_handle and is destroyed exactly once.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn run_rejects_out_of_range_sample_rate() {
    let handle = mock_handle(vec![]);
    let samples = [0.0f32; 16000];
    let mut out: *mut c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: handle and samples are valid; 100 Hz is outside the validated
    // sample-rate range.
    let rc = unsafe {
        polyvoice_pipeline_run(
            handle,
            samples.as_ptr(),
            samples.len(),
            100,
            &mut out,
            &mut out_len,
        )
    };
    assert_eq!(rc, INVALID_ARG);
    assert!(out.is_null());
    // SAFETY: handle was created by mock_handle and is destroyed exactly once.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn run_unsupported_sample_rate_maps_to_invalid_arg() {
    // 8 kHz is a valid `SampleRate` but the pipeline only runs at its
    // configured rate, so the UnsupportedSampleRate error maps to InvalidArg.
    let handle = mock_handle(vec![]);
    let samples = [0.0f32; 8000];
    let mut out: *mut c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: handle and samples are valid.
    let rc = unsafe {
        polyvoice_pipeline_run(
            handle,
            samples.as_ptr(),
            samples.len(),
            8000,
            &mut out,
            &mut out_len,
        )
    };
    assert_eq!(rc, INVALID_ARG);
    assert!(out.is_null());
    // SAFETY: handle was created by mock_handle and is destroyed exactly once.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn run_rejects_too_many_samples() {
    let handle = mock_handle(vec![]);
    let samples = [0.0f32; 16];
    let mut out: *mut c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // The length cap is checked before the slice is materialized, so the
    // small buffer is never read out of bounds.
    // SAFETY: handle is valid; the oversized length is the condition under test.
    let rc = unsafe {
        polyvoice_pipeline_run(
            handle,
            samples.as_ptr(),
            16000 * 3600 + 1,
            16000,
            &mut out,
            &mut out_len,
        )
    };
    assert_eq!(rc, AUDIO_TOO_LONG);
    assert!(out.is_null());
    // SAFETY: handle was created by mock_handle and is destroyed exactly once.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn run_json_success_round_trip() {
    let handle = mock_handle(vec![
        raw_segment(0.0, 1.0, 0, false),
        raw_segment(1.5, 2.5, 0, false),
    ]);
    let samples = vec![0.0f32; 16000 * 3];
    let mut out: *mut c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: handle, samples and out-params are all valid.
    let rc = unsafe {
        polyvoice_pipeline_run(
            handle,
            samples.as_ptr(),
            samples.len(),
            16000,
            &mut out,
            &mut out_len,
        )
    };
    assert_eq!(rc, PolyvoiceStatus::Ok as c_int);
    assert!(!out.is_null());
    // SAFETY: out was returned by polyvoice_pipeline_run and is non-null.
    let json = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();
    assert_eq!(json.len(), out_len);
    assert!(
        json.contains("\"num_speakers\":1"),
        "unexpected JSON: {json}"
    );
    assert!(json.contains("turns"));
    // SAFETY: out was returned by polyvoice_pipeline_run.
    unsafe { polyvoice_free_string(out, out_len) };
    // SAFETY: handle was created by mock_handle and is destroyed exactly once.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn run_maps_pipeline_failure_to_inference() {
    let handle = failing_handle();
    let samples = vec![0.0f32; 16000 * 3];
    let mut out: *mut c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: handle, samples and out-params are all valid; the embedder
    // fails by construction.
    let rc = unsafe {
        polyvoice_pipeline_run(
            handle,
            samples.as_ptr(),
            samples.len(),
            16000,
            &mut out,
            &mut out_len,
        )
    };
    assert_eq!(rc, INFERENCE);
    assert!(out.is_null());
    // SAFETY: handle was created by failing_handle and is destroyed exactly once.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn run_format_null_out_params_return_invalid_arg() {
    let samples = [0.0f32; 16000];
    let mut out: *mut c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: each null out-param is the condition under test; the pipeline
    // handle is never dereferenced because validation fails first.
    let rc = unsafe {
        polyvoice_pipeline_run_format(
            ptr::null_mut(),
            samples.as_ptr(),
            samples.len(),
            16000,
            PolyvoiceFormat::Json as c_int,
            ptr::null_mut(),
            &mut out_len,
        )
    };
    assert_eq!(rc, INVALID_ARG);
    // SAFETY: same contract as above.
    let rc = unsafe {
        polyvoice_pipeline_run_format(
            ptr::null_mut(),
            samples.as_ptr(),
            samples.len(),
            16000,
            PolyvoiceFormat::Json as c_int,
            &mut out,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, INVALID_ARG);
    assert!(out.is_null());
}

#[test]
fn run_format_rejects_unknown_format() {
    let handle = mock_handle(vec![]);
    let samples = [0.0f32; 16000];
    for format in [-1, 5, 42] {
        let mut out: *mut c_char = ptr::null_mut();
        let mut out_len: usize = 0;
        // SAFETY: handle, samples and out-params are valid; the format value
        // is the condition under test.
        let rc = unsafe {
            polyvoice_pipeline_run_format(
                handle,
                samples.as_ptr(),
                samples.len(),
                16000,
                format,
                &mut out,
                &mut out_len,
            )
        };
        assert_eq!(rc, INVALID_ARG, "format {format} must be rejected");
        assert!(out.is_null());
    }
    // SAFETY: handle was created by mock_handle and is destroyed exactly once.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn run_format_renders_every_format() {
    let handle = mock_handle(vec![raw_segment(0.0, 1.0, 0, false)]);
    let samples = vec![0.0f32; 16000 * 2];
    for format in PolyvoiceFormat::Json as c_int..=PolyvoiceFormat::Txt as c_int {
        let mut out: *mut c_char = ptr::null_mut();
        let mut out_len: usize = 0;
        // SAFETY: handle, samples and out-params are all valid.
        let rc = unsafe {
            polyvoice_pipeline_run_format(
                handle,
                samples.as_ptr(),
                samples.len(),
                16000,
                format,
                &mut out,
                &mut out_len,
            )
        };
        assert_eq!(rc, PolyvoiceStatus::Ok as c_int, "format {format} failed");
        assert!(!out.is_null());
        // SAFETY: out was returned by polyvoice_pipeline_run_format and is non-null.
        let s = unsafe { CStr::from_ptr(out) }
            .to_string_lossy()
            .into_owned();
        assert_eq!(s.len(), out_len);
        // SAFETY: out was returned by polyvoice_pipeline_run_format.
        unsafe { polyvoice_free_string(out, out_len) };
    }
    // SAFETY: handle was created by mock_handle and is destroyed exactly once.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn destroy_null_is_noop() {
    // SAFETY: destroy explicitly accepts null.
    unsafe { polyvoice_pipeline_destroy(ptr::null_mut()) };
}

#[test]
fn free_string_null_is_noop() {
    // SAFETY: free_string explicitly accepts null.
    unsafe { polyvoice_free_string(ptr::null_mut(), 0) };
}
