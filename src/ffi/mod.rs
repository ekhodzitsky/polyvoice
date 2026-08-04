//! C FFI ABI v3 for polyvoice Pipeline (v0.6.5+).
//!
//! The pipeline matches the CLI/MCP production default: pipeline v2 with VBx
//! clustering. On first use the builder resolves the VBx PLDA params from
//! `POLYVOICE_VBX_PLDA_DIR` or downloads them via the model registry (needs
//! network access unless pre-cached).
//!
//! Threading model: `PolyvoicePipeline` is `Send`. Each `*mut PolyvoicePipeline`
//! owns its data; callers must call `polyvoice_pipeline_destroy` exactly once.
//! All entry points are wrapped in `catch_unwind` per spec §8.4.

use crate::models::ModelRegistry;
use crate::pipeline_v2::{ClustererKind, Pipeline, PipelineConfig};
use crate::types::{Profile, SampleRate};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[repr(C)]
pub enum PolyvoiceProfile {
    Mobile = 0,
    Balanced = 1,
}

#[repr(C)]
pub enum PolyvoiceStatus {
    Ok = 0,
    InvalidArg = 1,
    /// Reserved for ABI stability: the current implementation never returns
    /// this status (pipeline_v2 has no matching error). Do not reuse the value.
    AudioTooShort = 2,
    AudioTooLong = 3,
    ModelLoad = 10,
    Inference = 11,
    OutOfMemory = 20,
    Registry = 30,
    Internal = 99,
}

/// Output format selector for `polyvoice_pipeline_run_format`.
#[repr(C)]
pub enum PolyvoiceFormat {
    Json = 0,
    Rttm = 1,
    Srt = 2,
    Vtt = 3,
    Txt = 4,
}

/// RTTM has a file-id column but the FFI runs on a raw sample buffer with no
/// filename; emit this fixed id (callers can post-process if they need another).
const FFI_RTTM_FILE_ID: &str = "audio";

/// Reject path-traversal attempts (e.g. `"../../evil"`) before the path is
/// passed to `ModelRegistry::with_cache_dir`. Absolute paths such as
/// `/opt/polyvoice/models` are legitimate cache locations and are accepted.
fn validate_cache_dir(s: &str) -> Result<(), c_int> {
    let cache_path = std::path::Path::new(s);
    if cache_path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(PolyvoiceStatus::InvalidArg as c_int);
    }
    Ok(())
}

/// Project `result` into the requested format. Returns a status code on failure.
fn render_result(result: &crate::types::DiarizationResult, format: c_int) -> Result<String, c_int> {
    let mut buf: Vec<u8> = Vec::new();
    match format {
        f if f == PolyvoiceFormat::Json as c_int => {
            return serde_json::to_string(result).map_err(|_| PolyvoiceStatus::Internal as c_int);
        }
        f if f == PolyvoiceFormat::Rttm as c_int => {
            crate::rttm::write_rttm(&mut buf, FFI_RTTM_FILE_ID, &result.turns)
                .map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        }
        f if f == PolyvoiceFormat::Srt as c_int => {
            crate::format::write_srt(&mut buf, &result.turns)
                .map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        }
        f if f == PolyvoiceFormat::Vtt as c_int => {
            crate::format::write_vtt(&mut buf, &result.turns)
                .map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        }
        f if f == PolyvoiceFormat::Txt as c_int => {
            crate::format::write_txt(&mut buf, &result.turns)
                .map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        }
        _ => return Err(PolyvoiceStatus::InvalidArg as c_int),
    }
    String::from_utf8(buf).map_err(|_| PolyvoiceStatus::Internal as c_int)
}

pub struct PolyvoicePipeline {
    inner: Pipeline,
}

/// Create a new pipeline from a profile.
///
/// # Safety
/// - `models_cache_dir`, if non-null, must point to a valid nul-terminated UTF-8 string.
/// - `out_handle` must be a valid non-null pointer to a `*mut PolyvoicePipeline`.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[unsafe(no_mangle)] // SAFETY: preserves symbol name for C linkage.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[rustfmt::skip]
pub unsafe extern "C" fn // SAFETY: caller upholds safety contract.
polyvoice_pipeline_create(
    profile: c_int,
    models_cache_dir: *const c_char,
    out_handle: *mut *mut PolyvoicePipeline,
) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(
        || -> Result<*mut PolyvoicePipeline, c_int> {
            if out_handle.is_null() {
                return Err(PolyvoiceStatus::InvalidArg as c_int);
            }
            let prof = match profile {
                0 => Profile::Mobile,
                1 => Profile::Balanced,
                _ => return Err(PolyvoiceStatus::InvalidArg as c_int),
            };
            let registry = if models_cache_dir.is_null() {
                ModelRegistry::default()
            } else {
                let s = unsafe { // SAFETY: caller guarantees models_cache_dir is a valid nul-terminated string.
                    CStr::from_ptr(models_cache_dir)
                }
                .to_str()
                    .map_err(|_| PolyvoiceStatus::InvalidArg as c_int)?;
                validate_cache_dir(s)?;
                ModelRegistry::with_cache_dir(s)
            }
            .map_err(|_| PolyvoiceStatus::Registry as c_int)?;
            // Same production default as the CLI/MCP front doors: pipeline v2
            // with VBx clustering (the builder resolves the PLDA params from
            // POLYVOICE_VBX_PLDA_DIR or the registry download).
            let config = PipelineConfig {
                profile: prof,
                clusterer: ClustererKind::Vbx,
                ..PipelineConfig::default()
            };
            let pipeline = Pipeline::builder()
                .config(config)
                .with_models_from(registry)
                .build()
                .map_err(|e| match e {
                    crate::pipeline_v2::ConfigError::Registry(_) |
                    crate::pipeline_v2::ConfigError::UnknownModel { .. } => {
                        PolyvoiceStatus::Registry as c_int
                    }
                    crate::pipeline_v2::ConfigError::Load { .. } => {
                        PolyvoiceStatus::ModelLoad as c_int
                    }
                    crate::pipeline_v2::ConfigError::MissingRegistry { .. } |
                    crate::pipeline_v2::ConfigError::CustomComponentInProfile { .. } |
                    crate::pipeline_v2::ConfigError::RegistryInCustomProfile |
                    crate::pipeline_v2::ConfigError::MissingCustomComponent { .. } => {
                        PolyvoiceStatus::InvalidArg as c_int
                    }
                })?;
            Ok(Box::into_raw(Box::new(PolyvoicePipeline { inner: pipeline })))
        },
    ));
    match r {
        Ok(Ok(handle)) => {
            unsafe { // SAFETY: out_handle was checked non-null inside the closure above.
                *out_handle = handle;
            }
            PolyvoiceStatus::Ok as c_int
        }
        Ok(Err(code)) => code,
        Err(_) => PolyvoiceStatus::Internal as c_int,
    }
}

/// Shared body of `polyvoice_pipeline_run` and `polyvoice_pipeline_run_format`:
/// validates the raw inputs, runs the pipeline, and renders the result in
/// `format` (see `PolyvoiceFormat`). Returns the rendered string; the caller
/// hands it to C via [`emit_c_string`].
///
/// # Safety
/// - `pipeline` must be a valid pointer returned by `polyvoice_pipeline_create`.
/// - `samples` must point to at least `n_samples` valid f32 values.
unsafe fn run_impl(
    pipeline: *mut PolyvoicePipeline,
    samples: *const c_float,
    n_samples: usize,
    sample_rate: u32,
    format: c_int,
) -> Result<String, c_int> {
    if pipeline.is_null() || samples.is_null() {
        return Err(PolyvoiceStatus::InvalidArg as c_int);
    }
    // Reject unknown formats before touching the pipeline.
    if !(PolyvoiceFormat::Json as c_int..=PolyvoiceFormat::Txt as c_int).contains(&format) {
        return Err(PolyvoiceStatus::InvalidArg as c_int);
    }
    let pipeline = unsafe {
        // SAFETY: pipeline was checked non-null; caller owns it for the duration of this call.
        &*pipeline
    };
    // SAFETY: samples was checked non-null; n_samples is caller-provided length.
    const MAX_SAMPLES: usize = 16000 * 3600; // 1 hour at 16 kHz
    if n_samples > MAX_SAMPLES {
        return Err(PolyvoiceStatus::AudioTooLong as c_int);
    }
    let samples = unsafe {
        // SAFETY: samples was checked non-null; n_samples was validated against MAX_SAMPLES.
        std::slice::from_raw_parts(samples, n_samples)
    };
    let sr = SampleRate::new(sample_rate).ok_or(PolyvoiceStatus::InvalidArg as c_int)?;
    let result = pipeline.inner.run(samples, sr).map_err(|e| match e {
        crate::pipeline_v2::PipelineError::UnsupportedSampleRate { .. } => {
            PolyvoiceStatus::InvalidArg as c_int
        }
        crate::pipeline_v2::PipelineError::Registry(_) => PolyvoiceStatus::Registry as c_int,
        _ => PolyvoiceStatus::Inference as c_int,
    })?;
    render_result(&result, format)
}

/// Hand `rendered` to C as a nul-terminated string written to
/// `out_str`/`out_str_len`. The string must later be freed with
/// `polyvoice_free_string`.
///
/// # Safety
/// `out_str` and `out_str_len` must be valid non-null pointers.
unsafe fn emit_c_string(
    rendered: String,
    out_str: *mut *mut c_char,
    out_str_len: *mut usize,
) -> Result<(), c_int> {
    let len = rendered.len();
    let cstr = CString::new(rendered).map_err(|_| PolyvoiceStatus::Internal as c_int)?;
    let ptr_out = cstr.into_raw();
    unsafe {
        // SAFETY: out_str and out_str_len were checked non-null by the caller.
        *out_str = ptr_out;
        *out_str_len = len;
    }
    Ok(())
}

/// Run diarization on a buffer of f32 samples and return JSON.
///
/// # Safety
/// - `pipeline` must be a valid pointer returned by `polyvoice_pipeline_create`.
/// - `samples` must point to at least `n_samples` valid f32 values.
/// - `out_json` and `out_json_len` must be valid non-null pointers.
/// - The returned `*out_json` string must be freed with `polyvoice_free_string`.
/// - Must not be called concurrently with another call to `polyvoice_pipeline_run`
///   or `polyvoice_pipeline_destroy` on the same handle.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[unsafe(no_mangle)] // SAFETY: preserves symbol name for C linkage.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[rustfmt::skip]
pub unsafe extern "C" fn // SAFETY: caller upholds safety contract.
polyvoice_pipeline_run(
    pipeline: *mut PolyvoicePipeline,
    samples: *const c_float,
    n_samples: usize,
    sample_rate: u32,
    out_json: *mut *mut c_char,
    out_json_len: *mut usize,
) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<(), c_int> {
        if out_json.is_null() || out_json_len.is_null() {
            return Err(PolyvoiceStatus::InvalidArg as c_int);
        }
        let rendered = unsafe { // SAFETY: caller upholds the safety contract documented in # Safety above.
            run_impl(pipeline, samples, n_samples, sample_rate, PolyvoiceFormat::Json as c_int)
        }?;
        unsafe { // SAFETY: out_json and out_json_len were checked non-null above.
            emit_c_string(rendered, out_json, out_json_len)
        }
    }));
    match r {
        Ok(Ok(())) => PolyvoiceStatus::Ok as c_int,
        Ok(Err(code)) => code,
        Err(_) => PolyvoiceStatus::Internal as c_int,
    }
}

/// Run diarization and return the result rendered in the requested format
/// (see `PolyvoiceFormat`: 0=JSON, 1=RTTM, 2=SRT, 3=VTT, 4=TXT).
///
/// Identical contract to `polyvoice_pipeline_run` otherwise. RTTM output uses
/// the fixed file id `audio`. Unknown `format` values return `InvalidArg`.
///
/// # Safety
/// - `pipeline` must be a valid pointer returned by `polyvoice_pipeline_create`.
/// - `samples` must point to at least `n_samples` valid f32 values.
/// - `out_str` and `out_str_len` must be valid non-null pointers.
/// - The returned `*out_str` string must be freed with `polyvoice_free_string`.
/// - Must not be called concurrently with another call to `polyvoice_pipeline_run`,
///   `polyvoice_pipeline_run_format`, or `polyvoice_pipeline_destroy` on the same handle.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[unsafe(no_mangle)] // SAFETY: preserves symbol name for C linkage.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[rustfmt::skip]
pub unsafe extern "C" fn // SAFETY: caller upholds safety contract.
polyvoice_pipeline_run_format(
    pipeline: *mut PolyvoicePipeline,
    samples: *const c_float,
    n_samples: usize,
    sample_rate: u32,
    format: c_int,
    out_str: *mut *mut c_char,
    out_str_len: *mut usize,
) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(|| -> Result<(), c_int> {
        if out_str.is_null() || out_str_len.is_null() {
            return Err(PolyvoiceStatus::InvalidArg as c_int);
        }
        let rendered = unsafe { // SAFETY: caller upholds the safety contract documented in # Safety above.
            run_impl(pipeline, samples, n_samples, sample_rate, format)
        }?;
        unsafe { // SAFETY: out_str and out_str_len were checked non-null above.
            emit_c_string(rendered, out_str, out_str_len)
        }
    }));
    match r {
        Ok(Ok(())) => PolyvoiceStatus::Ok as c_int,
        Ok(Err(code)) => code,
        Err(_) => PolyvoiceStatus::Internal as c_int,
    }
}

/// Destroy a pipeline created by `polyvoice_pipeline_create`.
///
/// # Safety
/// `pipeline` must be a valid pointer returned by `polyvoice_pipeline_create`, or null.
/// Must be called exactly once per handle.
/// Must not be called concurrently with any `polyvoice_pipeline_run` call on the
/// same handle.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[unsafe(no_mangle)] // SAFETY: preserves symbol name for C linkage.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[rustfmt::skip]
pub unsafe extern "C" fn // SAFETY: caller upholds safety contract.
polyvoice_pipeline_destroy(pipeline: *mut PolyvoicePipeline) {
    if !pipeline.is_null()
        && catch_unwind(AssertUnwindSafe(|| {
            unsafe { // SAFETY: pipeline is non-null and was created by Box::into_raw; caller destroys exactly once.
                drop(Box::from_raw(pipeline));
            }
        }))
        .is_err()
    {
        eprintln!("polyvoice: panic during cleanup (foreign thread?)");
    }
}

/// Free a string returned by `polyvoice_pipeline_run` or `polyvoice_pipeline_run_format`.
///
/// # Safety
/// `p` must be a pointer returned by `polyvoice_pipeline_run` /
/// `polyvoice_pipeline_run_format`, or null.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[unsafe(no_mangle)] // SAFETY: preserves symbol name for C linkage.
// SAFETY: caller upholds the safety contract documented in # Safety above.
#[rustfmt::skip]
pub unsafe extern "C" fn // SAFETY: caller upholds safety contract.
polyvoice_free_string(p: *mut c_char, _n: usize) {
    if !p.is_null()
        && catch_unwind(AssertUnwindSafe(|| {
            unsafe { // SAFETY: p is non-null and was created by CString::into_raw in a polyvoice run function.
                drop(CString::from_raw(p));
            }
        }))
        .is_err()
    {
        eprintln!("polyvoice: panic during cleanup (foreign thread?)");
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
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
        let rc = unsafe {
            polyvoice_pipeline_run(handle, ptr::null(), 0, 16000, &mut out, &mut out_len)
        };
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
}
