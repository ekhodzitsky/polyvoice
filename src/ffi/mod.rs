//! C FFI ABI v3 for polyvoice Pipeline (v0.6.5+).
//!
//! Threading model: `PolyvoicePipeline` is `Send`. Each `*mut PolyvoicePipeline`
//! owns its data; callers must call `polyvoice_pipeline_destroy` exactly once.
//! All entry points are wrapped in `catch_unwind` per spec §8.4.

use crate::models::ModelRegistry;
use crate::pipeline_v2::Pipeline;
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
                // Reject path-traversal attempts (e.g. "../../evil") before the
                // path is passed to ModelRegistry::with_cache_dir.  FFI-002.
                let cache_path = std::path::Path::new(s);
                if cache_path.is_absolute() {
                    return Err(PolyvoiceStatus::InvalidArg as c_int);
                }
                if cache_path
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
                {
                    return Err(PolyvoiceStatus::InvalidArg as c_int);
                }
                ModelRegistry::with_cache_dir(s)
            }
            .map_err(|_| PolyvoiceStatus::Registry as c_int)?;
            let pipeline = Pipeline::builder()
                .profile(prof)
                .with_models_from(registry)
                .build()
                .map_err(|e| match e {
                    crate::pipeline_v2::ConfigError::Registry(_) |
                    crate::pipeline_v2::ConfigError::UnknownModel { .. } => {
                        PolyvoiceStatus::Registry as c_int
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
        if pipeline.is_null() || samples.is_null() || out_json.is_null() || out_json_len.is_null() {
            return Err(PolyvoiceStatus::InvalidArg as c_int);
        }
        let pipeline = unsafe { // SAFETY: pipeline was checked non-null; caller owns it for the duration of this call.
            &*pipeline
        };
        // SAFETY: samples was checked non-null; n_samples is caller-provided length.
        const MAX_SAMPLES: usize = 16000 * 3600; // 1 hour at 16 kHz
        if n_samples > MAX_SAMPLES {
            return Err(PolyvoiceStatus::AudioTooLong as c_int);
        }
        let samples = unsafe { // SAFETY: samples was checked non-null; n_samples was validated against MAX_SAMPLES.
            std::slice::from_raw_parts(samples, n_samples)
        };
        let sr = SampleRate::new(sample_rate)
            .ok_or(PolyvoiceStatus::InvalidArg as c_int)?;
        let result = pipeline
            .inner
            .run(samples, sr)
            .map_err(|e| match e {
                crate::pipeline_v2::PipelineError::UnsupportedSampleRate { .. } => {
                    PolyvoiceStatus::InvalidArg as c_int
                }
                crate::pipeline_v2::PipelineError::ModelLoad { .. } => {
                    PolyvoiceStatus::ModelLoad as c_int
                }
                crate::pipeline_v2::PipelineError::Registry(_) => {
                    PolyvoiceStatus::Registry as c_int
                }
                _ => PolyvoiceStatus::Inference as c_int,
            })?;
        let json =
            serde_json::to_string(&result).map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        let len = json.len();
        let cstr = CString::new(json).map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        let ptr_out = cstr.into_raw();
        unsafe { // SAFETY: out_json and out_json_len were checked non-null above.
            *out_json = ptr_out;
            *out_json_len = len;
        }
        Ok(())
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
        if pipeline.is_null() || samples.is_null() || out_str.is_null() || out_str_len.is_null() {
            return Err(PolyvoiceStatus::InvalidArg as c_int);
        }
        // Reject unknown formats before touching the pipeline.
        if !(PolyvoiceFormat::Json as c_int..=PolyvoiceFormat::Txt as c_int).contains(&format) {
            return Err(PolyvoiceStatus::InvalidArg as c_int);
        }
        let pipeline = unsafe { // SAFETY: pipeline was checked non-null; caller owns it for the duration of this call.
            &*pipeline
        };
        const MAX_SAMPLES: usize = 16000 * 3600; // 1 hour at 16 kHz
        if n_samples > MAX_SAMPLES {
            return Err(PolyvoiceStatus::AudioTooLong as c_int);
        }
        let samples = unsafe { // SAFETY: samples was checked non-null; n_samples was validated against MAX_SAMPLES.
            std::slice::from_raw_parts(samples, n_samples)
        };
        let sr = SampleRate::new(sample_rate)
            .ok_or(PolyvoiceStatus::InvalidArg as c_int)?;
        let result = pipeline
            .inner
            .run(samples, sr)
            .map_err(|e| match e {
                crate::pipeline_v2::PipelineError::UnsupportedSampleRate { .. } => {
                    PolyvoiceStatus::InvalidArg as c_int
                }
                crate::pipeline_v2::PipelineError::ModelLoad { .. } => {
                    PolyvoiceStatus::ModelLoad as c_int
                }
                crate::pipeline_v2::PipelineError::Registry(_) => {
                    PolyvoiceStatus::Registry as c_int
                }
                _ => PolyvoiceStatus::Inference as c_int,
            })?;
        let rendered = render_result(&result, format)?;
        let len = rendered.len();
        let cstr = CString::new(rendered).map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        let ptr_out = cstr.into_raw();
        unsafe { // SAFETY: out_str and out_str_len were checked non-null above.
            *out_str = ptr_out;
            *out_str_len = len;
        }
        Ok(())
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
}
