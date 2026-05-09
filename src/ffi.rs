//! Legacy v0.5.2 — C FFI ABI for the legacy Pipeline.
//!
//! Threading model: `PolyvoicePipeline` is `Send`. Each `*mut PolyvoicePipeline`
//! owns its data; callers must call `polyvoice_pipeline_destroy` exactly once.
//! All entry points are wrapped in `catch_unwind` per spec §8.4.

use crate::models::ModelRegistry;
use crate::pipeline::Pipeline;
use crate::types::{DiarizationConfig, Profile};
use crate::VadConfig;
use crate::{FbankOnnxExtractor, SileroVad};
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

pub struct PolyvoicePipeline {
    inner: Pipeline,
    extractor: FbankOnnxExtractor,
    vad: SileroVad,
}

/// Create a new pipeline from a profile.
///
/// # Safety
/// - `models_cache_dir`, if non-null, must point to a valid nul-terminated UTF-8 string.
/// - `out_handle` must be a valid non-null pointer to a `*mut PolyvoicePipeline`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn polyvoice_pipeline_create(
    profile: PolyvoiceProfile,
    models_cache_dir: *const c_char,
    out_handle: *mut *mut PolyvoicePipeline,
) -> c_int {
    let r = catch_unwind(AssertUnwindSafe(
        || -> Result<*mut PolyvoicePipeline, c_int> {
            if out_handle.is_null() {
                return Err(PolyvoiceStatus::InvalidArg as c_int);
            }
            let prof = match profile {
                PolyvoiceProfile::Mobile => Profile::Mobile,
                PolyvoiceProfile::Balanced => Profile::Balanced,
            };
            let registry = if models_cache_dir.is_null() {
                ModelRegistry::default()
            } else {
                // SAFETY: caller guarantees models_cache_dir is a valid nul-terminated string.
                let s = unsafe { CStr::from_ptr(models_cache_dir) }
                    .to_str()
                    .map_err(|_| PolyvoiceStatus::InvalidArg as c_int)?;
                ModelRegistry::with_cache_dir(s)
            }
            .map_err(|_| PolyvoiceStatus::Registry as c_int)?;
            let models = registry
                .ensure_for_profile(prof)
                .map_err(|_| PolyvoiceStatus::Registry as c_int)?;
            let extractor = FbankOnnxExtractor::new(
                &models.embedder_path,
                prof.embedding_dim(),
                1,
            )
            .map_err(|_| PolyvoiceStatus::ModelLoad as c_int)?;
            let vad = SileroVad::new(&models.segmenter_path, 512)
                .map_err(|_| PolyvoiceStatus::ModelLoad as c_int)?;
            let pipeline = Pipeline::new(DiarizationConfig::default(), VadConfig::default());
            Ok(Box::into_raw(Box::new(PolyvoicePipeline {
                inner: pipeline,
                extractor,
                vad,
            })))
        },
    ));
    match r {
        Ok(Ok(handle)) => {
            // SAFETY: out_handle was checked non-null inside the closure above.
            unsafe {
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn polyvoice_pipeline_run(
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
        // SAFETY: pipeline was checked non-null; caller owns it for the duration of this call.
        let pipeline = unsafe { &mut *pipeline };
        // SAFETY: samples was checked non-null; n_samples is caller-provided length.
        let samples = unsafe { std::slice::from_raw_parts(samples, n_samples) };
        // Legacy pipeline expects 16 kHz audio.
        if sample_rate != 16000 {
            return Err(PolyvoiceStatus::InvalidArg as c_int);
        }
        let result = pipeline
            .inner
            .run(samples, &pipeline.extractor, &mut pipeline.vad)
            .map_err(|e| match e {
                crate::pipeline::PipelineError::AudioTooLong { .. } => PolyvoiceStatus::AudioTooLong as c_int,
                _ => PolyvoiceStatus::Inference as c_int,
            })?;
        let json =
            serde_json::to_string(&result).map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        let len = json.len();
        let cstr = CString::new(json).map_err(|_| PolyvoiceStatus::Internal as c_int)?;
        let ptr_out = cstr.into_raw();
        // SAFETY: out_json and out_json_len were checked non-null above.
        unsafe {
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

/// Destroy a pipeline created by `polyvoice_pipeline_create`.
///
/// # Safety
/// `pipeline` must be a valid pointer returned by `polyvoice_pipeline_create`, or null.
/// Must be called exactly once per handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn polyvoice_pipeline_destroy(pipeline: *mut PolyvoicePipeline) {
    if !pipeline.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: pipeline is non-null and was created by Box::into_raw; caller destroys exactly once.
            unsafe {
                drop(Box::from_raw(pipeline));
            }
        }));
    }
}

/// Free a JSON string returned by `polyvoice_pipeline_run`.
///
/// # Safety
/// `p` must be a pointer returned by `polyvoice_pipeline_run`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn polyvoice_free_string(p: *mut c_char, _n: usize) {
    if !p.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: p is non-null and was created by CString::into_raw in polyvoice_pipeline_run.
            unsafe {
                drop(CString::from_raw(p));
            }
        }));
    }
}
