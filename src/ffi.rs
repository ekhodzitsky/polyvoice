//! C FFI bindings for polyvoice.
//!
//! This module provides a C API so that polyvoice can be called from other
//! languages (C, C++, Python via ctypes, etc.).
//!
//! # Safety
//!
//! All functions in this module use raw pointers. Callers must ensure:
//! - Pointers are valid and properly aligned
//! - Buffers have the claimed lengths
//! - Returned pointers are freed with the corresponding free function
//!
//! # Example (C)
//!
//! ```c
//! #include <polyvoice.h>
//!
//! PolyvoiceDiarizer* d = polyvoice_diarizer_new(0.5f, 64);
//! PolyvoiceResult* r = polyvoice_diarizer_run(d, samples, sample_count);
//! for (size_t i = 0; i < r->num_turns; i++) {
//!     printf("%s: %.2f - %.2f\n", r->turns[i].speaker, r->turns[i].start, r->turns[i].end);
//! }
//! polyvoice_result_free(r);
//! polyvoice_diarizer_free(d);
//! ```

use crate::cluster::SpeakerCluster;
use crate::embedding::{DummyExtractor, EmbeddingExtractor};
use crate::types::DiarizationConfig;
use std::ffi::{c_char, c_float, CString};
use std::os::raw::c_int;
use std::ptr;

/// Opaque handle to a diarizer instance.
pub struct PolyvoiceDiarizer {
    config: DiarizationConfig,
    cluster: SpeakerCluster,
    extractor: DummyExtractor,
}

/// A single speaker turn returned to C.
#[repr(C)]
pub struct PolyvoiceTurn {
    pub speaker: *mut c_char,
    pub start: c_float,
    pub end: c_float,
}

/// Result of diarization returned to C.
#[repr(C)]
pub struct PolyvoiceResult {
    pub turns: *mut PolyvoiceTurn,
    pub num_turns: usize,
}

/// Create a new diarizer with the given threshold and max speakers.
///
/// # Safety
/// Returns a pointer that must be freed with `polyvoice_diarizer_free`.
#[unsafe(no_mangle)] // SAFETY: C ABI symbol export required for FFI linkage.
pub unsafe extern "C" fn polyvoice_diarizer_new( // SAFETY: C ABI entry point with raw pointer return; caller must free result.
    threshold: c_float,
    max_speakers: c_int,
) -> *mut PolyvoiceDiarizer {
    let config = DiarizationConfig {
        threshold,
        max_speakers: max_speakers as usize,
        ..Default::default()
    };
    let diarizer = PolyvoiceDiarizer {
        config,
        cluster: SpeakerCluster::new(config),
        extractor: DummyExtractor::new(256),
    };
    Box::into_raw(Box::new(diarizer))
}

/// Run diarization on a buffer of mono f32 samples at 16 kHz.
///
/// # Safety
/// - `diarizer` must be a valid pointer returned by `polyvoice_diarizer_new`.
/// - `samples` must point to at least `sample_count` valid f32 values.
///
/// Returns a `PolyvoiceResult` that must be freed with `polyvoice_result_free`.
/// Returns NULL on error.
#[unsafe(no_mangle)] // SAFETY: C ABI symbol export required for FFI linkage.
pub unsafe extern "C" fn polyvoice_diarizer_run( // SAFETY: C ABI entry point dereferencing raw pointers from caller.
    diarizer: *mut PolyvoiceDiarizer,
    samples: *const c_float,
    sample_count: usize,
) -> *mut PolyvoiceResult {
    if diarizer.is_null() || samples.is_null() || sample_count == 0 {
        return ptr::null_mut();
    }
    let d = unsafe { // SAFETY: we checked diarizer is non-null above.
        &mut *diarizer
    };
    let audio = unsafe { // SAFETY: we checked samples is non-null and sample_count > 0.
        std::slice::from_raw_parts(samples, sample_count)
    };

    let window = d.config.window_samples();
    let hop = d.config.hop_samples();
    if audio.len() < window {
        return ptr::null_mut();
    }

    let mut turns: Vec<PolyvoiceTurn> = Vec::new();
    let mut start = 0usize;
    while start + window <= audio.len() {
        let chunk = &audio[start..start + window];
        match d.extractor.extract(chunk, &d.config) {
            Ok(emb) => {
                let (speaker, _conf) = d.cluster.assign(&emb);
                let speaker_cstr = match CString::new(format!("SPEAKER_{:02}", speaker.0)) {
                    Ok(s) => s,
                    Err(_) => {
                        // Free already allocated strings before returning NULL.
                        for turn in &turns {
                            if !turn.speaker.is_null() {
                                unsafe { // SAFETY: speaker was created by CString::into_raw.
                                    let _ = CString::from_raw(turn.speaker);
                                }
                            }
                        }
                        return ptr::null_mut();
                    }
                };
                turns.push(PolyvoiceTurn {
                    speaker: speaker_cstr.into_raw(),
                    start: (start as f32 / d.config.sample_rate.get() as f32),
                    end: ((start + window) as f32 / d.config.sample_rate.get() as f32),
                });
            }
            Err(_) => {
                // Skip window on error.
            }
        }
        start += hop;
    }

    let num_turns = turns.len();
    let mut boxed = turns.into_boxed_slice();
    let turns_ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed); // Ownership transferred to C.

    let result = PolyvoiceResult {
        turns: turns_ptr,
        num_turns,
    };
    Box::into_raw(Box::new(result))
}

/// Free a diarizer instance.
///
/// # Safety
/// `diarizer` must be a valid pointer returned by `polyvoice_diarizer_new` or NULL.
#[unsafe(no_mangle)] // SAFETY: C ABI symbol export required for FFI linkage.
pub unsafe extern "C" fn polyvoice_diarizer_free( // SAFETY: C ABI entry point freeing raw pointer previously created by Box::into_raw.
    diarizer: *mut PolyvoiceDiarizer,
) {
    if !diarizer.is_null() {
        unsafe { // SAFETY: we checked diarizer is non-null; it was created by Box::into_raw.
            let _ = Box::from_raw(diarizer);
        }
    }
}

/// Free a result returned by `polyvoice_diarizer_run`.
///
/// # Safety
/// `result` must be a valid pointer returned by `polyvoice_diarizer_run` or NULL.
#[unsafe(no_mangle)] // SAFETY: C ABI symbol export required for FFI linkage.
pub unsafe extern "C" fn polyvoice_result_free( // SAFETY: C ABI entry point freeing raw pointer and its nested allocations.
    result: *mut PolyvoiceResult,
) {
    if result.is_null() {
        return;
    }
    unsafe { // SAFETY: we checked result is non-null; it was created by Box::into_raw.
        let r = &mut *result;
        if !r.turns.is_null() {
            // SAFETY: turns was created by Vec::into_boxed_slice + forget.
            let slice_ptr = std::ptr::slice_from_raw_parts_mut(r.turns, r.num_turns);
            let turns = Box::from_raw(slice_ptr);
            for turn in turns.iter() {
                if !turn.speaker.is_null() {
                    // SAFETY: speaker was created by CString::into_raw.
                    let _ = CString::from_raw(turn.speaker);
                }
            }
        }
        let _ = Box::from_raw(result);
    }
}

/// Return the library version as a static C string.
#[unsafe(no_mangle)] // SAFETY: C ABI symbol export required for FFI linkage.
pub extern "C" fn polyvoice_version() -> *const c_char { // SAFETY: C ABI entry point returning a static nul-terminated string.
    // SAFETY: c-string literal has static lifetime and is nul-terminated.
    c"0.4.2".as_ptr() as *const c_char
}
