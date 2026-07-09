//! M6b — FFI smoke tests for ABI v3.

#![cfg(feature = "ffi")]

use polyvoice::ffi::{
    PolyvoiceFormat, PolyvoicePipeline, PolyvoiceProfile, polyvoice_free_string,
    polyvoice_pipeline_create, polyvoice_pipeline_destroy, polyvoice_pipeline_run,
    polyvoice_pipeline_run_format,
};
use std::ptr;

#[test]
#[ignore = "requires cached Balanced ONNX bundle"]
fn ffi_create_destroy_balanced_round_trip() {
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    // SAFETY: handle is non-null, null cache dir uses default registry path.
    let rc = unsafe {
        polyvoice_pipeline_create(PolyvoiceProfile::Balanced as i32, ptr::null(), &mut handle)
    };
    assert_eq!(rc, 0, "create should succeed when ONNX is cached");
    assert!(!handle.is_null());
    // SAFETY: handle was returned by polyvoice_pipeline_create and is non-null.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn ffi_create_null_out_handle_returns_error() {
    let handle: *mut PolyvoicePipeline = ptr::null_mut();
    // Intentionally pass null out_handle — must not panic.
    // SAFETY: null out_handle is the condition under test; function must handle it gracefully.
    let rc = unsafe {
        polyvoice_pipeline_create(
            PolyvoiceProfile::Mobile as i32,
            ptr::null(),
            ptr::null_mut(),
        )
    };
    assert_ne!(rc, 0);
    assert!(handle.is_null());
}

#[test]
fn ffi_create_invalid_profile_returns_invalid_arg() {
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    // Profile values 0 and 1 are valid; 42 is not.
    // SAFETY: handle is non-null.
    let rc = unsafe { polyvoice_pipeline_create(42, ptr::null(), &mut handle) };
    assert_ne!(rc, 0, "invalid profile must return non-zero error code");
    assert!(handle.is_null(), "handle must remain null on error");
}

#[test]
#[ignore = "requires cached Balanced ONNX bundle"]
fn ffi_run_on_silence_returns_valid_json() {
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    // SAFETY: handle is non-null, null cache dir uses default registry path.
    let rc = unsafe {
        polyvoice_pipeline_create(PolyvoiceProfile::Balanced as i32, ptr::null(), &mut handle)
    };
    assert_eq!(rc, 0);
    assert!(!handle.is_null());

    let samples: Vec<f32> = vec![0.0f32; 16000 * 2];
    let mut json: *mut std::os::raw::c_char = ptr::null_mut();
    let mut json_len: usize = 0;

    // SAFETY: handle is valid, samples is valid, json pointers are non-null.
    let rc = unsafe {
        polyvoice_pipeline_run(
            handle,
            samples.as_ptr(),
            samples.len(),
            16000,
            &mut json,
            &mut json_len,
        )
    };
    assert_eq!(rc, 0, "run on silence should succeed");
    assert!(!json.is_null());

    // SAFETY: json was returned by polyvoice_pipeline_run and is non-null.
    let json_str = unsafe {
        std::ffi::CStr::from_ptr(json)
            .to_string_lossy()
            .into_owned()
    };
    assert!(json_str.contains("num_speakers"));
    assert!(json_str.contains("turns"));

    // SAFETY: json was returned by polyvoice_pipeline_run.
    unsafe { polyvoice_free_string(json, json_len) };
    // SAFETY: handle was returned by polyvoice_pipeline_create and is non-null.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn ffi_run_format_null_pipeline_returns_invalid_arg() {
    let samples: Vec<f32> = vec![0.0f32; 16000];
    let mut out: *mut std::os::raw::c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: null pipeline is the condition under test; function must reject it.
    let rc = unsafe {
        polyvoice_pipeline_run_format(
            ptr::null_mut(),
            samples.as_ptr(),
            samples.len(),
            16000,
            PolyvoiceFormat::Json as i32,
            &mut out,
            &mut out_len,
        )
    };
    assert_ne!(rc, 0);
    assert!(out.is_null());
}

#[test]
fn ffi_run_format_unknown_format_returns_invalid_arg() {
    let samples: Vec<f32> = vec![0.0f32; 16000];
    let mut out: *mut std::os::raw::c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // Format 42 is not a polyvoice_format_t value; rejected before pipeline use,
    // so a null pipeline never gets dereferenced either way.
    // SAFETY: all out-pointers are valid; the invalid format is the condition under test.
    let rc = unsafe {
        polyvoice_pipeline_run_format(
            ptr::null_mut(),
            samples.as_ptr(),
            samples.len(),
            16000,
            42,
            &mut out,
            &mut out_len,
        )
    };
    assert_eq!(rc, 1, "unknown format must return InvalidArg");
    assert!(out.is_null());
}

#[test]
#[ignore = "requires cached Balanced ONNX bundle"]
fn ffi_run_format_renders_every_format() {
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    // SAFETY: handle is non-null, null cache dir uses default registry path.
    let rc = unsafe {
        polyvoice_pipeline_create(PolyvoiceProfile::Balanced as i32, ptr::null(), &mut handle)
    };
    assert_eq!(rc, 0);
    assert!(!handle.is_null());

    let samples: Vec<f32> = vec![0.0f32; 16000 * 2];
    for (format, marker) in [
        (PolyvoiceFormat::Json as i32, "num_speakers"),
        (PolyvoiceFormat::Rttm as i32, ""),
        (PolyvoiceFormat::Srt as i32, ""),
        (PolyvoiceFormat::Vtt as i32, "WEBVTT"),
        (PolyvoiceFormat::Txt as i32, ""),
    ] {
        let mut out: *mut std::os::raw::c_char = ptr::null_mut();
        let mut out_len: usize = 0;
        // SAFETY: handle is valid, samples is valid, out pointers are non-null.
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
        assert_eq!(rc, 0, "run_format({format}) on silence should succeed");
        assert!(!out.is_null());
        // SAFETY: out was returned by polyvoice_pipeline_run_format and is non-null.
        let s = unsafe { std::ffi::CStr::from_ptr(out).to_string_lossy().into_owned() };
        assert_eq!(s.len(), out_len);
        if !marker.is_empty() {
            assert!(s.contains(marker), "format {format} missing marker {marker}");
        }
        // SAFETY: out was returned by polyvoice_pipeline_run_format.
        unsafe { polyvoice_free_string(out, out_len) };
    }

    // With a VALID handle, an unknown format is the only path to InvalidArg —
    // this (unlike the null-pipeline tests) actually exercises the format check.
    let mut out: *mut std::os::raw::c_char = ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: handle is valid, samples is valid, out pointers are non-null.
    let rc = unsafe {
        polyvoice_pipeline_run_format(
            handle,
            samples.as_ptr(),
            samples.len(),
            16000,
            42,
            &mut out,
            &mut out_len,
        )
    };
    assert_eq!(rc, 1, "unknown format with a valid handle must return InvalidArg");
    assert!(out.is_null());

    // SAFETY: handle was returned by polyvoice_pipeline_create and is non-null.
    unsafe { polyvoice_pipeline_destroy(handle) };
}
