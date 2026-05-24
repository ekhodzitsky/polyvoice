//! M6b — FFI smoke tests for ABI v3.

#![cfg(feature = "ffi")]

use polyvoice::ffi::{
    PolyvoicePipeline, PolyvoiceProfile, polyvoice_free_string, polyvoice_pipeline_create,
    polyvoice_pipeline_destroy, polyvoice_pipeline_run,
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
