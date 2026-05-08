//! M6b — FFI smoke tests for ABI v2.

#![cfg(feature = "ffi")]

use polyvoice::ffi::{
    PolyvoicePipeline, PolyvoiceProfile, polyvoice_pipeline_create,
    polyvoice_pipeline_destroy,
};
use std::ptr;

#[test]
#[ignore = "requires cached Balanced ONNX bundle"]
fn ffi_create_destroy_balanced_round_trip() {
    let mut handle: *mut PolyvoicePipeline = ptr::null_mut();
    // SAFETY: handle is non-null, null cache dir uses default registry path.
    let rc = unsafe {
        polyvoice_pipeline_create(PolyvoiceProfile::Balanced, ptr::null(), &mut handle)
    };
    assert_eq!(rc, 0, "create should succeed when ONNX is cached");
    assert!(!handle.is_null());
    // SAFETY: handle was returned by polyvoice_pipeline_create and is non-null.
    unsafe { polyvoice_pipeline_destroy(handle) };
}

#[test]
fn ffi_create_invalid_profile_arg_does_not_panic() {
    let handle: *mut PolyvoicePipeline = ptr::null_mut();
    // Intentionally pass null out_handle — must not panic.
    // SAFETY: null out_handle is the condition under test; function must handle it gracefully.
    let rc = unsafe {
        polyvoice_pipeline_create(PolyvoiceProfile::Mobile, ptr::null(), ptr::null_mut())
    };
    assert_ne!(rc, 0);
    assert!(handle.is_null());
}
