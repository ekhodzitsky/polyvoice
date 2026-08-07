use super::*;

#[test]
fn inference_failed_display_includes_detail() {
    let err = EmbedderError::InferenceFailed {
        detail: "session crashed".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("ONNX inference failed"));
    assert!(msg.contains("session crashed"));
}

#[test]
fn resource_exhausted_display_includes_detail() {
    let err = EmbedderError::ResourceExhausted {
        detail: "all sessions busy".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("resource exhausted"));
    assert!(msg.contains("all sessions busy"));
}

#[test]
fn dim_mismatch_display_includes_both_dims() {
    let err = EmbedderError::DimMismatch {
        expected: 192,
        actual: 256,
    };
    let msg = format!("{err}");
    assert!(msg.contains("192"));
    assert!(msg.contains("256"));
}

#[test]
fn model_io_display_includes_path_and_detail() {
    let err = EmbedderError::ModelIo {
        path: std::path::PathBuf::from("/tmp/missing.onnx"),
        detail: "no such file".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("/tmp/missing.onnx"));
    assert!(msg.contains("no such file"));
}

#[test]
fn legacy_display_includes_message() {
    let err = EmbedderError::Legacy("old adapter broke".into());
    assert!(format!("{err}").contains("old adapter broke"));
}

#[test]
fn legacy_variant_with_pool_exhausted_string_classifies() {
    let err = EmbedderError::Legacy("onnx session pool exhausted".into());
    assert!(err.is_resource_exhausted());
}

#[test]
fn legacy_variant_without_marker_does_not_classify() {
    let err = EmbedderError::Legacy("some other failure".into());
    assert!(!err.is_resource_exhausted());
}

#[test]
fn inference_failed_without_marker_does_not_classify() {
    let err = EmbedderError::InferenceFailed {
        detail: "shape mismatch".into(),
    };
    assert!(!err.is_resource_exhausted());
}

#[test]
fn unrelated_variants_do_not_classify_as_exhausted() {
    let too_short = EmbedderError::AudioTooShort {
        actual_secs: 0.1,
        min_secs: 0.5,
    };
    assert!(!too_short.is_resource_exhausted());

    let io = EmbedderError::ModelIo {
        path: std::path::PathBuf::from("m.onnx"),
        detail: "pool exhausted on disk".into(),
    };
    assert!(
        !io.is_resource_exhausted(),
        "ModelIo must not substring-match the legacy marker"
    );
}
