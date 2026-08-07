use super::*;

fn mismatched_embeddings() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0], // dimension 4 at index 2
    ]
}

#[test]
fn ahc_rejects_dim_mismatch() {
    let c = AhcClusterer::default();
    let err = c
        .cluster(&mismatched_embeddings())
        .expect_err("mismatched dims must fail");
    assert!(matches!(
        err,
        ClustererError::DimMismatch {
            expected: 3,
            actual: 4,
            index: 2,
        }
    ));
}

#[test]
fn kmeans_rejects_dim_mismatch() {
    let c = KmeansClusterer::default();
    let err = c
        .cluster(&mismatched_embeddings())
        .expect_err("mismatched dims must fail");
    assert!(matches!(
        err,
        ClustererError::DimMismatch {
            expected: 3,
            actual: 4,
            index: 2,
        }
    ));
}

#[cfg(feature = "spectral")]
#[test]
fn nme_sc_rejects_dim_mismatch() {
    let c = NmeScClusterer::default();
    let err = c
        .cluster(&mismatched_embeddings())
        .expect_err("mismatched dims must fail");
    assert!(matches!(
        err,
        ClustererError::DimMismatch {
            expected: 3,
            actual: 4,
            index: 2,
        }
    ));
}
