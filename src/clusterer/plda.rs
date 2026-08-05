//! PLDA feature transform for VBx clustering.
//!
//! Maps 256-d L2-normalized WeSpeaker embeddings into the diagonalized PLDA space
//! (128-d) that [`crate::clusterer::vbx`] scores in, and exposes the per-dimension
//! across-class eigenvalues (`phi`). The expensive diagonalization (a generalized
//! symmetric eigenproblem) is precomputed offline, so the runtime transform here
//! is pure `ndarray` (centering + matmul + L2-norm) — no eigendecomposition, no
//! BLAS/LAPACK, wasm32-clean.
//!
//! The transform recipe is ported from the Apache-2.0 `avencera/speakrs` crate
//! (`src/clustering/plda.rs`); attribution retained per Apache-2.0.

use ndarray::{Array1, Array2, ArrayView2, s};
use std::path::Path;

/// Errors loading or applying a [`PldaModel`].
#[derive(Debug, thiserror::Error)]
pub enum PldaError {
    #[error("plda param io error on {path}: {detail}")]
    Io { path: String, detail: String },
    #[error("plda param {name} has wrong shape: expected {expected}, got {actual}")]
    Shape {
        name: &'static str,
        expected: String,
        actual: String,
    },
}

/// Precomputed PLDA transform parameters.
#[derive(Debug, Clone)]
pub struct PldaModel {
    mean1: Array1<f64>,
    mean2: Array1<f64>,
    lda: Array2<f64>,
    mu: Array1<f64>,
    transform: Array2<f64>,
    phi: Array1<f64>,
}

impl PldaModel {
    /// Load from a directory of `.npy` files: `plda_mean1`, `plda_mean2`,
    /// `plda_lda`, `plda_mu`, `plda_transform`, `plda_phi_computed`. The latter
    /// two are the offline-precomputed diagonalization (no eigendecomposition at
    /// runtime).
    pub fn from_dir(dir: &Path) -> Result<Self, PldaError> {
        let mean1 = read_npy_1d(&dir.join("plda_mean1.npy"))?;
        let mean2 = read_npy_1d(&dir.join("plda_mean2.npy"))?;
        let lda = read_npy_2d(&dir.join("plda_lda.npy"))?;
        let mu = read_npy_1d(&dir.join("plda_mu.npy"))?;
        let transform = read_npy_2d(&dir.join("plda_transform.npy"))?;
        let phi = read_npy_1d(&dir.join("plda_phi_computed.npy"))?;
        Ok(Self {
            mean1,
            mean2,
            lda,
            mu,
            transform,
            phi,
        })
    }

    /// Per-dimension across-class eigenvalues consumed by VBx.
    pub fn phi(&self) -> Array1<f32> {
        self.phi.mapv(|v| v as f32)
    }

    /// Transform a batch of `(N, 256)` embeddings into `(N, lda_dim)` PLDA features.
    pub fn transform(&self, embeddings: &ArrayView2<f32>, lda_dim: usize) -> Array2<f32> {
        let emb = embeddings.mapv(|v| v as f64);
        let xvec = self.xvec_transform(&emb.view());
        self.plda_transform(&xvec.view(), lda_dim)
            .mapv(|v| v as f32)
    }

    /// WeSpeaker/Kaldi x-vector preprocessing: center, L2-norm, LDA-project,
    /// center, L2-norm (each norm step rescaled by sqrt of the working dim).
    fn xvec_transform(&self, embeddings: &ArrayView2<f64>) -> Array2<f64> {
        let centered = embeddings - &self.mean1;
        let normalized = l2_normalize_rows(&centered.view());
        let scaled = normalized * (self.lda.nrows() as f64).sqrt();
        let projected = scaled.dot(&self.lda);
        let centered_projected = projected - &self.mean2;
        l2_normalize_rows(&centered_projected.view()) * (self.lda.ncols() as f64).sqrt()
    }

    /// Project centered x-vectors onto the diagonalizing transform.
    fn plda_transform(&self, embeddings: &ArrayView2<f64>, lda_dim: usize) -> Array2<f64> {
        let lda_dim = lda_dim.min(self.transform.nrows());
        let centered = embeddings - &self.mu;
        centered.dot(&self.transform.slice(s![..lda_dim, ..]).t())
    }
}

/// Row-wise L2 normalization (f64).
fn l2_normalize_rows(embeddings: &ArrayView2<f64>) -> Array2<f64> {
    let mut out = embeddings.to_owned();
    for mut row in out.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }
    out
}

// ---- NPY parameter loading ----
// The NPY parsing itself lives in `crate::utils::npy` (shared with the AS-norm
// cohort loader); these wrappers only reshape into ndarray and translate
// errors into the public `PldaError` surface.

impl From<crate::utils::npy::NpyError> for PldaError {
    fn from(e: crate::utils::npy::NpyError) -> Self {
        match e {
            crate::utils::npy::NpyError::Io { path, detail } => PldaError::Io { path, detail },
        }
    }
}

fn read_npy_1d(path: &Path) -> Result<Array1<f64>, PldaError> {
    let (values, _shape) = crate::utils::npy::read_npy_flat(path)?;
    Ok(Array1::from(values))
}

fn read_npy_2d(path: &Path) -> Result<Array2<f64>, PldaError> {
    let (values, shape) = crate::utils::npy::read_npy_flat(path)?;
    if shape.len() != 2 {
        return Err(PldaError::Shape {
            name: "matrix",
            expected: "2-D".to_string(),
            actual: format!("{shape:?}"),
        });
    }
    Array2::from_shape_vec((shape[0], shape[1]), values).map_err(|e| PldaError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    /// Parity against the speakrs Python fixtures. Opt-in: set
    /// `POLYVOICE_VBX_FIXTURES` to a dir holding both the precomputed PLDA params
    /// (plda_mean1/mean2/lda/mu/transform/phi_computed.npy) and the fixtures
    /// (pipeline_train_embeddings.npy, pipeline_plda_phi.npy,
    /// pipeline_plda_features.npy). Skipped when unset.
    #[test]
    fn plda_transform_matches_python_fixture() {
        let Ok(dir) = std::env::var("POLYVOICE_VBX_FIXTURES") else {
            eprintln!("skip: POLYVOICE_VBX_FIXTURES unset");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let plda = PldaModel::from_dir(&dir).expect("load plda");
        let emb = read_npy_2d(&dir.join("pipeline_train_embeddings.npy")).unwrap();
        let exp_phi = read_npy_1d(&dir.join("pipeline_plda_phi.npy")).unwrap();
        let exp_feat = read_npy_2d(&dir.join("pipeline_plda_features.npy")).unwrap();

        let phi = plda.phi();
        for (a, b) in phi.iter().zip(exp_phi.iter()) {
            assert!((*a as f64 - *b).abs() < 1e-3, "phi mismatch: {a} vs {b}");
        }

        let emb_f32 = emb.mapv(|v| v as f32);
        let feat = plda.transform(&emb_f32.view(), 128);
        // Eigenvector sign is arbitrary per column — align before comparing.
        for col in 0..feat.ncols() {
            let dot: f64 = feat
                .column(col)
                .iter()
                .zip(exp_feat.column(col).iter())
                .map(|(a, b)| *a as f64 * *b)
                .sum();
            let sign = if dot < 0.0 { -1.0f32 } else { 1.0f32 };
            for (a, b) in feat.column(col).iter().zip(exp_feat.column(col).iter()) {
                assert!(
                    (*a * sign - *b as f32).abs() < 5e-3,
                    "feature mismatch at col {col}: {a} vs {b}"
                );
            }
        }
    }

    /// End-to-end VBx parity: AHC seed + PLDA features + phi → cluster_vbx must
    /// reproduce the Python reference responsibilities and prior. Opt-in on
    /// `POLYVOICE_VBX_FIXTURES`. This pins the full variational loop, not just PLDA.
    #[test]
    fn cluster_vbx_matches_python_fixture() {
        let Ok(dir) = std::env::var("POLYVOICE_VBX_FIXTURES") else {
            eprintln!("skip: POLYVOICE_VBX_FIXTURES unset");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let ahc = read_npy_1d(&dir.join("pipeline_ahc_clusters.npy")).unwrap();
        let features = read_npy_2d(&dir.join("pipeline_plda_features.npy")).unwrap();
        let phi = read_npy_1d(&dir.join("pipeline_plda_phi.npy")).unwrap();
        let exp_gamma = read_npy_2d(&dir.join("pipeline_vbx_gamma.npy")).unwrap();
        let exp_pi = read_npy_1d(&dir.join("pipeline_vbx_pi.npy")).unwrap();

        let ahc_labels: Vec<usize> = ahc.iter().map(|v| *v as usize).collect();
        let features = features.mapv(|v| v as f32);
        let phi = phi.mapv(|v| v as f32);
        let (gamma, pi) = crate::clusterer::vbx::cluster_vbx(
            &ahc_labels,
            &features.view(),
            &phi.view(),
            &crate::clusterer::vbx::VbxConfig::default(),
        );

        for (a, b) in gamma.iter().zip(exp_gamma.iter()) {
            assert!((*a as f64 - *b).abs() < 1e-4, "gamma mismatch: {a} vs {b}");
        }
        for (a, b) in pi.iter().zip(exp_pi.iter()) {
            assert!((*a as f64 - *b).abs() < 1e-5, "pi mismatch: {a} vs {b}");
        }
    }

    /// Directory of the checked-in VBx PLDA fixtures (256-d → 128-d).
    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/vbx-plda")
    }

    /// Write a minimal NPY v1.0 file (C-order) with the given dtype/shape/payload.
    fn write_npy_v1(
        dir: &std::path::Path,
        name: &str,
        descr: &str,
        shape: &[usize],
        data: &[u8],
    ) -> std::path::PathBuf {
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape[0])
        } else {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", dims.join(", "))
        };
        let header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        let path = dir.join(name);
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    fn f8_bytes(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn fixture_model_loads_and_transforms() {
        let plda = PldaModel::from_dir(&fixture_dir()).expect("load fixture plda");
        let phi = plda.phi();
        assert_eq!(phi.len(), 128);
        assert!(
            phi.iter().all(|v| v.is_finite() && *v > 0.0),
            "across-class eigenvalues must be positive and finite"
        );

        // Deterministic, distinct pseudo-embeddings (no RNG needed).
        let emb = Array2::from_shape_fn((3, 256), |(r, c)| ((r * 256 + c) % 17) as f32 * 0.1 - 0.8);
        let feat = plda.transform(&emb.view(), 128);
        assert_eq!(feat.dim(), (3, 128));
        assert!(feat.iter().all(|v| v.is_finite()));
        // The transform is a pure function of its input.
        let feat2 = plda.transform(&emb.view(), 128);
        assert_eq!(feat, feat2);
        // Distinct inputs must not collapse to the same feature row.
        assert_ne!(feat.row(0), feat.row(1));
    }

    #[test]
    fn transform_clamps_oversized_lda_dim() {
        let plda = PldaModel::from_dir(&fixture_dir()).expect("load fixture plda");
        let emb = Array2::from_shape_fn((2, 256), |(r, c)| (r * c) as f32 * 0.01);
        let feat = plda.transform(&emb.view(), 999);
        assert_eq!(feat.ncols(), 128, "lda_dim clamps to the transform rows");
    }

    #[test]
    fn l2_normalize_rows_scales_and_keeps_zero_rows() {
        let m = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let out = l2_normalize_rows(&m.view());
        let norm0: f64 = out.row(0).dot(&out.row(0)).sqrt();
        assert!((norm0 - 1.0).abs() < 1e-12);
        assert!((out[[0, 0]] - 0.6).abs() < 1e-12);
        // Zero row has no direction; it is left as zeros rather than NaN.
        assert!(out.row(1).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn read_npy_2d_rejects_non_2d_shape() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = write_npy_v1(
            tmp.path(),
            "v.npy",
            "<f8",
            &[3],
            &f8_bytes(&[1.0, 2.0, 3.0]),
        );
        let err = read_npy_2d(&path).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("wrong shape"), "{msg}");
        match err {
            PldaError::Shape {
                name,
                expected,
                actual,
            } => {
                assert_eq!(name, "matrix");
                assert_eq!(expected, "2-D");
                assert!(actual.contains('3'), "{actual}");
            }
            other => panic!("expected a shape error, got {other:?}"),
        }
    }

    #[test]
    fn read_npy_2d_rejects_data_shape_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Header claims (2, 2) but only three values follow.
        let path = write_npy_v1(
            tmp.path(),
            "m.npy",
            "<f8",
            &[2, 2],
            &f8_bytes(&[1.0, 2.0, 3.0]),
        );
        let err = read_npy_2d(&path).unwrap_err();
        assert!(matches!(err, PldaError::Io { .. }), "{err:?}");
    }

    #[test]
    fn from_dir_reports_first_missing_param() {
        let tmp = tempfile::TempDir::new().unwrap();
        let err = PldaModel::from_dir(tmp.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("plda_mean1.npy"), "{msg}");
    }
}
