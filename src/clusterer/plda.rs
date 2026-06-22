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

// ---- Minimal NPY v1.0 reader (C-order, '<f4' or '<f8') ----
// Avoids an ndarray-npy dependency (which pins an older ndarray major) by parsing
// the small, fixed NPY format we control. Supports 1-D and 2-D little-endian
// float arrays in C (row-major) order — exactly what the offline PLDA dump emits.

fn read_file(path: &Path) -> Result<Vec<u8>, PldaError> {
    std::fs::read(path).map_err(|e| PldaError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })
}

/// Parse an NPY header, returning `(dtype, shape, data_offset)`.
fn parse_npy_header(bytes: &[u8], path: &Path) -> Result<(String, Vec<usize>, usize), PldaError> {
    let bad = |detail: &str| PldaError::Io {
        path: path.display().to_string(),
        detail: detail.to_string(),
    };
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(bad("not an NPY file"));
    }
    let major = bytes[6];
    // v1.0: 2-byte header len; v2.0+: 4-byte. We emit v1.0 but accept both.
    let (header_len, header_start) = if major >= 2 {
        let l = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (l, 12usize)
    } else {
        let l = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (l, 10usize)
    };
    let header = std::str::from_utf8(
        bytes
            .get(header_start..header_start + header_len)
            .ok_or_else(|| bad("truncated NPY header"))?,
    )
    .map_err(|_| bad("non-utf8 NPY header"))?;

    let descr = extract_between(header, "'descr':", ',')
        .or_else(|| extract_between(header, "\"descr\":", ','))
        .ok_or_else(|| bad("no descr in NPY header"))?
        .trim()
        .trim_matches(['\'', '"', ' '])
        .to_string();

    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return Err(bad("fortran-order NPY unsupported"));
    }

    let shape_str = extract_between(header, "'shape':", ')')
        .or_else(|| extract_between(header, "\"shape\":", ')'))
        .ok_or_else(|| bad("no shape in NPY header"))?;
    let shape: Vec<usize> = shape_str
        .trim_start_matches([' ', '('])
        .split(',')
        .filter_map(|s| {
            let t = s.trim();
            if t.is_empty() { None } else { t.parse().ok() }
        })
        .collect();

    Ok((descr, shape, header_start + header_len))
}

/// Slice of `s` after `key` up to (not including) the next `end` char.
fn extract_between(s: &str, key: &str, end: char) -> Option<String> {
    let start = s.find(key)? + key.len();
    let rest = &s[start..];
    let stop = rest.find(end)?;
    Some(rest[..stop].to_string())
}

fn read_npy_flat(path: &Path) -> Result<(Vec<f64>, Vec<usize>), PldaError> {
    let bytes = read_file(path)?;
    let (descr, shape, offset) = parse_npy_header(&bytes, path)?;
    let data = &bytes[offset..];
    let values: Vec<f64> = match descr.as_str() {
        "<f8" => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
            .collect(),
        "<f4" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap_or([0; 4])) as f64)
            .collect(),
        "<i8" => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])) as f64)
            .collect(),
        other => {
            return Err(PldaError::Io {
                path: path.display().to_string(),
                detail: format!("unsupported NPY dtype {other} (expected <f4, <f8 or <i8)"),
            });
        }
    };
    Ok((values, shape))
}

fn read_npy_1d(path: &Path) -> Result<Array1<f64>, PldaError> {
    let (values, _shape) = read_npy_flat(path)?;
    Ok(Array1::from(values))
}

fn read_npy_2d(path: &Path) -> Result<Array2<f64>, PldaError> {
    let (values, shape) = read_npy_flat(path)?;
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
}
