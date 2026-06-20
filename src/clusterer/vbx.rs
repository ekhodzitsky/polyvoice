//! VBx clustering — Variational Bayes HMM over PLDA-transformed embeddings.
//!
//! Refines an over-segmented agglomerative seed with variational inference and a
//! per-speaker prior that drives unused speakers to zero, so the speaker count is
//! determined automatically rather than fixed by a global threshold. This is the
//! structural lever against over-clustering that a fixed cosine cutoff cannot
//! reach. Pure `ndarray` f64 math (no ONNX, wasm32-clean); the PLDA preprocessing
//! that produces the diagonalized features + per-dimension eigenvalues lives in
//! [`crate::clusterer::plda`].
//!
//! The variational-inference loop (M-step / E-step / ELBO / prior-driven speaker
//! pruning) and the AHC→responsibility seeding are ported from the Apache-2.0
//! `avencera/speakrs` crate (`src/clustering/vbx.rs`), which in turn implements
//! the diagonalized-PLDA VBx of Landini et al., "Bayesian HMM clustering of
//! x-vector sequences (VBx) in speaker diarization" (Computer Speech & Language,
//! 2022). Attribution retained per Apache-2.0.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Tunable VBx hyperparameters.
///
/// `fa` (acoustic scale) and `fb` (speaker regularization) trade off how strongly
/// the acoustics pull frames into speakers versus the prior pulling speakers to
/// the origin; `init_smoothing` softens the one-hot AHC seed before inference.
#[derive(Debug, Clone, Copy)]
pub struct VbxConfig {
    pub fa: f64,
    pub fb: f64,
    pub max_iters: usize,
    pub epsilon: f64,
    pub init_smoothing: f64,
}

impl Default for VbxConfig {
    fn default() -> Self {
        // Starting point from the speakrs WeSpeaker recipe; retune on dev, never test.
        Self {
            fa: 0.07,
            fb: 0.8,
            max_iters: 20,
            epsilon: 1e-4,
            init_smoothing: 7.0,
        }
    }
}

/// Numerically stable log-sum-exp over a 1-D view.
fn logsumexp_f64(values: &ArrayView1<f64>) -> f64 {
    let max = values.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    if max.is_infinite() {
        return max;
    }
    let sum_exp = values.mapv(|x| (x - max).exp()).sum();
    max + sum_exp.ln()
}

/// Run VBx variational inference.
///
/// `features` is `(T, D)` PLDA-transformed, `phi` is the `D` per-dimension
/// across-class eigenvalues, `gamma_init` is the `(T, K)` responsibility seed.
/// Returns the refined `(T, K)` responsibilities and the `K` speaker prior;
/// unused speakers end with a prior near zero (auto speaker count). All inner
/// math is f64 to match the numpy reference precision.
pub fn vbx(
    features: &ArrayView2<f32>,
    phi: &ArrayView1<f32>,
    gamma_init: &Array2<f32>,
    config: &VbxConfig,
) -> (Array2<f32>, Array1<f32>) {
    let (n_samples, dim) = features.dim();
    let n_speakers = gamma_init.ncols();
    let fa = config.fa;
    let fb = config.fb;
    let fa_over_fb = fa / fb;

    let features_f64 = features.mapv(|v| v as f64);
    let phi_f64: Array1<f64> = phi.mapv(|v| v as f64);

    let mut gamma = gamma_init.mapv(|v| v as f64);
    let mut pi = Array1::from_elem(n_speakers, 1.0 / n_speakers as f64);

    // Per-frame constant G = -0.5 * (sum(X^2) + D*ln(2*pi)).
    let frame_constants: Array1<f64> = features_f64
        .rows()
        .into_iter()
        .map(|row| -0.5 * (row.dot(&row) + dim as f64 * (2.0 * std::f64::consts::PI).ln()))
        .collect();

    let phi_sqrt = phi_f64.mapv(f64::sqrt);

    // rho = X * sqrt(phi) (per-column broadcast).
    let mut rho = features_f64;
    for mut row in rho.rows_mut() {
        row *= &phi_sqrt;
    }

    let mut prev_elbo = f64::NEG_INFINITY;
    let mut scratch = Array1::<f64>::zeros(n_speakers);

    for iter in 0..config.max_iters {
        // M-step: per-speaker precision invL and mean alpha (diagonal PLDA → no inverse).
        let n_k: Array1<f64> = gamma.sum_axis(Axis(0));
        let mut inv_l = Array2::zeros((n_speakers, dim));
        let mut alpha = Array2::zeros((n_speakers, dim));

        for speaker_idx in 0..n_speakers {
            for dim_idx in 0..dim {
                inv_l[[speaker_idx, dim_idx]] =
                    1.0 / (1.0 + fa_over_fb * n_k[speaker_idx] * phi_f64[dim_idx]);
            }
            let mut f_k = Array1::<f64>::zeros(dim);
            for sample_idx in 0..n_samples {
                f_k.scaled_add(gamma[[sample_idx, speaker_idx]], &rho.row(sample_idx));
            }
            for dim_idx in 0..dim {
                alpha[[speaker_idx, dim_idx]] =
                    fa_over_fb * inv_l[[speaker_idx, dim_idx]] * f_k[dim_idx];
            }
        }

        // E-step: PLDA log-likelihood-ratio emission per frame per speaker.
        let mut log_p = Array2::<f64>::zeros((n_samples, n_speakers));
        for sample_idx in 0..n_samples {
            for speaker_idx in 0..n_speakers {
                let rho_dot_alpha: f64 = rho.row(sample_idx).dot(&alpha.row(speaker_idx));
                let penalty: f64 = (0..dim)
                    .map(|dim_idx| {
                        (inv_l[[speaker_idx, dim_idx]]
                            + alpha[[speaker_idx, dim_idx]] * alpha[[speaker_idx, dim_idx]])
                            * phi_f64[dim_idx]
                    })
                    .sum();
                log_p[[sample_idx, speaker_idx]] =
                    fa * (rho_dot_alpha - 0.5 * penalty + frame_constants[sample_idx]);
            }
        }

        // GMM-style responsibility update with the pi prior.
        let lpi: Array1<f64> = pi.mapv(|p| (p + 1e-8).ln());
        let mut log_p_x = Array1::<f64>::zeros(n_samples);
        for sample_idx in 0..n_samples {
            scratch.assign(&log_p.row(sample_idx));
            scratch += &lpi;
            log_p_x[sample_idx] = logsumexp_f64(&scratch.view());
        }
        for sample_idx in 0..n_samples {
            for speaker_idx in 0..n_speakers {
                gamma[[sample_idx, speaker_idx]] =
                    (log_p[[sample_idx, speaker_idx]] + lpi[speaker_idx] - log_p_x[sample_idx])
                        .exp();
            }
        }

        // Update the prior; empty speakers shrink toward zero (auto count).
        pi = gamma.sum_axis(Axis(0));
        let pi_sum = pi.sum();
        pi /= pi_sum;

        // ELBO = sum(log_p_x) + Fb*0.5*sum(ln(invL) - invL - alpha^2 + 1).
        let log_px_sum: f64 = log_p_x.sum();
        let reg: f64 = inv_l
            .iter()
            .zip(alpha.iter())
            .map(|(&il, &a)| il.ln() - il - a * a + 1.0)
            .sum();
        let elbo = log_px_sum + fb * 0.5 * reg;

        if iter > 0 && elbo - prev_elbo < config.epsilon {
            break;
        }
        prev_elbo = elbo;
    }

    (gamma.mapv(|v| v as f32), pi.mapv(|v| v as f32))
}

/// Seed responsibilities from over-segmented AHC labels, then run [`vbx`].
pub fn cluster_vbx(
    ahc_labels: &[usize],
    features: &ArrayView2<f32>,
    phi: &ArrayView1<f32>,
    config: &VbxConfig,
) -> (Array2<f32>, Array1<f32>) {
    let gamma_init = build_gamma_init(ahc_labels, config.init_smoothing);
    vbx(features, phi, &gamma_init, config)
}

/// Build the `(T, K)` responsibility seed: a one-hot of the AHC labels, optionally
/// row-softmaxed by `smoothing` so the seed is near-hard but not degenerate.
fn build_gamma_init(labels: &[usize], smoothing: f64) -> Array2<f32> {
    let num_samples = labels.len();
    let num_speakers = labels.iter().copied().max().unwrap_or(0) + 1;
    let mut gamma = Array2::<f32>::zeros((num_samples, num_speakers));
    for (row, &label) in labels.iter().enumerate() {
        gamma[[row, label]] = 1.0;
    }
    if smoothing < 0.0 {
        return gamma;
    }
    let smoothing_f32 = smoothing as f32;
    for mut row in gamma.rows_mut() {
        row *= smoothing_f32;
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|v| (v - max).exp());
        let denom = row.sum();
        row /= denom;
    }
    gamma
}

/// Hard speaker label per frame: the argmax of the final responsibilities,
/// renumbered to a compact `0..K` so empty speakers leave no gaps.
pub fn hard_labels(gamma: &Array2<f32>) -> Vec<usize> {
    let raw: Vec<usize> = gamma
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect();
    // Compact: map used speaker ids to 0..K in first-seen order.
    let mut remap: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for &r in &raw {
        let next = remap.len();
        remap.entry(r).or_insert(next);
    }
    raw.into_iter().map(|r| remap[&r]).collect()
}

use crate::clusterer::plda::PldaModel;
use crate::clusterer::{Clusterer, ClustererError};

/// VBx clusterer: PLDA-transform 256-d embeddings, seed with an over-segmented
/// AHC pass, then run VBx variational inference whose prior auto-determines the
/// speaker count. Implements the [`Clusterer`] trait; embeddings are assumed to
/// arrive in temporal order (the seed and inference treat them as a sequence).
pub struct VbxClusterer {
    plda: PldaModel,
    config: VbxConfig,
    /// Cosine-similarity threshold for the over-segmenting AHC seed.
    ahc_threshold: f32,
    max_speakers: usize,
    lda_dim: usize,
    /// Scale applied to the (L2-normalized) input embeddings before the PLDA
    /// transform, to restore the raw WeSpeaker magnitude the PLDA mean-centering
    /// expects (the pipeline embedder L2-normalizes by contract, discarding it).
    emb_scale: f32,
}

impl VbxClusterer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        plda: PldaModel,
        config: VbxConfig,
        ahc_threshold: f32,
        max_speakers: usize,
        lda_dim: usize,
        emb_scale: f32,
    ) -> Self {
        Self {
            plda,
            config,
            ahc_threshold,
            max_speakers: max_speakers.max(1),
            lda_dim,
            emb_scale,
        }
    }

    /// Construct from the `POLYVOICE_VBX_PLDA_DIR` env var (proof/dev wiring;
    /// shipped builds will resolve the PLDA params through the model registry).
    pub fn from_env(max_speakers: usize) -> Result<Self, ClustererError> {
        let dir = std::env::var("POLYVOICE_VBX_PLDA_DIR").map_err(|_| {
            ClustererError::AlgorithmFailed {
                detail: "POLYVOICE_VBX_PLDA_DIR not set".to_owned(),
            }
        })?;
        let plda = PldaModel::from_dir(std::path::Path::new(&dir)).map_err(|e| {
            ClustererError::AlgorithmFailed {
                detail: format!("load PLDA: {e}"),
            }
        })?;
        // Over-init seed threshold: higher cosine cutoff → more seed clusters that
        // VBx then prunes. Tuned on dev later; this is a reasonable over-init.
        let ahc_threshold = std::env::var("POLYVOICE_VBX_AHC_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        // fa=0.3 is the polyvoice dev-calibrated optimum (VbxConfig::default keeps
        // the upstream 0.07 the speakrs parity fixtures were generated with).
        let config = VbxConfig {
            fa: std::env::var("POLYVOICE_VBX_FA")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.3),
            fb: std::env::var("POLYVOICE_VBX_FB")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| VbxConfig::default().fb),
            ..VbxConfig::default()
        };
        let emb_scale = std::env::var("POLYVOICE_VBX_EMB_SCALE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4.88);
        Ok(Self::new(plda, config, ahc_threshold, max_speakers, 128, emb_scale))
    }
}

impl Clusterer for VbxClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }
        if embeddings.len() == 1 {
            return Ok(vec![0]);
        }
        let dim = embeddings[0].len();
        for (i, e) in embeddings.iter().enumerate() {
            if e.len() != dim {
                return Err(ClustererError::DimMismatch {
                    expected: dim,
                    actual: e.len(),
                    index: i,
                });
            }
        }
        // Stack into (N, dim) and PLDA-transform to (N, lda_dim).
        let n = embeddings.len();
        let mut flat = Vec::with_capacity(n * dim);
        for e in embeddings {
            flat.extend_from_slice(e);
        }
        let emb = Array2::from_shape_vec((n, dim), flat).map_err(|e| {
            ClustererError::AlgorithmFailed {
                detail: format!("embedding reshape: {e}"),
            }
        })?;
        // Restore the raw WeSpeaker magnitude the PLDA mean-centering expects.
        let emb = emb * self.emb_scale;
        let features = self.plda.transform(&emb.view(), self.lda_dim);
        let phi = self.plda.phi();

        // Over-segmenting AHC seed on the PLDA features (cosine linkage).
        let feat_vecs: Vec<Vec<f32>> =
            features.rows().into_iter().map(|r| r.to_vec()).collect();
        let ahc_labels = crate::ahc::agglomerative_cluster_max_clusters(
            &feat_vecs,
            self.ahc_threshold,
            self.max_speakers,
        );

        let (gamma, _pi) = cluster_vbx(&ahc_labels, &features.view(), &phi.view(), &self.config);
        Ok(hard_labels(&gamma))
    }

    fn max_clusters(&self) -> usize {
        self.max_speakers
    }

    /// PLDA mean-centering needs the original embedding scale, so VBx requires raw
    /// (non-L2-normalized) embeddings — L2-normalized input collapses the centered
    /// vectors toward `-mean1` and degenerates the transform.
    fn wants_raw_embeddings(&self) -> bool {
        true
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn two_clusters_with_vbx() {
        // Two well-separated clusters; VBx must keep them distinct from the seed.
        let features = array![
            [10.0, 0.0],
            [10.1, 0.1],
            [9.9, -0.1],
            [-10.0, 0.0],
            [-10.1, 0.1],
            [-9.9, -0.1],
        ];
        let phi = array![1.0, 1.0];
        let mut gamma_init = Array2::zeros((6, 2));
        for t in 0..3 {
            gamma_init[[t, 0]] = 0.999;
            gamma_init[[t, 1]] = 0.001;
        }
        for t in 3..6 {
            gamma_init[[t, 0]] = 0.001;
            gamma_init[[t, 1]] = 0.999;
        }
        let (gamma, _pi) = vbx(&features.view(), &phi.view(), &gamma_init, &VbxConfig::default());
        let labels = hard_labels(&gamma);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn gamma_init_is_smoothed_one_hot() {
        let gamma = build_gamma_init(&[0, 0, 1], 7.0);
        assert_eq!(gamma.dim(), (3, 2));
        assert!(gamma[[0, 0]] > gamma[[0, 1]]);
        assert!(gamma[[2, 1]] > gamma[[2, 0]]);
    }

    #[test]
    fn vbx_prunes_redundant_seed_speaker() {
        // One true cluster split across two seed speakers — VBx's prior should
        // collapse the hard labels to a single speaker (auto-count downward).
        let features = array![[5.0, 0.0], [5.1, 0.1], [4.9, -0.1], [5.05, 0.05]];
        let phi = array![1.0, 1.0];
        let mut gamma_init = Array2::zeros((4, 2));
        gamma_init[[0, 0]] = 0.99;
        gamma_init[[0, 1]] = 0.01;
        gamma_init[[1, 0]] = 0.99;
        gamma_init[[1, 1]] = 0.01;
        gamma_init[[2, 0]] = 0.01;
        gamma_init[[2, 1]] = 0.99;
        gamma_init[[3, 0]] = 0.01;
        gamma_init[[3, 1]] = 0.99;
        let cfg = VbxConfig::default();
        let (gamma, _pi) = vbx(&features.view(), &phi.view(), &gamma_init, &cfg);
        let labels = hard_labels(&gamma);
        let distinct: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(distinct.len(), 1, "one acoustic cluster must collapse to one speaker");
    }

    #[test]
    fn hard_labels_are_compact() {
        let gamma = array![[0.1, 0.9, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
        // Used columns are 1 and 2 → compacted to 0 and 1.
        let labels = hard_labels(&gamma);
        assert_eq!(labels, vec![0, 1, 1]);
    }
}
