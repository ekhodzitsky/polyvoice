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
    /// Self-loop probability of the speaker HMM. `0.0` runs the GMM-style update
    /// (no temporal model, matches the speakrs reference fixtures); `> 0.0`
    /// enables the canonical forward-backward VBx whose self-loop-favoring
    /// transitions smooth labels over time and curb over-clustering.
    ///
    /// **GMM-VBx** is exactly `loop_prob = 0`: use it when embeddings are not a
    /// contiguous frame sequence (dense windowed extract, non-ordered batches).
    pub loop_prob: f64,
}

impl Default for VbxConfig {
    fn default() -> Self {
        // Starting point from the speakrs WeSpeaker recipe; retune on dev, never test.
        // loop_prob defaults to 0.0 (GMM) so the speakrs parity fixtures hold.
        // The shipped clusterer tuning (fa=0.3, loop_prob=0.9) lives in
        // VbxClustererConfig::default — the single source of truth for it.
        Self {
            fa: 0.07,
            fb: 0.8,
            max_iters: 20,
            epsilon: 1e-4,
            init_smoothing: 7.0,
            loop_prob: 0.0,
        }
    }
}

impl VbxConfig {
    /// Explicit GMM-VBx (`loop_prob = 0`): independent-frame updates, no HMM
    /// self-loop. Prefer this when embeddings come from non-contiguous windows.
    pub fn gmm(mut self) -> Self {
        self.loop_prob = 0.0;
        self
    }

    /// Canonical forward-backward HMM-VBx with the given self-loop probability
    /// (clamped into `[0, 1]`).
    pub fn hmm(mut self, loop_prob: f64) -> Self {
        self.loop_prob = loop_prob.clamp(0.0, 1.0);
        self
    }

    /// True when this config runs GMM-VBx (no temporal self-loop).
    pub fn is_gmm(&self) -> bool {
        self.loop_prob <= 0.0
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

/// Forward-backward over the speaker HMM with the self-loop-favoring transition
/// matrix `tr[i][j] = (i==j)*loop_prob + (1-loop_prob)*pi[j]`. Returns the
/// `(T, S)` occupation posteriors, the total log-likelihood, and the log forward
/// / log backward matrices (the latter two feed the prior update).
fn forward_backward(
    log_p: &Array2<f64>,
    pi: &Array1<f64>,
    loop_prob: f64,
) -> (Array2<f64>, f64, Array2<f64>, Array2<f64>) {
    let (t_len, s) = log_p.dim();
    let eps = 1e-8;
    let mut ltr = Array2::<f64>::zeros((s, s));
    for i in 0..s {
        for j in 0..s {
            let tr = if i == j { loop_prob } else { 0.0 } + (1.0 - loop_prob) * pi[j];
            ltr[[i, j]] = (tr + eps).ln();
        }
    }

    let mut lfw = Array2::<f64>::from_elem((t_len, s), f64::NEG_INFINITY);
    let mut lbw = Array2::<f64>::from_elem((t_len, s), f64::NEG_INFINITY);
    let mut tmp = Array1::<f64>::zeros(s);

    for j in 0..s {
        lfw[[0, j]] = log_p[[0, j]] + (pi[j] + eps).ln();
    }
    for t in 1..t_len {
        for j in 0..s {
            for i in 0..s {
                tmp[i] = lfw[[t - 1, i]] + ltr[[i, j]];
            }
            lfw[[t, j]] = log_p[[t, j]] + logsumexp_f64(&tmp.view());
        }
    }

    for j in 0..s {
        lbw[[t_len - 1, j]] = 0.0;
    }
    for t in (0..t_len.saturating_sub(1)).rev() {
        for i in 0..s {
            for j in 0..s {
                tmp[j] = ltr[[i, j]] + log_p[[t + 1, j]] + lbw[[t + 1, j]];
            }
            lbw[[t, i]] = logsumexp_f64(&tmp.view());
        }
    }

    let tll = logsumexp_f64(&lfw.row(t_len - 1));
    let mut gamma = Array2::<f64>::zeros((t_len, s));
    for t in 0..t_len {
        for j in 0..s {
            gamma[[t, j]] = (lfw[[t, j]] + lbw[[t, j]] - tll).exp();
        }
    }
    (gamma, tll, lfw, lbw)
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

        // Responsibility + prior update: temporal HMM forward-backward when
        // loop_prob > 0, else the GMM-style independent-frame update (parity path).
        let log_px_sum: f64 = if config.loop_prob > 0.0 {
            let (g, tll, log_a, log_b) = forward_backward(&log_p, &pi, config.loop_prob);
            gamma = g;
            // Prior / speaker-count update (24): empty speakers shrink toward zero.
            let one_minus_loop = 1.0 - config.loop_prob;
            let mut accum = Array1::<f64>::zeros(n_speakers);
            for t in 0..n_samples.saturating_sub(1) {
                let lse_fw = logsumexp_f64(&log_a.row(t));
                for s in 0..n_speakers {
                    accum[s] += (lse_fw + log_p[[t + 1, s]] + log_b[[t + 1, s]] - tll).exp();
                }
            }
            for s in 0..n_speakers {
                pi[s] = gamma[[0, s]] + one_minus_loop * pi[s] * accum[s];
            }
            let pi_sum = pi.sum();
            pi /= pi_sum;
            tll
        } else {
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
            log_p_x.sum()
        };

        // ELBO = log_pX + Fb*0.5*sum(ln(invL) - invL - alpha^2 + 1).
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

/// Full configuration for [`VbxClusterer`]: the VBx inference hyperparameters
/// plus the clusterer-level knobs (AHC seed threshold, embedding rescale,
/// short-segment filter, cAHC-ASC stop).
///
/// [`VbxClustererConfig::default`] is the dev-calibrated production tuning and
/// the single source of truth for it — [`VbxClusterer::from_dir`] uses it
/// verbatim, so the public constructor is deterministic in its arguments. The
/// nested [`VbxConfig`] here deliberately differs from [`VbxConfig::default`]:
/// `fa = 0.3` and `loop_prob = 0.9` are the polyvoice dev optimum, while the
/// bare `VbxConfig` default keeps the upstream speakrs values (`fa = 0.07`,
/// GMM mode) pinned by the parity fixtures.
#[derive(Debug, Clone, Copy)]
pub struct VbxClustererConfig {
    /// VBx variational-inference hyperparameters.
    pub vbx: VbxConfig,
    /// Cosine-similarity threshold for the over-segmenting AHC seed. A higher
    /// cutoff yields more seed clusters that the VBx prior then prunes.
    pub ahc_threshold: f32,
    /// Scale applied to the (L2-normalized) input embeddings before the PLDA
    /// transform, restoring the raw WeSpeaker magnitude the PLDA
    /// mean-centering expects.
    pub emb_scale: f32,
    /// Exclude embeddings shorter than this many seconds from AHC/VB; reassign
    /// afterward by nearest PLDA-feature centroid. `0.0` disables filtering.
    pub min_embedding_secs: f64,
    /// cAHC-ASC stop for the AHC seed: refuse to merge two clusters that both
    /// already have at least this many members. `0` disables.
    pub ahc_established_min_members: usize,
}

impl Default for VbxClustererConfig {
    fn default() -> Self {
        Self {
            vbx: VbxConfig {
                fa: 0.3,
                loop_prob: 0.9,
                ..VbxConfig::default()
            },
            ahc_threshold: 0.5,
            emb_scale: 4.88,
            // cVBx short-segment recipe.
            min_embedding_secs: 1.6,
            ahc_established_min_members: 0,
        }
    }
}

impl VbxClustererConfig {
    /// Explicit opt-in for offline tuning: overlay the
    /// `POLYVOICE_VBX_{FA,FB,LOOP_PROB,AHC_THRESHOLD,EMB_SCALE,MIN_EMB_SECS,AHC_ASC_MEMBERS}`
    /// env vars onto [`Self::default`]. Missing or malformed values keep the
    /// default. Nothing in the library calls this implicitly — production
    /// construction (`from_dir`, the pipeline builder) is env-free.
    pub fn from_env() -> Self {
        fn parse_or<T: std::str::FromStr>(name: &str, fallback: T) -> T {
            std::env::var(name)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(fallback)
        }
        let d = Self::default();
        Self {
            vbx: VbxConfig {
                fa: parse_or("POLYVOICE_VBX_FA", d.vbx.fa),
                fb: parse_or("POLYVOICE_VBX_FB", d.vbx.fb),
                loop_prob: parse_or("POLYVOICE_VBX_LOOP_PROB", d.vbx.loop_prob),
                ..d.vbx
            },
            ahc_threshold: parse_or("POLYVOICE_VBX_AHC_THRESHOLD", d.ahc_threshold),
            emb_scale: parse_or("POLYVOICE_VBX_EMB_SCALE", d.emb_scale),
            min_embedding_secs: parse_or("POLYVOICE_VBX_MIN_EMB_SECS", d.min_embedding_secs),
            ahc_established_min_members: parse_or(
                "POLYVOICE_VBX_AHC_ASC_MEMBERS",
                d.ahc_established_min_members,
            ),
        }
    }
}

/// VBx clusterer: PLDA-transform 256-d embeddings, seed with an over-segmented
/// AHC pass, then run VBx variational inference whose prior auto-determines the
/// speaker count. Implements the [`Clusterer`] trait; embeddings are assumed to
/// arrive in temporal order (the seed and inference treat them as a sequence)
/// unless GMM mode (`loop_prob = 0`) is selected for windowed extract.
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
    /// Exclude embeddings shorter than this many seconds from AHC/VB; reassign
    /// afterward by nearest PLDA-feature centroid. `0.0` disables filtering.
    /// Default for production is 1.6 s (cVBx short-segment recipe).
    min_embedding_secs: f64,
    /// cAHC-ASC stop for the AHC seed: refuse to merge two clusters that both
    /// already have at least this many members. `0` disables.
    ahc_established_min_members: usize,
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
            min_embedding_secs: 0.0,
            ahc_established_min_members: 0,
        }
    }

    /// Exclude embeddings shorter than `secs` from AHC/VB (reassign after).
    /// Pass `0.0` to disable. The cVBx recipe starts at 1.6 s.
    pub fn with_min_embedding_secs(mut self, secs: f64) -> Self {
        self.min_embedding_secs = secs.max(0.0);
        self
    }

    /// Enable cAHC-ASC on the AHC seed: stop before merging two clusters that
    /// both already have ≥ `min_members` members. `0` disables.
    pub fn with_ahc_established_min_members(mut self, min_members: usize) -> Self {
        self.ahc_established_min_members = min_members;
        self
    }

    /// Force GMM-VBx (`loop_prob = 0`) or HMM-VBx with the current loop_prob.
    pub fn with_gmm_mode(mut self, gmm: bool) -> Self {
        if gmm {
            self.config = self.config.gmm();
        }
        self
    }

    /// Override the full VBx hyperparameter block.
    pub fn with_config(mut self, config: VbxConfig) -> Self {
        self.config = config;
        self
    }

    /// Construct from an explicit PLDA directory — the precomputed
    /// `plda_{transform,phi_computed,mean1,mean2,lda,mu}.npy` set produced by
    /// `scripts/build-vbx-plda.py`. This is the path shipped builds use (the
    /// directory is resolved through the model registry / `--vbx-plda-dir`).
    ///
    /// Uses [`VbxClustererConfig::default`] verbatim (fa=0.3, loop_prob=0.9
    /// i.e. the canonical forward-backward VBx, ahc_threshold=0.5,
    /// emb_scale=4.88, min_embedding_secs=1.6) — one global set, never branched
    /// on dataset name, and no env reads. For explicit overrides use
    /// [`Self::from_dir_with_config`] (e.g. with [`VbxClustererConfig::from_env`]
    /// for offline tuning).
    pub fn from_dir(
        plda_dir: &std::path::Path,
        max_speakers: usize,
    ) -> Result<Self, ClustererError> {
        Self::from_dir_with_config(plda_dir, max_speakers, VbxClustererConfig::default())
    }

    /// [`Self::from_dir`] with an explicit configuration in place of the
    /// dev-calibrated defaults.
    pub fn from_dir_with_config(
        plda_dir: &std::path::Path,
        max_speakers: usize,
        config: VbxClustererConfig,
    ) -> Result<Self, ClustererError> {
        let plda = PldaModel::from_dir(plda_dir)?;
        Ok(Self::new(
            plda,
            config.vbx,
            config.ahc_threshold,
            max_speakers,
            128,
            config.emb_scale,
        )
        .with_min_embedding_secs(config.min_embedding_secs)
        .with_ahc_established_min_members(config.ahc_established_min_members))
    }

    /// Ensure the six PLDA `.npy` files via the model registry (SHA-256 verified
    /// download into the registry cache) and construct from that directory.
    ///
    /// Used by the pipeline builder when neither `--vbx-plda-dir` nor
    /// `POLYVOICE_VBX_PLDA_DIR` is set. Signatures are optional for these
    /// entries until a release engineer signs them.
    #[cfg(feature = "download")]
    pub fn from_registry(
        registry: &crate::models::ModelRegistry,
        max_speakers: usize,
    ) -> Result<Self, ClustererError> {
        let dir = registry
            .ensure_vbx_plda_dir()
            .map_err(|e| ClustererError::AlgorithmFailed {
                detail: format!("ensure VBx PLDA via model registry: {e}"),
            })?;
        Self::from_dir(&dir, max_speakers)
    }

    /// When embeddings come from dense non-contiguous windows, force GMM-VBx
    /// (the HMM self-loop assumption is invalid there). Windowed extract always
    /// wins over the configured `loop_prob`; run HMM-VBx by embedding
    /// contiguous per-segment units instead.
    pub fn auto_gmm_for_windowed(mut self, windowed: bool) -> Self {
        if windowed {
            self.config = self.config.gmm();
        }
        self
    }

    fn cluster_inner(
        &self,
        embeddings: &[Vec<f32>],
        durations_secs: &[f64],
    ) -> Result<Vec<usize>, ClustererError> {
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

        // Optional short-segment filter: cluster only long enough embeddings.
        // Durations are used only when they align 1:1 with embeddings.
        let (kept, short) =
            if self.min_embedding_secs > 0.0 && durations_secs.len() == embeddings.len() {
                crate::clusterer::partition_by_min_duration(durations_secs, self.min_embedding_secs)
            } else {
                ((0..embeddings.len()).collect(), Vec::new())
            };

        let kept_embs: Vec<&[f32]> = kept.iter().map(|&i| embeddings[i].as_slice()).collect();
        let labels_kept = self.cluster_kept(&kept_embs)?;

        if short.is_empty() {
            // `partition_by_min_duration` partitions 0..n into kept ∪ short,
            // so an empty `short` means `kept` is exactly 0..n and the kept
            // labels already cover every embedding in order.
            debug_assert!(
                kept.len() == embeddings.len() && kept.iter().enumerate().all(|(i, &k)| i == k)
            );
            return Ok(labels_kept);
        }

        // PLDA features for reassignment of short embeddings.
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
        let emb = emb * self.emb_scale;
        let features = self.plda.transform(&emb.view(), self.lda_dim);
        let feat_vecs: Vec<Vec<f32>> = features.rows().into_iter().map(|r| r.to_vec()).collect();
        Ok(crate::clusterer::reassign_short_by_features(
            &feat_vecs,
            &kept,
            &labels_kept,
            &short,
        ))
    }

    fn cluster_kept(&self, kept_embs: &[&[f32]]) -> Result<Vec<usize>, ClustererError> {
        let n = kept_embs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![0]);
        }
        let dim = kept_embs[0].len();
        let mut flat = Vec::with_capacity(n * dim);
        for e in kept_embs {
            flat.extend_from_slice(e);
        }
        let emb = Array2::from_shape_vec((n, dim), flat).map_err(|e| {
            ClustererError::AlgorithmFailed {
                detail: format!("embedding reshape: {e}"),
            }
        })?;
        let emb = emb * self.emb_scale;
        let features = self.plda.transform(&emb.view(), self.lda_dim);
        let phi = self.plda.phi();

        let feat_vecs: Vec<Vec<f32>> = features.rows().into_iter().map(|r| r.to_vec()).collect();
        let stop = if self.ahc_established_min_members > 0 {
            crate::ahc::AscStop::MinMembers(self.ahc_established_min_members)
        } else {
            crate::ahc::AscStop::Off
        };
        let ahc_labels = crate::ahc::agglomerative_cluster_asc(
            &feat_vecs,
            self.ahc_threshold,
            self.max_speakers,
            stop,
            None,
        );

        let (gamma, _pi) = cluster_vbx(&ahc_labels, &features.view(), &phi.view(), &self.config);
        Ok(hard_labels(&gamma))
    }
}

impl Clusterer for VbxClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        self.cluster_inner(embeddings, &[])
    }

    fn cluster_with_durations(
        &self,
        embeddings: &[Vec<f32>],
        durations_secs: &[f64],
    ) -> Result<Vec<usize>, ClustererError> {
        self.cluster_inner(embeddings, durations_secs)
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
#[path = "vbx_tests.rs"]
mod tests;
