//! AS-norm (adaptive symmetric score normalization) for cosine AHC.
//!
//! Raw cosine scores are uncalibrated across utterances and recording
//! domains: a merge threshold tuned on one domain drifts on another. AS-norm
//! re-centers and rescales each pairwise score against an **imposter cohort**
//! (speaker embeddings that stand in for "everyone else"):
//!
//! ```text
//! score'(a, b) = 0.5 * ((s - mean_a) / std_a + (s - mean_b) / std_b)
//! ```
//!
//! where `s = cosine(a, b)` and `mean_x` / `std_x` are the mean and standard
//! deviation of `x`'s top-N cohort scores. Per-embedding cohort stats are
//! precomputed once per `cluster` run (O(n · cohort)), then every pairwise
//! score — including the AHC post-merge centroid refresh — is a table lookup
//! plus one cosine.
//!
//! Scope: fixed-threshold AHC only. The auto-threshold path derives its
//! threshold from the raw matrix's gap structure and is left on raw cosine;
//! the VBx/PLDA backend emits its own scores and is not decorated here.

use crate::ahc::AhcScorer;
use crate::clusterer::{Clusterer, ClustererError};
use crate::utils::{cosine_similarity, l2_normalize};
use std::path::{Path, PathBuf};

/// Default cohort size (top-N) for the per-embedding normalization stats.
pub const DEFAULT_AS_NORM_TOP_N: usize = 100;

/// Model-registry id of the shipped imposter cohort (VoxConverse-dev
/// speakers; never the evaluation/test split). Resolved via
/// `ModelRegistry::ensure` when no explicit cohort path is configured.
pub const DEFAULT_ASNORM_COHORT_MODEL_ID: &str = "asnorm_cohort_voxdev";

/// Where the imposter cohort comes from.
#[derive(Clone, Debug)]
pub enum CohortSource {
    /// Explicit local `.npy` file: shape `(N, D)`, dtype `'<f4'`, C-order.
    Path(PathBuf),
    /// Model-registry id, resolved (and downloaded) at pipeline build time.
    ModelId(String),
}

/// AS-norm configuration for the fixed-threshold AHC clusterer.
#[derive(Clone, Debug)]
pub struct AsNormConfig {
    /// Number of top cohort scores per embedding used for mean/std estimation.
    pub top_n: usize,
    /// Cohort provenance.
    pub cohort: CohortSource,
}

/// Errors loading an [`AsNormCohort`].
#[derive(Debug, thiserror::Error)]
pub enum AsNormError {
    #[error("as-norm cohort io error on {path}: {detail}")]
    Io { path: String, detail: String },
}

impl From<crate::utils::npy::NpyError> for AsNormError {
    fn from(e: crate::utils::npy::NpyError) -> Self {
        match e {
            crate::utils::npy::NpyError::Io { path, detail } => AsNormError::Io { path, detail },
        }
    }
}

/// Imposter cohort: a set of speaker embeddings the clustered embeddings are
/// scored against. Rows are L2-normalized on load (the pipeline L2-normalizes
/// embeddings before clustering, so the cohort must live on the same scale).
#[derive(Clone, Debug)]
pub struct AsNormCohort {
    rows: Vec<Vec<f32>>,
}

impl AsNormCohort {
    /// Build from in-memory rows; every row is L2-normalized.
    pub fn from_rows(rows: Vec<Vec<f32>>) -> Self {
        let mut rows = rows;
        for row in &mut rows {
            l2_normalize(row);
        }
        Self { rows }
    }

    /// Load from a `.npy` file of shape `(N, D)`, dtype `'<f4'`, C-order.
    pub fn from_npy(path: &Path) -> Result<Self, AsNormError> {
        let (values, _rows, cols) = crate::utils::npy::read_npy_f32_2d(path)?;
        let rows = values.chunks_exact(cols).map(<[f32]>::to_vec).collect();
        Ok(Self::from_rows(rows))
    }

    /// The cohort rows (L2-normalized).
    pub fn rows(&self) -> &[Vec<f32>] {
        &self.rows
    }

    /// Embedding dimension of the cohort, `None` when empty.
    pub fn dim(&self) -> Option<usize> {
        self.rows.first().map(Vec::len)
    }
}

/// Mean/std of the `top_n` highest cosine scores of `embedding` against the
/// cohort, plus the number of cohort evaluations performed.
///
/// Degenerate inputs fall back to the identity normalizer `(0.0, 1.0)`, under
/// which the z-score collapses to the raw score: an empty cohort carries no
/// calibration information, and a zero-dispersion top-N set (single-row or
/// constant cohort) has no usable scale, so raw passthrough is safer than an
/// epsilon-divided blow-up. The dispersion floor is 1e-6 — loose enough to
/// absorb f32 summation noise on constant score sets, far below any real
/// cohort's top-N spread (O(0.01)).
fn top_score_stats(cohort: &[Vec<f32>], embedding: &[f32], top_n: usize) -> (f32, f32, usize) {
    if cohort.is_empty() {
        return (0.0, 1.0, 0);
    }
    let mut scores: Vec<f32> = cohort
        .iter()
        .map(|c| cosine_similarity(embedding, c))
        .collect();
    let evals = scores.len();
    scores.sort_by(|a, b| b.total_cmp(a));
    let k = top_n.max(1).min(scores.len());
    let top = &scores[..k];
    let mean = top.iter().sum::<f32>() / k as f32;
    let var = top.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / k as f32;
    let std = var.sqrt();
    if std < 1e-6 {
        (0.0, 1.0, evals)
    } else {
        (mean, std, evals)
    }
}

/// Per-run AS-norm scorer over the AHC scoring seam.
///
/// Cohort stats for every embedding of the run are computed once up front
/// (`O(n · cohort)`); `score` then only reads the cached stats of each
/// cluster's dominant member and never touches the cohort again.
pub(crate) struct AsNormScorer {
    /// Per-embedding `(mean, std)` of top-N cohort scores.
    stats: Vec<(f32, f32)>,
    /// Cohort cosine evaluations performed at construction — test-visible so
    /// the stats-once-per-run guarantee is asserted, not assumed.
    #[allow(dead_code)] // read by tests only
    cohort_evals: usize,
}

impl AsNormScorer {
    pub(crate) fn new(cohort: &AsNormCohort, embeddings: &[Vec<f32>], top_n: usize) -> Self {
        let mut cohort_evals = 0;
        let stats = embeddings
            .iter()
            .map(|e| {
                let (mean, std, evals) = top_score_stats(cohort.rows(), e, top_n);
                cohort_evals += evals;
                (mean, std)
            })
            .collect();
        Self {
            stats,
            cohort_evals,
        }
    }

    #[cfg(test)]
    pub(crate) fn cohort_evals(&self) -> usize {
        self.cohort_evals
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> &[(f32, f32)] {
        &self.stats
    }
}

impl AhcScorer for AsNormScorer {
    fn score(
        &self,
        centroid_a: &[f32],
        member_a: usize,
        centroid_b: &[f32],
        member_b: usize,
    ) -> f32 {
        let s = cosine_similarity(centroid_a, centroid_b);
        // Member indices always come from the AHC core (in-range by
        // construction); `get` keeps that contract defensive.
        let (mean_a, std_a) = self.stats.get(member_a).copied().unwrap_or((0.0, 1.0));
        let (mean_b, std_b) = self.stats.get(member_b).copied().unwrap_or((0.0, 1.0));
        0.5 * ((s - mean_a) / std_a + (s - mean_b) / std_b)
    }
}

/// Fixed-threshold AHC clusterer whose pairwise scores are AS-norm
/// z-normalized against an imposter cohort.
///
/// Same shell as [`crate::clusterer::AhcClusterer::with_threshold`], with the
/// scoring seam decorated: the merge threshold applies to z-scores, not raw
/// cosines, so it must be chosen on the normalized scale (see the per-domain
/// profiles in [`crate::clusterer::domain`]).
pub struct AsNormClusterer {
    max_clusters: usize,
    threshold: f32,
    cohort: AsNormCohort,
    top_n: usize,
}

impl AsNormClusterer {
    /// `max_clusters == 0` means no ceiling, matching
    /// [`crate::clusterer::AhcClusterer::with_threshold`]. `top_n == 0` clamps
    /// to a single top cohort score per embedding.
    pub fn new(max_clusters: usize, threshold: f32, cohort: AsNormCohort, top_n: usize) -> Self {
        Self {
            max_clusters,
            threshold,
            cohort,
            top_n,
        }
    }
}

impl Clusterer for AsNormClusterer {
    fn cluster(&self, embeddings: &[Vec<f32>]) -> Result<Vec<usize>, ClustererError> {
        if embeddings.is_empty() {
            return Err(ClustererError::TooFewEmbeddings { actual: 0, min: 1 });
        }
        if embeddings.len() == 1 {
            return Ok(vec![0]);
        }
        super::uniform_dim(embeddings)?;
        if let Some(cohort_dim) = self.cohort.dim() {
            let expected = embeddings[0].len();
            if cohort_dim != expected {
                return Err(ClustererError::AlgorithmFailed {
                    detail: format!(
                        "as-norm cohort dim {cohort_dim} does not match embedding dim {expected}"
                    ),
                });
            }
        }
        let scorer = AsNormScorer::new(&self.cohort, embeddings, self.top_n);
        Ok(crate::ahc::agglomerative_cluster_scored(
            embeddings,
            self.threshold,
            self.max_clusters,
            &scorer,
        ))
    }

    fn max_clusters(&self) -> usize {
        self.max_clusters
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::clusterer::AhcClusterer;

    /// Deterministic PRNG (xorshift64) so the synthetic scenes are stable
    /// across runs and platforms without an RNG dependency.
    struct XorShift(u64);

    impl XorShift {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        /// Uniform in [-1, 1).
        fn next_f32(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    fn noise(rng: &mut XorShift, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.next_f32()).collect()
    }

    fn unit(v: &[f32]) -> Vec<f32> {
        let mut out = v.to_vec();
        l2_normalize(&mut out);
        out
    }

    fn mean(v: &[f32]) -> f32 {
        v.iter().sum::<f32>() / v.len() as f32
    }

    fn var(v: &[f32]) -> f32 {
        let m = mean(v);
        v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32
    }

    /// Deterministic "two speakers + channel" scene. Each utterance is
    /// `normalize(center + gamma * channel + noise)` with `gamma` taken from
    /// `gammas`: a recording-channel component inflates the raw cosine of two
    /// high-gamma utterances regardless of speaker — the cross-utterance
    /// drift AS-norm exists to remove. The cohort exposes the channel
    /// direction (every row carries it), so per-utterance cohort stats track
    /// `gamma`.
    ///
    /// Returns `(cohort, speaker_a, speaker_b)` utterance sets.
    fn channel_scene(gammas: &[f32]) -> (AsNormCohort, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        const DIM: usize = 16;
        let mut rng = XorShift(0x9E3779B97F4A7C15);
        let basis = |axis: usize| {
            let mut v = vec![0.0f32; DIM];
            v[axis] = 1.0;
            v
        };
        let channel = basis(0);
        let center_a = basis(1);
        let center_b = basis(2);

        // Imposter cohort: 48 speakers near the channel direction.
        let cohort: Vec<Vec<f32>> = (0..48)
            .map(|_| {
                let mut v = noise(&mut rng, DIM);
                for (x, c) in v.iter_mut().zip(&channel) {
                    *x = 0.9 * *x + 0.6 * c;
                }
                unit(&v)
            })
            .collect();

        let utterances = |rng: &mut XorShift, center: &[f32]| {
            gammas
                .iter()
                .map(|&gamma| {
                    let mut v = noise(rng, DIM);
                    for d in 0..DIM {
                        v[d] = 0.15 * v[d] + center[d] + gamma * channel[d];
                    }
                    unit(&v)
                })
                .collect::<Vec<_>>()
        };
        let speaker_a = utterances(&mut rng, &center_a);
        let speaker_b = utterances(&mut rng, &center_b);
        (AsNormCohort::from_rows(cohort), speaker_a, speaker_b)
    }

    /// All same-speaker and cross-speaker scores under `score_pair`.
    fn same_cross_scores(
        a: &[Vec<f32>],
        b: &[Vec<f32>],
        mut score_pair: impl FnMut(&[f32], usize, &[f32], usize) -> f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut same = Vec::new();
        for spk in [a, b] {
            for i in 0..spk.len() {
                for j in (i + 1)..spk.len() {
                    same.push(score_pair(&spk[i], i, &spk[j], j));
                }
            }
        }
        let mut cross = Vec::new();
        for (i, ua) in a.iter().enumerate() {
            for (j, ub) in b.iter().enumerate() {
                // `i` indexes into speaker_a, `j` into speaker_b; the caller
                // maps them onto the joint embedding slice.
                cross.push(score_pair(ua, i, ub, a.len() + j));
            }
        }
        (same, cross)
    }

    #[test]
    fn as_norm_increases_same_vs_different_separation() {
        // Strong channel drift: raw cross-speaker scores overlap same-speaker
        // ones, so this scene exercises the normalization, not the geometry.
        let (cohort, speaker_a, speaker_b) = channel_scene(&[0.0, 0.7, 1.4, 2.1]);
        let all: Vec<Vec<f32>> = speaker_a.iter().chain(speaker_b.iter()).cloned().collect();
        let scorer = AsNormScorer::new(&cohort, &all, 20);

        let (same_raw, cross_raw) = same_cross_scores(&speaker_a, &speaker_b, |ua, _i, ub, _j| {
            cosine_similarity(ua, ub)
        });
        let (same_z, cross_z) = same_cross_scores(&speaker_a, &speaker_b, |ua, i, ub, j| {
            scorer.score(ua, i, ub, j)
        });

        // Scale-free separation: same/cross gap in units of cross-speaker std.
        let t_raw = (mean(&same_raw) - mean(&cross_raw)) / var(&cross_raw).sqrt();
        let t_z = (mean(&same_z) - mean(&cross_z)) / var(&cross_z).sqrt();
        assert!(
            t_z > t_raw,
            "as-norm must sharpen separation: t_raw={t_raw:.3} t_z={t_z:.3}"
        );

        // Scale-free same-speaker spread, relative to the squared gap.
        let gap_raw = (mean(&same_raw) - mean(&cross_raw)).powi(2);
        let gap_z = (mean(&same_z) - mean(&cross_z)).powi(2);
        let spread_raw = var(&same_raw) / gap_raw;
        let spread_z = var(&same_z) / gap_z;
        assert!(
            spread_z < spread_raw,
            "as-norm must reduce relative cross-utterance spread: raw={spread_raw:.3} z={spread_z:.3}"
        );
    }

    #[test]
    fn as_norm_clusterer_separates_channel_scene_at_z_threshold() {
        // Mild channel drift: the speakers stay z-separable, so the clusterer
        // must recover the two-speaker partition at a z-scale threshold.
        let (cohort, speaker_a, speaker_b) = channel_scene(&[0.0, 0.2, 0.4, 0.6]);
        let all: Vec<Vec<f32>> = speaker_a.iter().chain(speaker_b.iter()).cloned().collect();

        // Pick the threshold from the z-score gap itself, then assert the
        // partition: 2 clusters, one per speaker.
        let scorer = AsNormScorer::new(&cohort, &all, 20);
        let (same_z, cross_z) = same_cross_scores(&speaker_a, &speaker_b, |ua, i, ub, j| {
            scorer.score(ua, i, ub, j)
        });
        let min_same = same_z.iter().copied().fold(f32::INFINITY, f32::min);
        let max_cross = cross_z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            min_same > max_cross,
            "scene must be z-separable: min_same={min_same:.3} max_cross={max_cross:.3}"
        );

        let c = AsNormClusterer::new(0, (min_same + max_cross) / 2.0, cohort, 20);
        let labels = c.cluster(&all).unwrap();
        assert_eq!(labels, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn cohort_stats_computed_once_per_run() {
        let (cohort, speaker_a, speaker_b) = channel_scene(&[0.0, 0.7, 1.4, 2.1]);
        let all: Vec<Vec<f32>> = speaker_a.iter().chain(speaker_b.iter()).cloned().collect();
        let top_n = 10;
        let scorer = AsNormScorer::new(&cohort, &all, top_n);

        // Up front: exactly one cohort pass per embedding — O(n · cohort).
        let upfront = scorer.cohort_evals();
        assert_eq!(upfront, all.len() * cohort.rows().len());

        // Scoring any pair, any number of times, touches only cached stats:
        // no further cohort work, and results are a pure function of the pair.
        let first = scorer.score(&all[0], 0, &all[1], 1);
        for _ in 0..64 {
            assert_eq!(scorer.score(&all[0], 0, &all[1], 1), first);
        }
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                let _ = scorer.score(&all[i], i, &all[j], j);
            }
        }
        assert_eq!(
            scorer.cohort_evals(),
            upfront,
            "score() must never re-evaluate the cohort"
        );
    }

    #[test]
    fn top_n_clamps_to_cohort_size() {
        let mut rng = XorShift(42);
        let cohort = AsNormCohort::from_rows((0..5).map(|_| noise(&mut rng, 8)).collect());
        let emb = unit(&noise(&mut rng, 8));

        // top_n larger than the cohort uses the whole cohort, no panic.
        let (m_all, s_all, evals) = top_score_stats(cohort.rows(), &emb, 1000);
        assert_eq!(evals, 5);
        assert!(s_all > 0.0);

        // top-1 is degenerate (a single score has zero dispersion) → identity
        // normalizer by design.
        let (m1, s1, _) = top_score_stats(cohort.rows(), &emb, 1);
        assert_eq!((m1, s1), (0.0, 1.0));

        // top-2 stats track the two highest cohort scores exactly.
        let mut scores: Vec<f32> = cohort
            .rows()
            .iter()
            .map(|c| cosine_similarity(&emb, c))
            .collect();
        scores.sort_by(|a, b| b.total_cmp(a));
        let (m2, s2, _) = top_score_stats(cohort.rows(), &emb, 2);
        assert!((m2 - (scores[0] + scores[1]) / 2.0).abs() < 1e-6);
        assert!(s2 > 0.0);
        assert!(m_all <= scores[0]);
    }

    #[test]
    fn degenerate_cohorts_fall_back_to_raw_cosine() {
        let embeddings = vec![
            vec![1.0, 0.05, 0.0],
            vec![0.95, 0.0, 0.05],
            vec![0.0, 1.0, 0.0],
            vec![0.05, 0.95, 0.0],
        ];
        let plain = AhcClusterer::with_threshold(0, 0.5)
            .cluster(&embeddings)
            .unwrap();

        for (name, cohort) in [
            ("empty", AsNormCohort::from_rows(Vec::new())),
            (
                "single-row",
                AsNormCohort::from_rows(vec![vec![1.0, 0.0, 0.0]]),
            ),
            (
                "constant (std=0)",
                AsNormCohort::from_rows(vec![vec![0.3, 0.9, 0.1]; 16]),
            ),
        ] {
            let scorer = AsNormScorer::new(&cohort, &embeddings, 10);
            assert!(
                scorer.stats().iter().all(|&(m, s)| m == 0.0 && s == 1.0),
                "{name}: degenerate cohort must yield the identity normalizer"
            );
            let raw = cosine_similarity(&embeddings[0], &embeddings[1]);
            let z = scorer.score(&embeddings[0], 0, &embeddings[1], 1);
            assert!(
                (z - raw).abs() < 1e-6,
                "{name}: degenerate cohort must pass the raw score through"
            );
            let c = AsNormClusterer::new(0, 0.5, cohort, 10);
            assert_eq!(
                c.cluster(&embeddings).unwrap(),
                plain,
                "{name}: labels must match raw-cosine AHC"
            );
        }
    }

    /// Minimal NPY v1.0 writer for cohort round-trip tests.
    fn write_npy_f4_2d(path: &Path, rows: usize, cols: usize, data: &[f32]) {
        let dict =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
        let pad = (64 - (10 + dict.len() + 1) % 64) % 64;
        let header = format!("{dict}{}{}", " ".repeat(pad), "\n");
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, &bytes).unwrap();
    }

    #[test]
    fn cohort_npy_round_trip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("cohort.npy");
        let data = [3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.6, 0.0, 0.8, 0.0];
        write_npy_f4_2d(&path, 3, 4, &data);

        let cohort = AsNormCohort::from_npy(&path).unwrap();
        assert_eq!(cohort.rows().len(), 3);
        assert_eq!(cohort.dim(), Some(4));
        // Rows are L2-normalized on load.
        for row in cohort.rows() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "row must be unit norm: {row:?}");
        }
        assert!((cohort.rows()[0][0] - 0.6).abs() < 1e-6);
        assert!((cohort.rows()[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn cohort_npy_errors_name_the_path() {
        let err = AsNormCohort::from_npy(Path::new("/no/such/cohort.npy")).unwrap_err();
        assert!(format!("{err}").contains("/no/such/cohort.npy"), "{err}");

        let tmp = tempfile::TempDir::new().unwrap();
        let bad = tmp.path().join("bad.npy");
        std::fs::write(&bad, b"NOTNUMPY!!garbage payload").unwrap();
        let err = AsNormCohort::from_npy(&bad).unwrap_err();
        assert!(format!("{err}").contains("not an NPY file"), "{err}");
    }

    #[test]
    fn as_norm_clusterer_trait_contract() {
        let (cohort, _, _) = channel_scene(&[0.0, 0.2, 0.4, 0.6]);

        let c = AsNormClusterer::new(8, 1.0, cohort.clone(), 20);
        let empty: &[Vec<f32>] = &[];
        assert!(matches!(
            c.cluster(empty).unwrap_err(),
            ClustererError::TooFewEmbeddings { .. }
        ));
        assert_eq!(c.cluster(&[vec![1.0, 0.0, 0.0]]).unwrap(), vec![0]);
        assert_eq!(c.max_clusters(), 8);
        assert!(!c.wants_raw_embeddings());

        // Embedding dim mismatch still reports DimMismatch.
        let err = c
            .cluster(&[vec![1.0, 0.0], vec![1.0, 0.0, 0.0]])
            .unwrap_err();
        assert!(matches!(err, ClustererError::DimMismatch { .. }));

        // Cohort dim mismatch is an explicit error, not a silent 0-score.
        let bad = AsNormClusterer::new(0, 1.0, cohort, 20);
        let err = bad.cluster(&[vec![1.0, 0.0], vec![0.9, 0.1]]).unwrap_err();
        match err {
            ClustererError::AlgorithmFailed { detail } => {
                assert!(detail.contains("cohort dim"), "{detail}");
            }
            other => panic!("expected AlgorithmFailed, got {other:?}"),
        }
    }

    /// The checked-in cohort fixture (VoxConverse-dev imposters) must load and
    /// match the embedder's 256-d output. Skipped until the fixture has been
    /// generated by the cohort-builder example.
    #[test]
    fn shipped_cohort_fixture_loads() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/asnorm/cohort_voxdev.npy");
        if !path.is_file() {
            eprintln!("skip: fixtures/asnorm/cohort_voxdev.npy not generated yet");
            return;
        }
        let cohort = AsNormCohort::from_npy(&path).unwrap();
        assert!(
            cohort.rows().len() >= 8,
            "a usable cohort has many speakers"
        );
        assert_eq!(cohort.dim(), Some(256));
        for row in cohort.rows() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "fixture rows must be unit norm");
        }
    }
}
