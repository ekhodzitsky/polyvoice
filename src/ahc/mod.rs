//! Agglomerative Hierarchical Clustering (AHC) for speaker diarization.
//!
//! Bottom-up clustering: each embedding starts as its own cluster, then the
//! two most similar clusters are merged iteratively until no pair exceeds
//! the cosine similarity threshold.

use crate::types::TimeRange;
use crate::utils::{
    cosine_similarity, l2_normalize, normalized_mean_centroids, pairwise_cosine_similarity_matrix,
};
use std::collections::HashMap;

/// Pairwise scoring seam for the AHC core.
///
/// Clusters are scored at two sites: the initial similarity matrix and the
/// post-merge refresh (a merged centroid is re-scored against every active
/// centroid). Both go through this trait so a normalizing scorer (AS-norm)
/// stays in effect across merges — transforming only the initial matrix would
/// be undone by the first refresh.
///
/// `member_a` / `member_b` index the *dominant* original embedding of each
/// cluster: at every merge the dominant member is inherited from the
/// heavier side (larger member count), so a merged centroid is normalized
/// with the cohort statistics of its largest-weight member embedding.
/// Cohort-free scorers ignore the indices.
pub(crate) trait AhcScorer {
    /// Score two clusters by their (L2-normalized) centroids.
    fn score(
        &self,
        centroid_a: &[f32],
        member_a: usize,
        centroid_b: &[f32],
        member_b: usize,
    ) -> f32;
}

/// Plain cosine similarity; the default scorer. Ignores member indices.
pub(crate) struct CosineScorer;

impl AhcScorer for CosineScorer {
    fn score(
        &self,
        centroid_a: &[f32],
        _member_a: usize,
        centroid_b: &[f32],
        _member_b: usize,
    ) -> f32 {
        cosine_similarity(centroid_a, centroid_b)
    }
}

/// { embeddings.is_empty() || embeddings.iter().all(|e| e.len() == embeddings`[0]`.len()) }
/// `pub fn agglomerative_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize>`
/// { ret.len() == embeddings.len() && ret.iter().all(|&l| l < embeddings.len()) }
/// Run agglomerative hierarchical clustering on a set of embeddings.
///
/// Returns a label vector of the same length as `embeddings`, where each
/// element is the cluster index (0-based, contiguous) for that embedding.
///
/// `threshold` is the minimum cosine similarity to merge two clusters.
/// Higher threshold → more clusters (stricter merging).
pub fn agglomerative_cluster(embeddings: &[Vec<f32>], threshold: f32) -> Vec<usize> {
    ahc_impl(embeddings, threshold, 0, AscStop::Off).0
}

/// Run AHC with a fixed threshold and a hard ceiling on the number of clusters.
pub fn agglomerative_cluster_max_clusters(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_clusters: usize,
) -> Vec<usize> {
    ahc_impl(embeddings, threshold, max_clusters, AscStop::Off).0
}

/// Stopping rule for **cAHC-ASC** (agglomerative clustering with an
/// "already-sufficient cluster" stop): refuse to merge two clusters that are
/// both already *established* by member count or total speech duration.
///
/// This protects genuine speakers from being glued together once each has
/// enough evidence, and is the main lever the cVBx line uses to improve
/// speaker-count estimates without lowering the global merge threshold.
#[derive(Debug, Clone, Copy, Default)]
pub enum AscStop {
    /// No established-cluster stop (classic AHC).
    #[default]
    Off,
    /// A cluster is established once it has at least this many members.
    /// `0` is treated as off.
    MinMembers(usize),
    /// A cluster is established once its overlap-merged speech duration reaches
    /// this many seconds. Requires time ranges of equal length to the embeddings.
    /// Non-positive values are treated as off.
    MinSecs(f64),
}

/// Run AHC with a fixed threshold, optional cluster ceiling, and optional
/// cAHC-ASC established-cluster stop. When `time_ranges` is required by the
/// stop rule but missing or length-mismatched, the stop is ignored (AHC still
/// runs on threshold / max_clusters alone).
pub fn agglomerative_cluster_asc(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_clusters: usize,
    stop: AscStop,
    time_ranges: Option<&[TimeRange]>,
) -> Vec<usize> {
    let stop = match stop {
        AscStop::MinSecs(s) if s > 0.0 => {
            if time_ranges.is_some_and(|t| t.len() == embeddings.len()) {
                AscStop::MinSecs(s)
            } else {
                AscStop::Off
            }
        }
        AscStop::MinMembers(0) => AscStop::Off,
        AscStop::MinSecs(s) if s <= 0.0 => AscStop::Off,
        other => other,
    };
    ahc_impl_with_times(embeddings, threshold, max_clusters, stop, time_ranges).0
}

/// Dissolve clusters smaller than `min_size` members by reassigning each of their
/// members to the nearest surviving (>= `min_size`) cluster centroid by cosine
/// similarity, returning compact `0..K` labels. If no cluster reaches `min_size`
/// the single largest is promoted to sole survivor so the result is never empty.
/// `min_size <= 1` and the "no small clusters" case are pass-throughs.
///
/// Fixes over-clustering — speakers fragmented into tiny spurious clusters that
/// inflate the speaker count — without merging large genuine speakers the way a
/// lower global merge threshold would. Requires `embeddings.len() == labels.len()`.
pub fn prune_small_clusters(
    embeddings: &[Vec<f32>],
    labels: Vec<usize>,
    min_size: usize,
) -> Vec<usize> {
    if min_size <= 1 {
        return labels;
    }
    let mut sizes: HashMap<usize, usize> = HashMap::new();
    for &l in &labels {
        *sizes.entry(l).or_insert(0) += 1;
    }
    // Survivors: clusters with >= min_size members; tie-break for the degenerate
    // "all small" case is the largest by size then smallest label.
    let survivors = survivors_or_largest(&sizes, |&size| size >= min_size);
    finish_prune(embeddings, labels, survivors, sizes.len())
}

/// Like [`prune_small_clusters`] but length-invariant: a cluster survives when
/// its total (overlap-merged) speech **duration** is at least `min_secs`. This
/// avoids the short-audio failure of a fixed member count — a real minority
/// speaker keeps its few-but-long windows while genuinely brief spurious clusters
/// are dissolved, on clips of any length. `min_secs <= 0` is a pass-through.
/// Requires `time_ranges.len() == embeddings.len() == labels.len()`.
pub fn prune_small_clusters_by_duration(
    time_ranges: &[TimeRange],
    embeddings: &[Vec<f32>],
    labels: Vec<usize>,
    min_secs: f64,
) -> Vec<usize> {
    if min_secs <= 0.0 {
        return labels;
    }
    let durations = cluster_durations(time_ranges, &labels);
    let survivors = survivors_or_largest(&durations, |&d| d >= min_secs);
    finish_prune(embeddings, labels, survivors, durations.len())
}

/// Overlap-merged total duration of each cluster's member windows.
fn cluster_durations(time_ranges: &[TimeRange], labels: &[usize]) -> HashMap<usize, f64> {
    let mut by_label: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if let Some(t) = time_ranges.get(i) {
            by_label.entry(l).or_default().push((t.start, t.end));
        }
    }
    let mut out = HashMap::with_capacity(by_label.len());
    for (l, mut spans) in by_label {
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut total = 0.0_f64;
        let mut cur: Option<(f64, f64)> = None;
        for (s, e) in spans {
            cur = match cur {
                Some((cs, ce)) if s <= ce => Some((cs, ce.max(e))),
                Some((cs, ce)) => {
                    total += ce - cs;
                    Some((s, e))
                }
                None => Some((s, e)),
            };
        }
        if let Some((cs, ce)) = cur {
            total += ce - cs;
        }
        out.insert(l, total);
    }
    out
}

/// Choose the surviving labels: those whose metric passes `keep`. If none pass,
/// promote the single label with the largest metric (tie-break: smallest label)
/// so the result is never empty.
fn survivors_or_largest<M>(metrics: &HashMap<usize, M>, keep: impl Fn(&M) -> bool) -> Vec<usize>
where
    M: PartialOrd + Copy,
{
    let mut survivors: Vec<usize> = metrics
        .iter()
        .filter(|kv| keep(kv.1))
        .map(|kv| *kv.0)
        .collect();
    if survivors.is_empty() {
        let mut best: Option<(M, usize)> = None;
        for (&l, &m) in metrics {
            best = Some(match best {
                Some((bm, bl)) if bm > m || (bm == m && bl <= l) => (bm, bl),
                _ => (m, l),
            });
        }
        if let Some((_, l)) = best {
            survivors.push(l);
        }
    }
    survivors
}

/// Shared tail: reassign members of non-surviving clusters to the nearest
/// survivor centroid by cosine similarity, returning compact `0..K` labels.
/// `total_clusters` is the count of distinct labels before pruning — when every
/// cluster survives the input (already compact) is returned untouched.
fn finish_prune(
    embeddings: &[Vec<f32>],
    labels: Vec<usize>,
    mut survivors: Vec<usize>,
    total_clusters: usize,
) -> Vec<usize> {
    if survivors.is_empty() || survivors.len() == total_clusters {
        return labels;
    }
    survivors.sort_unstable();
    let sidx: HashMap<usize, usize> = survivors.iter().enumerate().map(|(i, &l)| (l, i)).collect();

    // L2-normalized centroid per survivor (indexed by survivor position).
    // Non-survivor labels map to an out-of-range slot and are skipped.
    let indices: Vec<usize> = (0..labels.len()).collect();
    let slots: Vec<usize> = labels
        .iter()
        .map(|l| sidx.get(l).copied().unwrap_or(survivors.len()))
        .collect();
    let centroids = normalized_mean_centroids(embeddings, &indices, &slots, survivors.len());

    // Survivors keep their compact index; others go to the nearest survivor
    // centroid. Result is compact 0..survivors.len() by construction.
    let mut out = vec![0usize; labels.len()];
    for (i, &l) in labels.iter().enumerate() {
        out[i] = match sidx.get(&l) {
            Some(&si) => si,
            None => {
                let mut best = 0usize;
                let mut best_sim = f32::NEG_INFINITY;
                for (si, c) in centroids.iter().enumerate() {
                    let sim = cosine_similarity(&embeddings[i], c);
                    if sim > best_sim {
                        best_sim = sim;
                        best = si;
                    }
                }
                best
            }
        };
    }
    out
}

/// { embeddings.is_empty() || embeddings.iter().all(|e| e.len() == embeddings`[0]`.len()) }
/// `pub fn agglomerative_cluster_auto_max_clusters(embeddings: &[Vec<f32>], max_clusters: usize) -> (Vec<usize>, f32)`
/// { ret.0.len() == embeddings.len() && ret.0.iter().all(|&l| l < embeddings.len()) && ret.1 >= 0.0 }
/// Run AHC with automatic threshold selection via largest-gap heuristic on the
/// sorted pairwise similarities, and a hard ceiling on the number of clusters.
pub fn agglomerative_cluster_auto_max_clusters(
    embeddings: &[Vec<f32>],
    max_clusters: usize,
) -> (Vec<usize>, f32) {
    let n = embeddings.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    // Build the pairwise matrix once: the threshold estimate reads its upper
    // triangle and the clustering core reuses it verbatim.
    let sim_matrix = similarity_matrix(embeddings);
    let threshold = estimate_threshold_from_matrix(&sim_matrix);
    let dim = embeddings[0].len();
    if !embeddings.iter().all(|e| e.len() == dim) {
        // Defensive fallback: mixed dimensions would break similarity math.
        // Return a single cluster rather than panicking.
        return (vec![0; n], 0.0);
    }
    ahc_impl_with_matrix(
        embeddings,
        threshold,
        max_clusters,
        AscStop::Off,
        None,
        &CosineScorer,
        sim_matrix,
    )
}

/// Run fixed-threshold AHC with a custom pairwise scorer (e.g. AS-norm).
///
/// The scorer is used BOTH for the initial similarity matrix and for every
/// post-merge centroid refresh, so normalization holds across the whole merge
/// sequence. Only the fixed-threshold path accepts a custom scorer: the auto
/// path derives its threshold from the raw matrix's gap structure, which a
/// normalizing scorer would rescale.
#[cfg(feature = "clusterer")]
pub(crate) fn agglomerative_cluster_scored(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_clusters: usize,
    scorer: &dyn AhcScorer,
) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    if !embeddings.iter().all(|e| e.len() == dim) {
        // Defensive fallback: mixed dimensions would break similarity math.
        return vec![0; n];
    }
    let sim_matrix = similarity_matrix_scored(embeddings, scorer);
    ahc_impl_with_matrix(
        embeddings,
        threshold,
        max_clusters,
        AscStop::Off,
        None,
        scorer,
        sim_matrix,
    )
    .0
}

/// Nested (row-per-embedding) cosine-similarity matrix over `embeddings`:
/// diagonal `1.0`, off-diagonal pairs from the shared flat pairwise matrix.
fn similarity_matrix(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let flat = pairwise_cosine_similarity_matrix(embeddings);
    flat.chunks(n).map(<[f32]>::to_vec).collect()
}

/// Like [`similarity_matrix`] but scored by `scorer` — used when a normalizing
/// scorer (AS-norm) replaces plain cosine.
#[cfg(feature = "clusterer")]
#[allow(clippy::needless_range_loop)] // triangular mirror indexing, same style as the AHC core
fn similarity_matrix_scored(embeddings: &[Vec<f32>], scorer: &dyn AhcScorer) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut matrix = vec![vec![1.0f32; n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate().skip(i + 1) {
            *cell = scorer.score(&embeddings[i], i, &embeddings[j], j);
        }
    }
    for i in 0..n {
        for j in 0..i {
            matrix[i][j] = matrix[j][i];
        }
    }
    matrix
}

fn ahc_impl(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_clusters: usize,
    stop: AscStop,
) -> (Vec<usize>, f32) {
    ahc_impl_with_times(embeddings, threshold, max_clusters, stop, None)
}

fn ahc_impl_with_times(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_clusters: usize,
    stop: AscStop,
    time_ranges: Option<&[TimeRange]>,
) -> (Vec<usize>, f32) {
    let n = embeddings.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let dim = embeddings[0].len();
    if !embeddings.iter().all(|e| e.len() == dim) {
        // Defensive fallback: mixed dimensions would break similarity math.
        // Return a single cluster rather than panicking.
        return (vec![0; n], 0.0);
    }
    let sim_matrix = similarity_matrix(embeddings);
    ahc_impl_with_matrix(
        embeddings,
        threshold,
        max_clusters,
        stop,
        time_ranges,
        &CosineScorer,
        sim_matrix,
    )
}

#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn ahc_impl_with_matrix(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_clusters: usize,
    stop: AscStop,
    time_ranges: Option<&[TimeRange]>,
    scorer: &dyn AhcScorer,
    mut sim_matrix: Vec<Vec<f32>>,
) -> (Vec<usize>, f32) {
    // Callers guarantee n >= 1 and uniform embedding dimensions.
    let n = embeddings.len();

    let mut labels: Vec<usize> = (0..n).collect();
    let mut centroids: Vec<Vec<f32>> = embeddings.to_vec();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    // Dominant member of each cluster root: the original-embedding index a
    // cohort-based scorer uses for this centroid's normalization stats. Starts
    // as the singleton itself; at each merge the heavier side's dominant
    // member wins (ties keep the merge target's), matching the weighted
    // centroid update.
    let mut dominant: Vec<usize> = (0..n).collect();
    // Per-root total speech duration (only used for AscStop::MinSecs).
    let mut cluster_dur: Vec<f64> = match (stop, time_ranges) {
        (AscStop::MinSecs(_), Some(times)) if times.len() == n => {
            times.iter().map(|t| t.duration().max(0.0)).collect()
        }
        _ => vec![0.0; n],
    };
    let mut active: Vec<bool> = vec![true; n];

    // sim_matrix[i][j] holds the similarity between centroids i and j
    // (precomputed by the caller). Inactive rows/columns are kept at
    // NEG_INFINITY.
    let neg_inf = f32::NEG_INFINITY;

    loop {
        let mut best_sim = neg_inf;
        let mut best_i = 0;
        let mut best_j = 0;

        // Find the best pair among active clusters, skipping merges that would
        // glue two established clusters (cAHC-ASC).
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                if both_established(
                    stop,
                    cluster_sizes[i],
                    cluster_sizes[j],
                    cluster_dur[i],
                    cluster_dur[j],
                ) {
                    continue;
                }
                let sim = sim_matrix[i][j];
                if sim > best_sim {
                    best_sim = sim;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let active_count = active.iter().filter(|&&a| a).count();
        let above_ceiling = max_clusters > 0 && max_clusters < n && active_count > max_clusters;
        if !above_ceiling && best_sim < threshold {
            break;
        }
        if above_ceiling && best_sim == neg_inf {
            break;
        }
        // cAHC-ASC: every remaining pair is two established clusters → stop even
        // if the ceiling still wants fewer clusters (prefer correct count).
        if best_sim == neg_inf {
            break;
        }

        // Merge j into i.
        let total = cluster_sizes[best_i] + cluster_sizes[best_j];
        let w_i = cluster_sizes[best_i] as f32 / total as f32;
        let w_j = cluster_sizes[best_j] as f32 / total as f32;
        let dim = centroids[best_i].len();
        let mut new_centroid = vec![0.0f32; dim];
        for k in 0..dim {
            new_centroid[k] = centroids[best_i][k] * w_i + centroids[best_j][k] * w_j;
        }
        l2_normalize(&mut new_centroid);

        centroids[best_i] = new_centroid;
        // The merged cluster's dominant member comes from the heavier side
        // (ties keep best_i's): its cohort stats approximate the merged
        // centroid's normalization better than any per-member recomputation.
        if cluster_sizes[best_j] > cluster_sizes[best_i] {
            dominant[best_i] = dominant[best_j];
        }
        cluster_sizes[best_i] = total;
        cluster_dur[best_i] += cluster_dur[best_j];
        active[best_j] = false;

        // Invalidate best_j from the similarity matrix.
        for k in 0..n {
            sim_matrix[best_j][k] = neg_inf;
            sim_matrix[k][best_j] = neg_inf;
        }

        // Update similarities for best_i against all other active clusters.
        // The scorer sees the dominant member indices so cohort-based
        // normalization survives merges (a matrix-only transform would not).
        for k in 0..n {
            if k == best_i || !active[k] {
                continue;
            }
            let sim = scorer.score(
                &centroids[best_i],
                dominant[best_i],
                &centroids[k],
                dominant[k],
            );
            sim_matrix[best_i][k] = sim;
            sim_matrix[k][best_i] = sim;
        }

        for label in &mut labels {
            if *label == best_j {
                *label = best_i;
            }
        }
    }

    // Canonicalize cluster ids: relabel by descending cluster size, ties broken
    // by smallest member index, into a contiguous 0..K range. This makes the
    // integer ids a deterministic function of the PARTITION (independent of
    // embedding/merge order), so golden/DER tests are not brittle to input
    // ordering. The partition itself is unchanged; downstream treats ids as
    // opaque and DER maps speakers optimally (order-invariant).
    let mut group: HashMap<usize, (usize, usize)> = HashMap::new(); // raw label -> (size, min_index)
    for (idx, &label) in labels.iter().enumerate() {
        // idx ascends, so the first sight of a label already holds its
        // smallest member index.
        group.entry(label).or_insert((0, idx)).0 += 1;
    }
    let mut order: Vec<(usize, usize, usize)> = group
        .iter()
        .map(|(&label, &(size, min_idx))| (size, min_idx, label))
        .collect();
    // Descending size, then ascending smallest-member index (stable tie-break).
    order.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut canonical: HashMap<usize, usize> = HashMap::new();
    for (new_id, &(_, _, label)) in order.iter().enumerate() {
        canonical.insert(label, new_id);
    }
    for label in &mut labels {
        *label = canonical[label];
    }

    (labels, threshold)
}

/// True when both clusters already meet the established threshold and must not
/// be merged under cAHC-ASC.
fn both_established(stop: AscStop, size_i: usize, size_j: usize, dur_i: f64, dur_j: f64) -> bool {
    match stop {
        AscStop::Off => false,
        AscStop::MinMembers(m) => m > 0 && size_i >= m && size_j >= m,
        AscStop::MinSecs(s) => s > 0.0 && dur_i >= s && dur_j >= s,
    }
}

/// Estimate a good AHC threshold from the precomputed pairwise similarity
/// matrix.
///
/// Collects the upper triangle, sorts it, and finds the largest gap in the
/// lower half of the distribution (between 0.0 and median). This tends to
/// separate within-speaker from between-speaker pairs.
fn estimate_threshold_from_matrix(sim_matrix: &[Vec<f32>]) -> f32 {
    let n = sim_matrix.len();
    if n < 2 {
        return 0.5;
    }

    let mut sims: Vec<f32> = Vec::with_capacity(n * (n - 1) / 2);
    for (i, row) in sim_matrix.iter().enumerate() {
        sims.extend_from_slice(&row[i + 1..]);
    }
    sims.sort_by(|a, b| a.total_cmp(b));

    // n >= 2 here, so the upper triangle holds at least one similarity.
    let median_idx = sims.len() / 2;
    // Search for the largest gap in the range [0.0, median].
    let mut best_gap = 0.0f32;
    let mut best_idx = 0usize;
    for i in 0..median_idx.saturating_sub(1) {
        let gap = sims[i + 1] - sims[i];
        if gap > best_gap {
            best_gap = gap;
            best_idx = i;
        }
    }

    // Threshold is the similarity value after the gap.
    if sims.len() <= 1 {
        return 0.5;
    }
    let th = sims[best_idx + 1];
    // Clamp to a reasonable range.
    th.clamp(0.2, 0.7)
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agglomerative_cluster_basic() {
        // Two clear clusters.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let labels = agglomerative_cluster(&embeddings, 0.5);
        assert_eq!(labels.len(), 4);
        assert_eq!(labels.iter().copied().max(), Some(1));
        // First two should be same cluster, last two should be same cluster.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_agglomerative_cluster_empty() {
        let labels = agglomerative_cluster(&[], 0.5);
        assert!(labels.is_empty());
    }

    // --- duration-based pruning ---

    fn ax(a: usize) -> Vec<f32> {
        let mut v = vec![0.02f32, 0.02, 0.02];
        v[a] = 1.0;
        v
    }
    fn tr(s: f64, e: f64) -> TimeRange {
        TimeRange { start: s, end: e }
    }
    fn ndistinct(labels: &[usize]) -> usize {
        let set: std::collections::HashSet<usize> = labels.iter().copied().collect();
        set.len()
    }

    #[test]
    fn prune_by_duration_dissolves_brief_cluster() {
        // cluster 0: ~3 s on axis 0; cluster 1: one 0.5 s window on axis 1.
        let embeddings = vec![ax(0), ax(0), ax(0), ax(0), ax(1)];
        let times = vec![
            tr(0.0, 1.0),
            tr(0.75, 1.75),
            tr(1.5, 2.5),
            tr(2.25, 3.25),
            tr(5.0, 5.5),
        ];
        let out = prune_small_clusters_by_duration(&times, &embeddings, vec![0, 0, 0, 0, 1], 1.5);
        assert_eq!(out.len(), 5);
        assert_eq!(ndistinct(&out), 1, "the brief 0.5 s cluster is dissolved");
    }

    #[test]
    fn prune_by_duration_keeps_few_but_long_speaker() {
        // cluster 1 has only TWO windows but 4 s of speech — it survives duration
        // pruning, whereas the member-count rule (min 4) wrongly dissolves it.
        let embeddings = vec![ax(0), ax(0), ax(0), ax(0), ax(0), ax(0), ax(1), ax(1)];
        let times = vec![
            tr(0.0, 1.0),
            tr(0.75, 1.75),
            tr(1.5, 2.5),
            tr(2.25, 3.25),
            tr(3.0, 4.0),
            tr(3.75, 4.75),
            tr(10.0, 12.0),
            tr(12.0, 14.0),
        ];
        let labels = vec![0, 0, 0, 0, 0, 0, 1, 1];
        let dur = prune_small_clusters_by_duration(&times, &embeddings, labels.clone(), 1.5);
        assert_eq!(
            ndistinct(&dur),
            2,
            "few-but-long cluster survives duration prune"
        );
        assert_ne!(dur[0], dur[6]);
        // Contrast: the count rule over-prunes the same 2-member cluster.
        let cnt = prune_small_clusters(&embeddings, labels, 4);
        assert_eq!(
            ndistinct(&cnt),
            1,
            "count rule over-prunes the long-but-few cluster"
        );
    }

    #[test]
    fn prune_by_duration_zero_is_passthrough() {
        let embeddings = vec![ax(0), ax(1)];
        let times = vec![tr(0.0, 0.1), tr(1.0, 1.1)];
        let labels = vec![0, 1];
        assert_eq!(
            prune_small_clusters_by_duration(&times, &embeddings, labels.clone(), 0.0),
            labels
        );
    }

    #[test]
    fn prune_by_duration_all_short_keeps_one() {
        let embeddings = vec![ax(0), ax(1), ax(2)];
        let times = vec![tr(0.0, 0.2), tr(1.0, 1.2), tr(2.0, 2.2)];
        let out = prune_small_clusters_by_duration(&times, &embeddings, vec![0, 1, 2], 5.0);
        assert_eq!(out.len(), 3);
        assert_eq!(ndistinct(&out), 1, "all-short collapses to one survivor");
    }

    #[test]
    fn test_agglomerative_cluster_single() {
        let embeddings = vec![vec![1.0, 0.0, 0.0]];
        let labels = agglomerative_cluster(&embeddings, 0.5);
        assert_eq!(labels, vec![0]);
    }

    #[test]
    fn test_agglomerative_cluster_auto_max_clusters_caps_count() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let (labels, _th) = agglomerative_cluster_auto_max_clusters(&embeddings, 2);
        let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(
            unique.len(),
            2,
            "max_clusters=2 must produce exactly 2 clusters"
        );
    }

    #[test]
    fn test_agglomerative_cluster_mismatched_dimensions() {
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1]];
        let labels = agglomerative_cluster(&embeddings, 0.5);
        assert_eq!(labels, vec![0, 0]);
    }

    #[test]
    fn cluster_ids_are_canonical_and_shuffle_invariant() {
        // A 3-member cluster near [1,0,0] and a 2-member cluster near [0,1,0].
        let a = vec![1.0, 0.0, 0.0];
        let a2 = vec![0.95, 0.05, 0.0];
        let a3 = vec![0.9, 0.1, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let b2 = vec![0.05, 0.95, 0.0];

        // Canonical ordering: the larger cluster (3 members) must get id 0,
        // regardless of input order — descending size, tie-break min index.
        let base = vec![a.clone(), a2.clone(), a3.clone(), b.clone(), b2.clone()];
        let l1 = agglomerative_cluster(&base, 0.5);
        assert_eq!(l1, vec![0, 0, 0, 1, 1], "big cluster must be id 0");

        // Shuffled copy of the SAME points: the 3-member cluster (a-points at
        // shuffled indices 1,3,4) must still be id 0. The old first-appearance
        // relabel would have made the first-seen cluster id 0 instead.
        let shuffled = vec![b2.clone(), a3.clone(), b.clone(), a.clone(), a2.clone()];
        let l2 = agglomerative_cluster(&shuffled, 0.5);
        assert_eq!(l2[1], 0, "a3 is in the big cluster -> id 0");
        assert_eq!(l2[3], 0, "a is in the big cluster -> id 0");
        assert_eq!(l2[4], 0, "a2 is in the big cluster -> id 0");
        assert_eq!(l2[0], 1, "b2 is in the small cluster -> id 1");
        assert_eq!(l2[2], 1, "b is in the small cluster -> id 1");
    }

    #[test]
    fn cahc_asc_refuses_to_merge_two_established_clusters() {
        // Two well-separated speakers (3 members each). With a very low
        // threshold classic AHC would still merge them; MinMembers(3) stops
        // before that final merge because both sides are already established.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.05, 0.0],
            vec![0.97, 0.0, 0.05],
            vec![0.0, 1.0, 0.0],
            vec![0.05, 0.98, 0.0],
            vec![0.0, 0.97, 0.05],
        ];
        let classic = agglomerative_cluster(&embeddings, 0.0);
        assert_eq!(
            ndistinct(&classic),
            1,
            "threshold 0 must glue everything without ASC"
        );
        let asc = agglomerative_cluster_asc(&embeddings, 0.0, 0, AscStop::MinMembers(3), None);
        assert_eq!(
            ndistinct(&asc),
            2,
            "cAHC-ASC must keep two established speakers separate"
        );
        assert_eq!(asc[0], asc[1]);
        assert_eq!(asc[0], asc[2]);
        assert_eq!(asc[3], asc[4]);
        assert_eq!(asc[3], asc[5]);
        assert_ne!(asc[0], asc[3]);
    }

    #[test]
    fn cahc_asc_min_secs_uses_durations() {
        // Same geometry as above, but establish by total duration (each speaker
        // has ~3 s of speech across three 1 s windows).
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.05, 0.0],
            vec![0.97, 0.0, 0.05],
            vec![0.0, 1.0, 0.0],
            vec![0.05, 0.98, 0.0],
            vec![0.0, 0.97, 0.05],
        ];
        let times = vec![
            tr(0.0, 1.0),
            tr(1.0, 2.0),
            tr(2.0, 3.0),
            tr(10.0, 11.0),
            tr(11.0, 12.0),
            tr(12.0, 13.0),
        ];
        let asc =
            agglomerative_cluster_asc(&embeddings, 0.0, 0, AscStop::MinSecs(2.5), Some(&times));
        assert_eq!(ndistinct(&asc), 2);
        assert_ne!(asc[0], asc[3]);
    }

    #[test]
    fn cahc_asc_off_matches_classic() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let classic = agglomerative_cluster_max_clusters(&embeddings, 0.5, 0);
        let asc = agglomerative_cluster_asc(&embeddings, 0.5, 0, AscStop::Off, None);
        assert_eq!(classic, asc);
    }

    #[test]
    fn cahc_asc_min_secs_ignored_without_matching_time_ranges() {
        // MinSecs with no time ranges (or a length mismatch) degrades to Off:
        // threshold 0 then glues everything, exactly like classic AHC.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.05, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.05, 0.98, 0.0],
        ];
        let classic = agglomerative_cluster(&embeddings, 0.0);
        assert_eq!(ndistinct(&classic), 1);

        let no_times = agglomerative_cluster_asc(&embeddings, 0.0, 0, AscStop::MinSecs(2.0), None);
        assert_eq!(no_times, classic, "missing time ranges → stop ignored");

        let short = vec![tr(0.0, 1.0), tr(1.0, 2.0)];
        let mismatched =
            agglomerative_cluster_asc(&embeddings, 0.0, 0, AscStop::MinSecs(2.0), Some(&short));
        assert_eq!(mismatched, classic, "length mismatch → stop ignored");
    }

    #[test]
    fn cahc_asc_degenerate_stops_are_off() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let classic = agglomerative_cluster(&embeddings, 0.0);
        assert_eq!(ndistinct(&classic), 1);
        for stop in [
            AscStop::MinMembers(0),
            AscStop::MinSecs(0.0),
            AscStop::MinSecs(-1.5),
        ] {
            let asc = agglomerative_cluster_asc(&embeddings, 0.0, 0, stop, None);
            assert_eq!(asc, classic, "{stop:?} must behave as Off");
        }
    }

    #[test]
    fn cahc_asc_ceiling_yields_to_established_clusters() {
        // Two tight pairs, each established after one merge. max_clusters = 1
        // wants more merging, but every remaining pair is two established
        // clusters — the ASC stop wins over the ceiling.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.05, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.05, 0.98, 0.0],
        ];
        let asc = agglomerative_cluster_asc(&embeddings, 0.0, 1, AscStop::MinMembers(2), None);
        assert_eq!(
            ndistinct(&asc),
            2,
            "ceiling must not glue two established clusters"
        );
        assert_eq!(asc[0], asc[1]);
        assert_eq!(asc[2], asc[3]);
        assert_ne!(asc[0], asc[2]);
    }

    #[test]
    fn cahc_asc_stop_holds_even_at_unbounded_threshold() {
        // threshold = -inf merges anything allowed; the only blocked merge is
        // the established-vs-established pair, which must still be refused.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.98, 0.05, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.05, 0.98, 0.0],
        ];
        let asc = agglomerative_cluster_asc(
            &embeddings,
            f32::NEG_INFINITY,
            0,
            AscStop::MinMembers(2),
            None,
        );
        assert_eq!(ndistinct(&asc), 2);
        // Sanity: without the stop, -inf glues everything into one cluster.
        let classic = agglomerative_cluster(&embeddings, f32::NEG_INFINITY);
        assert_eq!(ndistinct(&classic), 1);
    }

    #[test]
    fn cluster_durations_sums_disjoint_spans_separately() {
        // Disjoint spans in one cluster must not be merged into a single span:
        // total = (1.0) + (1.0) = 2.0, not the 4.0 outer envelope.
        let times = vec![tr(0.0, 1.0), tr(3.0, 4.0), tr(0.5, 2.0)];
        let labels = vec![0, 0, 1];
        let durs = cluster_durations(&times, &labels);
        assert!((durs[&0] - 2.0).abs() < 1e-12);
        assert!((durs[&1] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn auto_max_clusters_empty_input() {
        let (labels, th) = agglomerative_cluster_auto_max_clusters(&[], 3);
        assert!(labels.is_empty());
        assert_eq!(th, 0.0);
    }

    #[test]
    fn auto_max_clusters_mismatched_dimensions_fall_back_to_one_cluster() {
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1]];
        let (labels, th) = agglomerative_cluster_auto_max_clusters(&embeddings, 2);
        assert_eq!(labels, vec![0, 0]);
        assert_eq!(th, 0.0);
    }

    // --- custom-scorer (AS-norm seam) tests ---

    /// Scorer driven by a fixed member-keyed score table, recording every
    /// `(member_a, member_b)` pair it is called with, so tests can observe
    /// which dominant members the merge loop scores after each merge.
    #[cfg(feature = "clusterer")]
    struct TableScorer {
        table: HashMap<(usize, usize), f32>,
        calls: std::cell::RefCell<Vec<(usize, usize)>>,
    }

    #[cfg(feature = "clusterer")]
    impl TableScorer {
        fn from_pairs(pairs: &[((usize, usize), f32)]) -> Self {
            Self {
                table: pairs.iter().copied().collect(),
                calls: std::cell::RefCell::new(Vec::new()),
            }
        }
    }

    #[cfg(feature = "clusterer")]
    impl AhcScorer for TableScorer {
        fn score(&self, _a: &[f32], ma: usize, _b: &[f32], mb: usize) -> f32 {
            self.calls.borrow_mut().push((ma, mb));
            self.table
                .get(&(ma, mb))
                .or_else(|| self.table.get(&(mb, ma)))
                .copied()
                .unwrap_or(f32::NEG_INFINITY)
        }
    }

    #[cfg(feature = "clusterer")]
    #[test]
    fn scored_ahc_uses_scorer_for_matrix_and_post_merge_refresh() {
        // Four embeddings; geometry irrelevant — the table drives every score.
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.9, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 0.9],
        ];
        // s(0,1) = 0.99 merges first; then {0,1} (dominant 0, heavier side)
        // merges with 2 at 0.9; 3 stays out at 0.1. Threshold 0.85.
        let scorer = TableScorer::from_pairs(&[
            ((0, 1), 0.99),
            ((0, 2), 0.9),
            ((1, 2), 0.9),
            ((0, 3), 0.1),
            ((1, 3), 0.1),
            ((2, 3), 0.1),
        ]);
        let labels = agglomerative_cluster_scored(&embeddings, 0.85, 0, &scorer);
        assert_eq!(labels, vec![0, 0, 0, 1], "table-driven merges");
        let calls = scorer.calls.borrow();
        assert_eq!(
            &calls[..6],
            &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            "initial matrix scored once per pair, upper triangle"
        );
        // After merging 1 into 0 (tie 1v1 keeps best_i's dominant) the refresh
        // re-scores centroid 0 via its dominant member 0 — not raw cosine.
        assert_eq!(calls[6], (0, 2), "post-merge refresh uses dominant member");
        assert_eq!(calls[7], (0, 3));
        // After the second merge ({0,1} heavier than {2}), the dominant member
        // stays 0 — the heavier side's member, not the freshly merged 2.
        assert_eq!(
            calls[8],
            (0, 3),
            "dominant member of an unequal merge comes from the heavier side"
        );
        assert_eq!(calls.len(), 9, "no rescoring beyond the merge refreshes");
    }

    #[cfg(feature = "clusterer")]
    #[test]
    fn scored_ahc_constant_scorer_controls_merging() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];
        struct Const(f32);
        impl AhcScorer for Const {
            fn score(&self, _a: &[f32], _ma: usize, _b: &[f32], _mb: usize) -> f32 {
                self.0
            }
        }
        // All pairs at 1.0 ≥ 0.5: everything merges despite the geometry.
        let merged = agglomerative_cluster_scored(&embeddings, 0.5, 0, &Const(1.0));
        assert_eq!(merged, vec![0, 0, 0]);
        // All pairs at 0.0 < 0.5: nothing merges.
        let split = agglomerative_cluster_scored(&embeddings, 0.5, 0, &Const(0.0));
        assert_eq!(split, vec![0, 1, 2]);
    }

    #[cfg(feature = "clusterer")]
    #[test]
    fn scored_ahc_degenerate_inputs() {
        let empty: Vec<Vec<f32>> = Vec::new();
        assert!(agglomerative_cluster_scored(&empty, 0.5, 0, &CosineScorer).is_empty());
        // Mixed dimensions fall back to a single cluster, like the cosine path.
        let mixed = vec![vec![1.0, 0.0], vec![0.9]];
        assert_eq!(
            agglomerative_cluster_scored(&mixed, 0.5, 0, &CosineScorer),
            vec![0, 0]
        );
    }

    #[cfg(feature = "clusterer")]
    #[test]
    fn scored_ahc_with_cosine_scorer_matches_classic() {
        // The cosine scorer through the scored path must reproduce plain AHC.
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let classic = agglomerative_cluster_max_clusters(&embeddings, 0.5, 0);
        let scored = agglomerative_cluster_scored(&embeddings, 0.5, 0, &CosineScorer);
        assert_eq!(classic, scored);
    }
}
