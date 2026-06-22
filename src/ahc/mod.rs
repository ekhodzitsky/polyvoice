//! Agglomerative Hierarchical Clustering (AHC) for speaker diarization.
//!
//! Bottom-up clustering: each embedding starts as its own cluster, then the
//! two most similar clusters are merged iteratively until no pair exceeds
//! the cosine similarity threshold.

use crate::types::TimeRange;
use crate::utils::{cosine_similarity, l2_normalize};
use std::collections::HashMap;

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
    ahc_impl(embeddings, threshold, 0).0
}

/// Run AHC with a fixed threshold and a hard ceiling on the number of clusters.
pub fn agglomerative_cluster_max_clusters(
    embeddings: &[Vec<f32>],
    threshold: f32,
    max_clusters: usize,
) -> Vec<usize> {
    ahc_impl(embeddings, threshold, max_clusters).0
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
    let dim = embeddings.first().map(Vec::len).unwrap_or(0);
    let mut centroids: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; survivors.len()];
    let mut counts: Vec<usize> = vec![0; survivors.len()];
    for (i, &l) in labels.iter().enumerate() {
        if let Some(&si) = sidx.get(&l) {
            for (a, &x) in centroids[si].iter_mut().zip(embeddings[i].iter()) {
                *a += x;
            }
            counts[si] += 1;
        }
    }
    for (c, &n) in centroids.iter_mut().zip(counts.iter()) {
        if n > 0 {
            for v in c.iter_mut() {
                *v /= n as f32;
            }
        }
        let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in c.iter_mut() {
                *v /= norm;
            }
        }
    }

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
/// `pub fn agglomerative_cluster_auto(embeddings: &[Vec<f32>]) -> (Vec<usize>, f32)`
/// { ret.0.len() == embeddings.len() && ret.0.iter().all(|&l| l < embeddings.len()) && ret.1 >= 0.0 }
/// Run AHC with automatic threshold selection via largest-merge-gap heuristic.
///
/// Returns labels and the automatically selected threshold.
pub fn agglomerative_cluster_auto(embeddings: &[Vec<f32>]) -> (Vec<usize>, f32) {
    agglomerative_cluster_auto_max_clusters(embeddings, 0)
}

/// Run AHC with automatic threshold selection and a hard ceiling on the number
/// of clusters.
pub fn agglomerative_cluster_auto_max_clusters(
    embeddings: &[Vec<f32>],
    max_clusters: usize,
) -> (Vec<usize>, f32) {
    let n = embeddings.len();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let threshold = estimate_threshold_from_similarities(embeddings);
    ahc_impl(embeddings, threshold, max_clusters)
}

#[allow(clippy::needless_range_loop)]
fn ahc_impl(embeddings: &[Vec<f32>], threshold: f32, max_clusters: usize) -> (Vec<usize>, f32) {
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

    let mut labels: Vec<usize> = (0..n).collect();
    let mut centroids: Vec<Vec<f32>> = embeddings.to_vec();
    let mut cluster_sizes: Vec<usize> = vec![1; n];
    let mut active: Vec<bool> = vec![true; n];

    // Precompute similarity matrix. sim_matrix[i][j] holds the similarity
    // between centroids i and j. Inactive rows/columns are kept at NEG_INFINITY.
    let neg_inf = f32::NEG_INFINITY;
    let mut sim_matrix: Vec<Vec<f32>> = vec![vec![neg_inf; n]; n];
    for i in 0..n {
        sim_matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let sim = cosine_similarity(&centroids[i], &centroids[j]);
            sim_matrix[i][j] = sim;
            sim_matrix[j][i] = sim;
        }
    }

    loop {
        let mut best_sim = neg_inf;
        let mut best_i = 0;
        let mut best_j = 0;

        // Find the best pair among active clusters.
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
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
        cluster_sizes[best_i] = total;
        active[best_j] = false;

        // Invalidate best_j from the similarity matrix.
        for k in 0..n {
            sim_matrix[best_j][k] = neg_inf;
            sim_matrix[k][best_j] = neg_inf;
        }

        // Update similarities for best_i against all other active clusters.
        for k in 0..n {
            if k == best_i || !active[k] {
                continue;
            }
            let sim = cosine_similarity(&centroids[best_i], &centroids[k]);
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
        let e = group.entry(label).or_insert((0, idx));
        e.0 += 1;
        if idx < e.1 {
            e.1 = idx;
        }
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

/// Estimate a good AHC threshold from the distribution of pairwise similarities.
///
/// Computes all pairwise cosine similarities, sorts them, and finds the largest
/// gap in the lower half of the distribution (between 0.0 and median).
/// This tends to separate within-speaker from between-speaker pairs.
fn estimate_threshold_from_similarities(embeddings: &[Vec<f32>]) -> f32 {
    let n = embeddings.len();
    if n < 2 {
        return 0.5;
    }

    let mut sims: Vec<f32> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            sims.push(cosine_similarity(&embeddings[i], &embeddings[j]));
        }
    }
    sims.sort_by(|a, b| a.total_cmp(b));

    if sims.is_empty() {
        return 0.5;
    }

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
    fn test_agglomerative_cluster_auto_basic() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
        ];
        let (labels, th) = agglomerative_cluster_auto(&embeddings);
        assert_eq!(labels.len(), 4);
        assert!((0.2..=0.7).contains(&th), "threshold {} out of range", th);
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
}
