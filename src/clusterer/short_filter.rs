//! Short-segment embedding filtering (cVBx-style).
//!
//! Unreliable short embeddings are excluded from AHC / VB iterations and
//! reassigned afterward to the nearest surviving cluster. This is the main
//! speaker-count lever in the cVBx recipe: brief windows produce noisy vectors
//! that spawn spurious singleton clusters.

use crate::utils::cosine_similarity;

/// Split embedding indices into *kept* (≥ `min_secs`) and *short* (< `min_secs`).
///
/// When `min_secs <= 0` every index is kept. When every embedding is short, the
/// single longest is promoted into `kept` so clustering still has an input
/// (ties broken by lowest index).
pub fn partition_by_min_duration(
    durations_secs: &[f64],
    min_secs: f64,
) -> (Vec<usize>, Vec<usize>) {
    if min_secs <= 0.0 || durations_secs.is_empty() {
        return ((0..durations_secs.len()).collect(), Vec::new());
    }
    let mut kept = Vec::new();
    let mut short = Vec::new();
    for (i, &d) in durations_secs.iter().enumerate() {
        if d >= min_secs {
            kept.push(i);
        } else {
            short.push(i);
        }
    }
    if kept.is_empty() {
        // Promote the longest (then lowest index) so clustering is never empty.
        let best = durations_secs
            .iter()
            .enumerate()
            .max_by(|(i, a), (j, b)| a.total_cmp(b).then(j.cmp(i)))
            .map(|(i, _)| i)
            .unwrap_or(0);
        short.retain(|&i| i != best);
        kept.push(best);
    }
    (kept, short)
}

/// Scatter `kept_labels` (one label per kept index) into a full label vector of
/// length `n`, filling short indices by nearest kept embedding under cosine
/// similarity. Labels of kept members are left as-is (not recompacted).
///
/// **Requires:** `embeddings.len() == n`, `kept.len() == kept_labels.len()`,
/// every index in `kept` / `short` is `< n`, and the two index sets are disjoint
/// and cover `0..n` when combined with no duplicates. Empty `kept` returns all
/// zeros.
pub fn reassign_short_by_cosine(
    embeddings: &[Vec<f32>],
    kept: &[usize],
    kept_labels: &[usize],
    short: &[usize],
) -> Vec<usize> {
    let n = embeddings.len();
    let mut out = vec![0usize; n];
    if kept.is_empty() || kept.len() != kept_labels.len() {
        return out;
    }
    for (&idx, &lab) in kept.iter().zip(kept_labels.iter()) {
        if idx < n {
            out[idx] = lab;
        }
    }
    // L2-normalized centroid per distinct kept label; slots with no members
    // stay zero and are skipped by the `counts` check below.
    let max_lab = kept_labels.iter().copied().max().unwrap_or(0);
    let centroids =
        crate::utils::normalized_mean_centroids(embeddings, kept, kept_labels, max_lab + 1);
    let mut counts = vec![0usize; max_lab + 1];
    for (&idx, &lab) in kept.iter().zip(kept_labels.iter()) {
        if idx < n && lab <= max_lab {
            counts[lab] += 1;
        }
    }
    for &idx in short {
        if idx >= n {
            continue;
        }
        let mut best_lab = kept_labels[0];
        let mut best_sim = f32::NEG_INFINITY;
        for lab in 0..=max_lab {
            if counts[lab] == 0 {
                continue;
            }
            let sim = cosine_similarity(&embeddings[idx], &centroids[lab]);
            if sim > best_sim {
                best_sim = sim;
                best_lab = lab;
            }
        }
        out[idx] = best_lab;
    }
    out
}

/// Like [`reassign_short_by_cosine`] but scores in a pre-transformed feature
/// space (e.g. PLDA features). `features[i]` must align with `embeddings` index
/// `i`. Cosine is used on the feature rows (PLDA space is roughly spherical
/// after the transform).
pub fn reassign_short_by_features(
    features: &[Vec<f32>],
    kept: &[usize],
    kept_labels: &[usize],
    short: &[usize],
) -> Vec<usize> {
    reassign_short_by_cosine(features, kept, kept_labels, short)
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    fn ax(a: usize) -> Vec<f32> {
        let mut v = vec![0.02f32, 0.02, 0.02];
        v[a] = 1.0;
        v
    }

    #[test]
    fn partition_keeps_long_drops_short() {
        let durs = vec![2.0, 0.5, 3.0, 1.0];
        let (kept, short) = partition_by_min_duration(&durs, 1.6);
        assert_eq!(kept, vec![0, 2]);
        assert_eq!(short, vec![1, 3]);
    }

    #[test]
    fn partition_zero_threshold_keeps_all() {
        let durs = vec![0.1, 0.2];
        let (kept, short) = partition_by_min_duration(&durs, 0.0);
        assert_eq!(kept, vec![0, 1]);
        assert!(short.is_empty());
    }

    #[test]
    fn partition_all_short_promotes_longest() {
        let durs = vec![0.3, 0.9, 0.5];
        let (kept, short) = partition_by_min_duration(&durs, 1.6);
        assert_eq!(kept, vec![1]);
        assert_eq!(short, vec![0, 2]);
    }

    #[test]
    fn reassign_maps_short_to_nearest_cluster() {
        // kept: two long axes; short: near axis 0.
        let embeddings = vec![ax(0), ax(0), ax(1), ax(1), ax(0)];
        let kept = vec![0, 1, 2, 3];
        let kept_labels = vec![0, 0, 1, 1];
        let short = vec![4];
        let out = reassign_short_by_cosine(&embeddings, &kept, &kept_labels, &short);
        assert_eq!(out, vec![0, 0, 1, 1, 0]);
    }

    #[test]
    fn reassign_short_near_axis1_goes_to_cluster1() {
        let embeddings = vec![ax(0), ax(0), ax(1), ax(1), ax(1)];
        let kept = vec![0, 1, 2, 3];
        let kept_labels = vec![0, 0, 1, 1];
        let short = vec![4];
        let out = reassign_short_by_cosine(&embeddings, &kept, &kept_labels, &short);
        assert_eq!(out[4], 1);
    }

    #[test]
    fn filter_then_cluster_avoids_spurious_short_singleton() {
        // Three long embeddings of speaker A, three of B, plus one short noisy
        // singleton that would form its own AHC cluster if included.
        let mut embeddings = vec![ax(0), ax(0), ax(0), ax(1), ax(1), ax(1)];
        // Noisy short near origin — cosine to either axis is weak but non-zero.
        embeddings.push(vec![0.3, 0.3, 0.3]);
        let durs = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.4];
        let (kept, short) = partition_by_min_duration(&durs, 1.6);
        assert_eq!(kept.len(), 6);
        assert_eq!(short, vec![6]);
        let kept_embs: Vec<Vec<f32>> = kept.iter().map(|&i| embeddings[i].clone()).collect();
        let kept_labels = crate::ahc::agglomerative_cluster(&kept_embs, 0.5);
        let full = reassign_short_by_cosine(&embeddings, &kept, &kept_labels, &short);
        let n_speakers = full
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert_eq!(n_speakers, 2, "short must not create a third speaker");
        assert!(full[6] == full[0] || full[6] == full[3]);
    }
}
