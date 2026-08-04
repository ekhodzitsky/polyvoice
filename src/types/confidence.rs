//! Confidence heuristics and per-speaker embedding means.
use super::ids::SpeakerId;

/// Default midpoint for [`confidence_from_similarity`] (cosine similarity scale).
pub const CONFIDENCE_SIM_MIDPOINT: f32 = 0.5;
/// Default steepness for the logistic confidence curve.
pub const CONFIDENCE_SIM_STEEPNESS: f32 = 10.0;

/// Map a cosine similarity in `[-1, 1]` to a confidence score in `(0, 1]`.
///
/// Uses a fixed logistic curve centered at [`CONFIDENCE_SIM_MIDPOINT`] with
/// slope [`CONFIDENCE_SIM_STEEPNESS`]. **Monotone increasing** in similarity
/// (equivalently monotone decreasing in cosine distance `1 − sim`).
///
/// This is a cheap heuristic for ranking / low-confidence triage, **not** a
/// calibrated probability of label correctness. Full isotonic calibration needs
/// labeled dev data and is intentionally not hard-coded here.
///
/// Non-finite inputs map to confidence near 0.
pub fn confidence_from_similarity(sim: f32) -> f32 {
    confidence_from_similarity_params(sim, CONFIDENCE_SIM_MIDPOINT, CONFIDENCE_SIM_STEEPNESS)
}

/// Logistic confidence from cosine similarity with explicit midpoint/steepness.
///
/// `steepness` must be positive for the intended "higher sim → higher conf"
/// direction; non-positive values are treated as [`CONFIDENCE_SIM_STEEPNESS`].
pub fn confidence_from_similarity_params(sim: f32, midpoint: f32, steepness: f32) -> f32 {
    let s = if sim.is_finite() {
        sim.clamp(-1.0, 1.0)
    } else {
        -1.0
    };
    let k = if steepness.is_finite() && steepness > 0.0 {
        steepness
    } else {
        CONFIDENCE_SIM_STEEPNESS
    };
    let m = if midpoint.is_finite() {
        midpoint
    } else {
        CONFIDENCE_SIM_MIDPOINT
    };
    let x = k * (s - m);
    // sigmoid(x) = 1 / (1 + e^{-x}); clamp for numerical safety.
    let conf = if x >= 20.0 {
        1.0
    } else if x <= -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    };
    conf.clamp(0.0, 1.0)
}

/// Confidence from cosine distance `d = 1 − sim` (L2-normalized embeddings).
///
/// Monotone **decreasing** in `distance`. Equivalent to
/// [`confidence_from_similarity`]`(1 − distance)`.
pub fn confidence_from_distance(distance: f32) -> f32 {
    let d = if distance.is_finite() { distance } else { 2.0 };
    confidence_from_similarity(1.0 - d)
}

/// Mean L2-normalized embedding per speaker from parallel label/embedding slices.
///
/// Speakers appear sorted by numeric id. Empty / mismatched input yields an empty
/// vec. Each output vector is L2-normalized.
pub fn mean_speaker_embeddings(
    labels: &[SpeakerId],
    embeddings: &[Vec<f32>],
) -> Vec<(SpeakerId, Vec<f32>)> {
    use std::collections::BTreeMap;
    if labels.is_empty() || embeddings.is_empty() {
        return Vec::new();
    }
    let n = labels.len().min(embeddings.len());
    let mut sums: BTreeMap<u32, (Vec<f32>, usize)> = BTreeMap::new();
    for i in 0..n {
        let emb = &embeddings[i];
        if emb.is_empty() || emb.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let id = labels[i].0;
        let entry = sums.entry(id).or_insert_with(|| (vec![0.0; emb.len()], 0));
        if entry.0.len() != emb.len() {
            continue; // dimension mismatch — skip
        }
        for (s, &v) in entry.0.iter_mut().zip(emb.iter()) {
            *s += v;
        }
        entry.1 += 1;
    }
    sums.into_iter()
        .filter_map(|(id, (mut sum, count))| {
            if count == 0 {
                return None;
            }
            let inv = 1.0 / count as f32;
            for v in &mut sum {
                *v *= inv;
            }
            crate::utils::l2_normalize(&mut sum);
            Some((SpeakerId(id), sum))
        })
        .collect()
}

/// Per-embedding confidence from cosine similarity to the speaker's mean centroid.
///
/// Returns one score per pair in `labels.zip(embeddings)` (length
/// `min(labels.len(), embeddings.len())`). Embeddings whose label has no usable
/// centroid get confidence `0.0`.
pub fn segment_confidences_from_embeddings(
    labels: &[SpeakerId],
    embeddings: &[Vec<f32>],
) -> Vec<f32> {
    let centroids = mean_speaker_embeddings(labels, embeddings);
    let n = labels.len().min(embeddings.len());
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let Some((_, centroid)) = centroids.iter().find(|(id, _)| *id == labels[i]) else {
            continue;
        };
        let sim = crate::utils::cosine_similarity(&embeddings[i], centroid);
        out[i] = confidence_from_similarity(sim);
    }
    out
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_is_monotone_in_similarity_and_bounded() {
        let mut prev = 0.0f32;
        for i in -10..=10 {
            let sim = i as f32 / 10.0;
            let c = confidence_from_similarity(sim);
            assert!((0.0..=1.0).contains(&c), "sim={sim}");
            assert!(c >= prev, "sim={sim}");
            prev = c;
        }
        // The midpoint similarity maps to exactly 0.5.
        assert!((confidence_from_similarity(CONFIDENCE_SIM_MIDPOINT) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn non_finite_similarity_maps_near_zero() {
        assert!(confidence_from_similarity(f32::NAN) < 0.01);
        assert!(confidence_from_similarity(f32::INFINITY) < 0.01);
        assert!(confidence_from_similarity(f32::NEG_INFINITY) < 0.01);
    }

    #[test]
    fn extreme_logit_saturates_to_zero_and_one() {
        assert_eq!(confidence_from_similarity_params(1.0, 0.5, 100.0), 1.0);
        assert_eq!(confidence_from_similarity_params(-1.0, 0.5, 100.0), 0.0);
    }

    #[test]
    fn invalid_params_fall_back_to_defaults() {
        let reference = confidence_from_similarity(0.7);
        // Non-positive or non-finite steepness falls back to the default slope.
        assert_eq!(confidence_from_similarity_params(0.7, 0.5, -1.0), reference);
        assert_eq!(confidence_from_similarity_params(0.7, 0.5, 0.0), reference);
        assert_eq!(
            confidence_from_similarity_params(0.7, 0.5, f32::NAN),
            reference
        );
        // Non-finite midpoint falls back to the default midpoint.
        assert_eq!(
            confidence_from_similarity_params(0.7, f32::INFINITY, 10.0),
            reference
        );
        // Out-of-range similarity is clamped into [-1, 1].
        assert_eq!(
            confidence_from_similarity_params(5.0, 0.5, 10.0),
            confidence_from_similarity_params(1.0, 0.5, 10.0)
        );
    }

    #[test]
    fn distance_confidence_decreases_in_distance() {
        assert_eq!(
            confidence_from_distance(0.3),
            confidence_from_similarity(0.7)
        );
        let near = confidence_from_distance(0.1);
        let far = confidence_from_distance(0.9);
        assert!(near > far);
        // Non-finite distance is treated as maximally far.
        assert!(confidence_from_distance(f32::NAN) < 0.01);
    }

    #[test]
    fn mean_embeddings_empty_inputs_yield_empty() {
        assert!(mean_speaker_embeddings(&[], &[]).is_empty());
        assert!(mean_speaker_embeddings(&[SpeakerId(0)], &[]).is_empty());
    }

    #[test]
    fn mean_embeddings_skip_bad_vectors_and_sort_by_id() {
        let labels = [SpeakerId(1), SpeakerId(0), SpeakerId(1), SpeakerId(1)];
        let embeddings = vec![
            vec![f32::NAN, 0.0], // non-finite — skipped
            vec![1.0, 0.0],      // speaker 0
            vec![0.0, 1.0],      // speaker 1 (first usable, fixes dim 2)
            vec![0.0, 1.0, 0.0], // dimension mismatch — skipped
        ];
        let means = mean_speaker_embeddings(&labels, &embeddings);
        assert_eq!(means.len(), 2);
        assert_eq!(means[0].0, SpeakerId(0));
        assert_eq!(means[1].0, SpeakerId(1));
        // Every mean is L2-normalized.
        for (_, v) in &means {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
        }
        // Speaker 0's single usable embedding is its own mean.
        assert!((means[0].1[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mean_embeddings_average_repeated_speakers() {
        let labels = [SpeakerId(0), SpeakerId(0)];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let means = mean_speaker_embeddings(&labels, &embeddings);
        assert_eq!(means.len(), 1);
        let v = &means[0].1;
        // Mean of [1,0] and [0,1] is [0.5, 0.5], normalized to [1/√2, 1/√2].
        let expect = 1.0 / 2.0f32.sqrt();
        assert!((v[0] - expect).abs() < 1e-6);
        assert!((v[1] - expect).abs() < 1e-6);
    }

    #[test]
    fn segment_confidences_score_against_own_centroid() {
        let labels = [SpeakerId(0), SpeakerId(0), SpeakerId(1)];
        let embeddings = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];
        let confs = segment_confidences_from_embeddings(&labels, &embeddings);
        assert_eq!(confs.len(), 3);
        assert!(confs.iter().all(|c| (0.0..=1.0).contains(c)));
        // Close-to-centroid embeddings rank confidently.
        assert!(confs[0] > 0.5);
        assert!(confs[1] > 0.5);
        // Mismatched slice lengths truncate to the shorter side.
        let confs = segment_confidences_from_embeddings(&labels[..2], &embeddings);
        assert_eq!(confs.len(), 2);
    }
}
