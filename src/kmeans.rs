//! K-Means++ clustering with automatic k selection via silhouette score.

use crate::utils::cosine_similarity;

/// { TODO: precondition }
/// pub fn kmeans_auto_k(embeddings: &[Vec<f32>], max_k: usize, max_iter: usize) -> Vec<usize>
/// { TODO: postcondition }
/// Run K-means++ with automatic k selection.
///
/// Searches k in 1..=max_k and picks the one with the highest silhouette score.
/// Falls back to k=1 if all scores are negative.
pub fn kmeans_auto_k(embeddings: &[Vec<f32>], max_k: usize, max_iter: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    // Cap k at a reasonable value for diarization (rarely > 10 speakers).
    let max_k = max_k.min(n).min(10);
    let max_iter = max_iter.min(20);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_labels = vec![0usize; n];

    for k in 1..=max_k {
        let labels = kmeans_pp(embeddings, k, max_iter);
        let score = silhouette_score(embeddings, &labels);
        if score > best_score {
            best_score = score;
            best_labels = labels;
        }
    }

    best_labels
}

/// { TODO: precondition }
/// pub fn kmeans_pp(embeddings: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<usize>
/// { TODO: postcondition }
/// K-means++ initialization + Lloyd's algorithm.
pub fn kmeans_pp(embeddings: &[Vec<f32>], k: usize, max_iter: usize) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let dim = embeddings[0].len();

    // K-means++ initialization.
    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut rng = fastrand::Rng::new();
    let first_idx = rng.usize(0..n);
    centroids.push(embeddings[first_idx].iter().map(|&v| v as f64).collect());

    let mut dists = vec![f64::INFINITY; n];
    for _ in 1..k {
        for (i, emb) in embeddings.iter().enumerate() {
            let d = cosine_distance_f32_f64(emb, &centroids[centroids.len() - 1]);
            if d < dists[i] {
                dists[i] = d;
            }
        }
        let total: f64 = dists.iter().sum();
        let target = rng.f64() * total;
        let mut cumsum = 0.0;
        let mut chosen = 0;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= target {
                chosen = i;
                break;
            }
        }
        centroids.push(embeddings[chosen].iter().map(|&v| v as f64).collect());
    }

    // Lloyd's algorithm.
    let mut labels = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        // Assign.
        for (i, emb) in embeddings.iter().enumerate() {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for (c_idx, c) in centroids.iter().enumerate() {
                let dist = cosine_distance_f32_f64(emb, c);
                if dist < best_dist {
                    best_dist = dist;
                    best = c_idx;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update.
        let mut new_centroids = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, emb) in embeddings.iter().enumerate() {
            let c = labels[i];
            for (d, &v) in emb.iter().enumerate() {
                new_centroids[c][d] += v as f64;
            }
            counts[c] += 1;
        }
        for (c, new_centroid) in new_centroids.iter_mut().enumerate().take(k) {
            if counts[c] > 0 {
                for v in new_centroid.iter_mut().take(dim) {
                    *v /= counts[c] as f64;
                }
            }
        }
        centroids = new_centroids;
    }

    labels
}

fn cosine_distance_f32_f64(a: &[f32], b: &[f64]) -> f64 {
    let sim = crate::utils::cosine_similarity_f32_f64(a, b);
    (1.0 - sim).max(0.0) as f64
}

fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f64 {
    let sim = cosine_similarity(a, b);
    (1.0 - sim).max(0.0) as f64
}

fn silhouette_score(embeddings: &[Vec<f32>], labels: &[usize]) -> f64 {
    let n = embeddings.len();
    if n < 2 {
        return 0.0;
    }

    let mut total = 0.0;
    let mut count = 0usize;
    let unique_labels: std::collections::HashSet<_> = labels.iter().copied().collect();

    for i in 0..n {
        let label = labels[i];
        let mut a = 0.0f64; // avg distance to same cluster
        let mut a_count = 0usize;
        let mut b = f64::INFINITY; // min avg distance to other clusters

        for j in 0..n {
            if i == j {
                continue;
            }
            let dist = cosine_distance_f32(&embeddings[i], &embeddings[j]);
            if labels[j] == label {
                a += dist;
                a_count += 1;
            }
        }

        if a_count == 0 {
            continue;
        }
        a /= a_count as f64;

        for &other_label in &unique_labels {
            if other_label == label {
                continue;
            }
            let mut b_sum = 0.0f64;
            let mut b_count = 0usize;
            for j in 0..n {
                if labels[j] == other_label {
                    let dist = cosine_distance_f32(&embeddings[i], &embeddings[j]);
                    b_sum += dist;
                    b_count += 1;
                }
            }
            if b_count > 0 {
                let avg = b_sum / b_count as f64;
                if avg < b {
                    b = avg;
                }
            }
        }

        if b.is_finite() {
            let s = (b - a) / a.max(b);
            total += s;
            count += 1;
        }
    }

    if count > 0 {
        total / count as f64
    } else {
        -1.0
    }
}
