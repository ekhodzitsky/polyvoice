//! K-Means++ clustering with automatic k selection via silhouette score.


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


