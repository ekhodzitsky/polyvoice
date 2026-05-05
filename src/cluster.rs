//! Speaker clustering with online incremental centroid updates.

use crate::types::{DiarizationConfig, SpeakerId};
use crate::utils::{cosine_similarity, l2_normalize};

/// State for a single speaker centroid.
#[derive(Debug, Clone)]
struct Centroid {
    /// Running mean vector (L2-normalized after each update).
    vector: Vec<f32>,
    /// Number of assigned embeddings.
    count: usize,
    /// Total accumulated confidence (sum of similarities).
    confidence_sum: f32,
}

/// Online incremental speaker clusterer.
///
/// Maintains a set of speaker centroids and assigns incoming embeddings to the
/// nearest centroid if the cosine similarity exceeds the configured threshold.
/// Otherwise creates a new speaker identity.
pub struct SpeakerCluster {
    centroids: Vec<Centroid>,
    config: DiarizationConfig,
}

impl SpeakerCluster {
    /// Create an empty clusterer with the given configuration.
    pub fn new(config: DiarizationConfig) -> Self {
        Self {
            centroids: Vec::new(),
            config,
        }
    }

    /// Assign an embedding to a speaker ID.
    ///
    /// Returns the speaker ID and the confidence (cosine similarity) of the assignment.
    pub fn assign(&mut self, embedding: &[f32]) -> (SpeakerId, f32) {
        let mut best_id: Option<usize> = None;
        let mut best_sim = f32::NEG_INFINITY;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, &centroid.vector);
            if sim > best_sim {
                best_sim = sim;
                best_id = Some(i);
            }
        }

        // At speaker limit — always assign to closest centroid regardless of threshold
        if self.centroids.len() >= self.config.max_speakers {
            let id = best_id.unwrap_or(0);
            self.update_centroid(id, embedding, best_sim);
            return (SpeakerId(id as u32), best_sim);
        }

        if let Some(id) = best_id
            && best_sim > self.config.threshold
        {
            self.update_centroid(id, embedding, best_sim);
            return (SpeakerId(id as u32), best_sim);
        }

        // New speaker
        let new_id = self.centroids.len();
        self.centroids.push(Centroid {
            vector: embedding.to_vec(),
            count: 1,
            confidence_sum: 1.0,
        });
        (SpeakerId(new_id as u32), 1.0)
    }

    /// Return the number of distinct speakers seen so far.
    pub fn num_speakers(&self) -> usize {
        self.centroids.len()
    }

    /// Return centroids as immutable slices.
    pub fn centroids(&self) -> Vec<(SpeakerId, &[f32], f32)> {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let avg_conf = if c.count > 0 {
                    c.confidence_sum / c.count as f32
                } else {
                    0.0
                };
                (SpeakerId(i as u32), c.vector.as_slice(), avg_conf)
            })
            .collect()
    }

    /// Merge two speakers (used by post-processing or re-clustering).
    pub fn merge(&mut self, from: SpeakerId, into: SpeakerId) {
        let from_idx = from.0 as usize;
        let into_idx = into.0 as usize;
        if from_idx >= self.centroids.len() || into_idx >= self.centroids.len() || from_idx == into_idx {
            return;
        }
        let from_c = self.centroids.remove(from_idx);
        // After removal, if into_idx was after from_idx, it shifts left by one.
        let adjusted_into = if into_idx > from_idx { into_idx - 1 } else { into_idx };
        if adjusted_into >= self.centroids.len() {
            return;
        }
        // Note: this invalidates SpeakerIds. Use only in offline context where IDs are remapped.
        let into_c = &mut self.centroids[adjusted_into];
        let total_count = into_c.count + from_c.count;
        for (i, v) in into_c.vector.iter_mut().enumerate() {
            *v = (*v * into_c.count as f32 + from_c.vector[i] * from_c.count as f32) / total_count as f32;
        }
        l2_normalize(&mut into_c.vector);
        into_c.count = total_count;
        into_c.confidence_sum += from_c.confidence_sum;
    }

    fn update_centroid(&mut self, id: usize, embedding: &[f32], sim: f32) {
        let c = &mut self.centroids[id];
        let n = c.count as f32;
        for (vec, &emb) in c.vector.iter_mut().zip(embedding.iter()) {
            *vec = (*vec * n + emb) / (n + 1.0);
        }
        l2_normalize(&mut c.vector);
        c.count += 1;
        c.confidence_sum += sim;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(val: f32, dim: usize) -> Vec<f32> {
        let mut v = vec![val; dim];
        l2_normalize(&mut v);
        v
    }

    #[test]
    fn test_new_speaker() {
        let mut cluster = SpeakerCluster::new(DiarizationConfig::default());
        let e = emb(1.0, 256);
        let (id, conf) = cluster.assign(&e);
        assert_eq!(id.0, 0);
        assert!((conf - 1.0).abs() < 1e-5);
        assert_eq!(cluster.num_speakers(), 1);
    }

    #[test]
    fn test_same_speaker() {
        let mut cluster = SpeakerCluster::new(DiarizationConfig::default());
        let e1 = emb(1.0, 256);
        let mut e2 = e1.clone();
        e2[0] += 0.001;
        l2_normalize(&mut e2);

        let (id1, _) = cluster.assign(&e1);
        let (id2, _) = cluster.assign(&e2);
        assert_eq!(id1, id2);
        assert_eq!(cluster.num_speakers(), 1);
    }

    #[test]
    fn test_different_speakers() {
        let mut cluster = SpeakerCluster::new(DiarizationConfig::default());
        let mut e1 = vec![0.0f32; 256];
        e1[0] = 1.0;
        let mut e2 = vec![0.0f32; 256];
        e2[1] = 1.0;

        let (id1, _) = cluster.assign(&e1);
        let (id2, _) = cluster.assign(&e2);
        assert_ne!(id1, id2);
        assert_eq!(cluster.num_speakers(), 2);
    }

    #[test]
    fn test_max_speakers_limit() {
        let config = DiarizationConfig {
            max_speakers: 2,
            ..Default::default()
        };
        let mut cluster = SpeakerCluster::new(config);
        let e1 = emb(1.0, 256);
        let e2 = emb(-1.0, 256);
        let e3 = emb(0.5, 256);

        cluster.assign(&e1);
        cluster.assign(&e2);
        // Third speaker should be forced into closest existing cluster
        let (id3, _) = cluster.assign(&e3);
        assert!(id3.0 < 2);
        assert_eq!(cluster.num_speakers(), 2);
    }
}
