//! Speaker clustering with online incremental centroid updates.
//!
//! Assign incoming embeddings to existing speakers or create new ones in a
//! single pass. See [`SpeakerCluster`] and [`ClusterConfig`].

use crate::types::{ClusterConfig, SpeakerId, SpeakerIdRemap};
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
    config: ClusterConfig,
}

impl SpeakerCluster {
    /// { true }
    /// pub fn new(config: ClusterConfig) -> Self
    /// { ret.num_speakers() == 0 }
    /// Create a new empty speaker clusterer.
    ///
    /// ```rust
    /// use polyvoice::{SpeakerCluster, ClusterConfig};
    /// let cluster = SpeakerCluster::new(ClusterConfig::default());
    /// assert_eq!(cluster.num_speakers(), 0);
    /// ```
    pub fn new(config: ClusterConfig) -> Self {
        Self {
            centroids: Vec::new(),
            config,
        }
    }

    /// { !embedding.is_empty() }
    /// pub fn assign(&mut self, embedding: &[f32]) -> (SpeakerId, f32)
    /// { ret.1 >= -1.0 && ret.1 <= 1.0 && ret.0.0 < self.num_speakers() }
    /// Assign an embedding to the closest speaker centroid.
    ///
    /// Returns the speaker ID and the cosine similarity score. If no existing
    /// centroid is close enough (above threshold) and the speaker limit has not
    /// been reached, a new speaker is created.
    ///
    /// ```rust
    /// use polyvoice::{SpeakerCluster, ClusterConfig};
    /// let mut cluster = SpeakerCluster::new(ClusterConfig::default());
    /// let mut emb = vec![0.0f32; 256];
    /// emb[0] = 1.0;
    /// let (id, conf) = cluster.assign(&emb);
    /// assert_eq!(id.0, 0);
    /// assert!(conf > 0.0);
    /// ```
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

    /// { true }
    /// pub fn num_speakers(&self) -> usize
    /// { ret <= self.config.max_speakers }
    /// Return the current number of speaker centroids.
    ///
    /// ```rust
    /// use polyvoice::{SpeakerCluster, ClusterConfig};
    /// let cluster = SpeakerCluster::new(ClusterConfig::default());
    /// assert_eq!(cluster.num_speakers(), 0);
    /// ```
    pub fn num_speakers(&self) -> usize {
        self.centroids.len()
    }

    /// { true }
    /// `pub fn centroids(&self) -> Vec<(SpeakerId, &[f32], f32)>`
    /// { ret.len() == self.num_speakers() }
    /// Return a view of all current centroids.
    ///
    /// Each tuple contains `(SpeakerId, centroid_vector, average_confidence)`.
    ///
    /// ```rust
    /// use polyvoice::{SpeakerCluster, ClusterConfig};
    /// let mut cluster = SpeakerCluster::new(ClusterConfig::default());
    /// let mut emb = vec![0.0f32; 128];
    /// emb[0] = 1.0;
    /// cluster.assign(&emb);
    /// let centroids = cluster.centroids();
    /// assert_eq!(centroids.len(), 1);
    /// assert_eq!(centroids[0].0.0, 0);
    /// ```
    pub fn centroids(&self) -> Vec<(SpeakerId, &[f32], f32)> {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let avg_conf = c.confidence_sum / c.count as f32;
                (SpeakerId(i as u32), c.vector.as_slice(), avg_conf)
            })
            .collect()
    }

    /// { true }
    /// `pub fn merge(&mut self, from: SpeakerId, into: SpeakerId) -> Option<SpeakerIdRemap>`
    /// { ret.is_none() || self.num_speakers() >= 1 }
    /// Merge one speaker centroid into another.
    ///
    /// The `from` centroid is removed and its statistics are averaged into `into`.
    /// Returns a [`SpeakerIdRemap`] describing how existing IDs changed, or `None`
    /// if the indices are invalid or equal.
    ///
    /// ```rust
    /// use polyvoice::{SpeakerCluster, ClusterConfig, SpeakerId};
    /// let mut cluster = SpeakerCluster::new(ClusterConfig::default());
    /// let mut e0 = vec![0.0f32; 128]; e0[0] = 1.0;
    /// let mut e1 = vec![0.0f32; 128]; e1[1] = 1.0;
    /// let (id0, _) = cluster.assign(&e0);
    /// let (id1, _) = cluster.assign(&e1);
    /// let remap = cluster.merge(id1, id0).expect("valid merge");
    /// assert_eq!(cluster.num_speakers(), 1);
    /// assert_eq!(remap.remap(id1), id0);
    /// ```
    pub fn merge(&mut self, from: SpeakerId, into: SpeakerId) -> Option<SpeakerIdRemap> {
        let from_idx = from.0 as usize;
        let into_idx = into.0 as usize;
        let old_len = self.centroids.len();
        if from_idx >= old_len || into_idx >= old_len || from_idx == into_idx {
            return None;
        }
        let from_c = self.centroids.remove(from_idx);
        // After removal, if into_idx was after from_idx, it shifts left by one.
        let adjusted_into = if into_idx > from_idx {
            into_idx - 1
        } else {
            into_idx
        };
        if adjusted_into >= self.centroids.len() {
            return None;
        }
        let into_c = &mut self.centroids[adjusted_into];
        let total_count = into_c.count + from_c.count;
        for (i, v) in into_c.vector.iter_mut().enumerate() {
            *v = (*v * into_c.count as f32 + from_c.vector[i] * from_c.count as f32)
                / total_count as f32;
        }
        l2_normalize(&mut into_c.vector);
        into_c.count = total_count;
        into_c.confidence_sum += from_c.confidence_sum;

        // Build remap: every index >= from_idx shifts left by 1.
        let mut mapping = Vec::with_capacity(old_len - self.centroids.len());
        for old_id in from_idx..old_len {
            let new_id = if old_id == from_idx {
                adjusted_into
            } else {
                old_id - 1
            };
            mapping.push((SpeakerId(old_id as u32), SpeakerId(new_id as u32)));
        }
        Some(SpeakerIdRemap::from_mapping(mapping))
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

#[allow(clippy::unwrap_used)]
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
        let mut cluster = SpeakerCluster::new(ClusterConfig::default());
        let e = emb(1.0, 256);
        let (id, conf) = cluster.assign(&e);
        assert_eq!(id.0, 0);
        assert!((conf - 1.0).abs() < 1e-5);
        assert_eq!(cluster.num_speakers(), 1);
    }

    #[test]
    fn test_same_speaker() {
        let mut cluster = SpeakerCluster::new(ClusterConfig::default());
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
        let mut cluster = SpeakerCluster::new(ClusterConfig::default());
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
        let config = ClusterConfig {
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

    #[test]
    fn test_merge_remaps_speaker_ids() {
        let mut cluster = SpeakerCluster::new(ClusterConfig::default());
        // Three orthogonal unit vectors to guarantee distinct speakers.
        let mut e0 = vec![0.0f32; 256];
        e0[0] = 1.0;
        let mut e1 = vec![0.0f32; 256];
        e1[1] = 1.0;
        let mut e2 = vec![0.0f32; 256];
        e2[2] = 1.0;

        let (id0, _) = cluster.assign(&e0); // SpeakerId(0)
        let (id1, _) = cluster.assign(&e1); // SpeakerId(1)
        let (id2, _) = cluster.assign(&e2); // SpeakerId(2)
        assert_eq!(cluster.num_speakers(), 3);

        // Merge speaker 1 into speaker 0.
        let remap = cluster.merge(id1, id0).expect("merge should succeed");
        assert_eq!(cluster.num_speakers(), 2);

        // id1 should now map to id0.
        assert_eq!(remap.remap(id1), id0);
        // id2 should shift from 2 to 1.
        assert_eq!(remap.remap(id2), SpeakerId(1));
        // id0 should remain unchanged.
        assert_eq!(remap.remap(id0), id0);
    }

    #[test]
    fn test_merge_into_higher_index() {
        let mut cluster = SpeakerCluster::new(ClusterConfig::default());
        let mut e0 = vec![0.0f32; 256];
        e0[0] = 1.0;
        let mut e1 = vec![0.0f32; 256];
        e1[1] = 1.0;
        let mut e2 = vec![0.0f32; 256];
        e2[2] = 1.0;

        let (id0, _) = cluster.assign(&e0);
        let (id1, _) = cluster.assign(&e1);
        let (id2, _) = cluster.assign(&e2);

        // Merge speaker 0 into speaker 2.
        let remap = cluster.merge(id0, id2).expect("merge should succeed");
        assert_eq!(cluster.num_speakers(), 2);

        // id0 maps to id2 (adjusted to index 1 after removal).
        assert_eq!(remap.remap(id0), SpeakerId(1));
        // id1 stays at 0 (was before removed index).
        assert_eq!(remap.remap(id1), SpeakerId(0));
        // id2 shifts from 2 to 1.
        assert_eq!(remap.remap(id2), SpeakerId(1));
    }

    #[test]
    fn test_merge_invalid_returns_none() {
        let mut cluster = SpeakerCluster::new(ClusterConfig::default());
        let e0 = emb(1.0, 256);
        cluster.assign(&e0);

        // Merge into self — invalid.
        assert!(cluster.merge(SpeakerId(0), SpeakerId(0)).is_none());
        // Merge out of range.
        assert!(cluster.merge(SpeakerId(5), SpeakerId(0)).is_none());
        assert!(cluster.merge(SpeakerId(0), SpeakerId(5)).is_none());
    }
}
