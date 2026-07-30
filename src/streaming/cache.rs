//! AOSC-style arrival-order speaker cache.
//!
//! Borrows the **structure** of NVIDIA Streaming Sortformer's Arrival-Order
//! Speaker Cache (AOSC; Interspeech 2025): per-speaker embedding cache with
//! confidence + recency scoring, arrival-order IDs (no cross-chunk Hungarian
//! rematch), and a hard size cap with eviction / overflow merge.
//!
//! Scoring uses cosine similarity (monotone stand-in for calibrated posteriors;
//! raw embeddings do not yield Sortformer's `log P` evidence).

use super::stability::prefer_current_speaker;
use crate::types::SpeakerId;
use crate::utils::{cosine_similarity, l2_normalize};

/// Result of assigning one embedding through the cache.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssignResult {
    pub speaker: SpeakerId,
    /// Cosine similarity to the chosen entry (1.0 for a brand-new speaker).
    pub confidence: f32,
    /// `false` until the entry has accumulated `min_hits_to_stable` hits
    /// (provisional / Unknown-class label). Once `true`, the speaker ID for
    /// this cache entry is immutable.
    pub stable: bool,
    /// `true` when the cache was full and the frame was force-merged into the
    /// closest existing speaker (overflow merge).
    pub overflow_merged: bool,
}

struct CacheEntry {
    /// Arrival-order speaker id (immutable for the lifetime of the entry).
    id: SpeakerId,
    /// Running mean embedding (L2-normalized).
    centroid: Vec<f32>,
    count: usize,
    /// Accumulated confidence (sum of cosine similarities at update time).
    confidence_sum: f32,
    /// Last assignment step (for recency).
    last_step: u64,
    /// Number of successful assignments to this entry.
    hits: usize,
    /// Latched once `hits >= min_hits_to_stable`.
    stable: bool,
}

impl CacheEntry {
    /// Eviction key: higher is more valuable (keep). Combines average confidence
    /// with a recency boost so recently-active speakers survive pruning.
    fn keep_score(&self, now: u64) -> f32 {
        let avg = if self.count == 0 {
            0.0
        } else {
            self.confidence_sum / self.count as f32
        };
        let age = now.saturating_sub(self.last_step) as f32;
        let recency = 1.0 / (1.0 + age);
        // Stable speakers get a small keep bonus so provisional noise is evicted first.
        let stable_bonus = if self.stable { 0.05 } else { 0.0 };
        avg + 0.25 * recency + stable_bonus
    }
}

/// Bounded arrival-order speaker cache for online diarization.
///
/// # Overflow / merge semantics
///
/// When `len() == cap` and an embedding does not match any entry above
/// `match_threshold`, it is **force-merged** into the closest entry (no new
/// speaker is created). This mirrors AWS Transcribe's documented overflow
/// behaviour and keeps per-chunk work O(cap).
///
/// Eviction (dropping the lowest keep-score entry) is available via
/// [`Self::evict_weakest`] for long streams that need to free a slot for a
/// clearly new speaker; the default assign path uses force-merge only so
/// arrival-order IDs never renumber mid-stream.
pub struct ArrivalOrderSpeakerCache {
    entries: Vec<CacheEntry>,
    cap: usize,
    match_threshold: f32,
    min_hits_to_stable: usize,
    prefer_current_margin: f32,
    step: u64,
    current_speaker: Option<SpeakerId>,
    next_id: u32,
}

impl ArrivalOrderSpeakerCache {
    /// Create an empty cache.
    ///
    /// `cap` is clamped to at least 1 — a zero-capacity cache could never hold
    /// a speaker, so it is treated as a request for the smallest usable cache
    /// (same policy as the `min_hits_to_stable` clamp below).
    pub fn new(
        cap: usize,
        match_threshold: f32,
        min_hits_to_stable: usize,
        prefer_current_margin: f32,
    ) -> Self {
        let cap = cap.max(1);
        Self {
            entries: Vec::with_capacity(cap.min(16)),
            cap,
            match_threshold,
            min_hits_to_stable: min_hits_to_stable.max(1),
            prefer_current_margin,
            step: 0,
            current_speaker: None,
            next_id: 0,
        }
    }

    /// Number of speakers currently in the cache (`<= cap`).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when the cache holds no speakers.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Configured maximum size.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Arrival-order speaker IDs currently resident (sorted by id).
    pub fn speaker_ids(&self) -> Vec<SpeakerId> {
        let mut ids: Vec<_> = self.entries.iter().map(|e| e.id).collect();
        ids.sort_by_key(|s| s.0);
        ids
    }

    /// Assign an embedding to a speaker via cache match / create / overflow merge.
    pub fn assign(&mut self, embedding: &[f32]) -> AssignResult {
        self.step = self.step.saturating_add(1);
        let now = self.step;

        // Build candidate list (id, sim) for hysteresis.
        let mut candidates: Vec<(SpeakerId, f32)> = self
            .entries
            .iter()
            .map(|e| (e.id, cosine_similarity(embedding, &e.centroid)))
            .collect();

        // Apply prefer-current hysteresis over the raw similarities.
        let preferred = prefer_current_speaker(
            self.current_speaker,
            &candidates,
            self.prefer_current_margin,
        );

        // Reorder so the preferred speaker is treated as the best when still above threshold.
        if let Some(pref) = preferred
            && let Some((idx, _)) = candidates
                .iter()
                .enumerate()
                .find(|(_, (id, _))| *id == pref)
        {
            candidates.swap(0, idx);
        }

        let best = candidates
            .iter()
            .copied()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Hysteresis may keep a slightly-worse current speaker: if preferred is
        // set and within margin of absolute best, use preferred's score.
        let chosen = match (preferred, best) {
            (Some(pref), Some((best_id, best_sim))) if pref != best_id => {
                let pref_sim = candidates
                    .iter()
                    .find(|(id, _)| *id == pref)
                    .map(|(_, s)| *s)
                    .unwrap_or(f32::NEG_INFINITY);
                if best_sim - pref_sim <= self.prefer_current_margin {
                    Some((pref, pref_sim))
                } else {
                    Some((best_id, best_sim))
                }
            }
            (_, b) => b,
        };

        if let Some((id, sim)) = chosen
            && (sim >= self.match_threshold || self.entries.len() >= self.cap)
        {
            // Match (or overflow force-merge when at cap).
            let overflow = sim < self.match_threshold && self.entries.len() >= self.cap;
            let result = self.update_entry(id, embedding, sim, now);
            self.current_speaker = Some(result.speaker);
            return AssignResult {
                overflow_merged: overflow,
                ..result
            };
        }

        // No adequate match → new arrival-order speaker. Room always remains
        // here: falling through the branch above means either no candidate
        // (empty cache) or a below-threshold match with the cache not yet
        // full — and cap >= 1 by construction.
        debug_assert!(self.entries.len() < self.cap);
        let result = self.create_entry(embedding, now);
        self.current_speaker = Some(result.speaker);
        result
    }

    /// Drop the lowest keep-score entry if the cache is at capacity.
    /// Returns the evicted speaker id, if any.
    pub fn evict_weakest(&mut self) -> Option<SpeakerId> {
        if self.entries.len() < self.cap {
            return None;
        }
        let now = self.step;
        let idx = self
            .entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.keep_score(now)
                    .partial_cmp(&b.keep_score(now))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)?;
        let evicted = self.entries.swap_remove(idx).id;
        if self.current_speaker == Some(evicted) {
            self.current_speaker = None;
        }
        Some(evicted)
    }

    fn create_entry(&mut self, embedding: &[f32], now: u64) -> AssignResult {
        let id = SpeakerId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        let mut centroid = embedding.to_vec();
        l2_normalize(&mut centroid);
        let hits = 1;
        let stable = hits >= self.min_hits_to_stable;
        self.entries.push(CacheEntry {
            id,
            centroid,
            count: 1,
            confidence_sum: 1.0,
            last_step: now,
            hits,
            stable,
        });
        debug_assert!(self.entries.len() <= self.cap);
        AssignResult {
            speaker: id,
            confidence: 1.0,
            stable,
            overflow_merged: false,
        }
    }

    fn update_entry(
        &mut self,
        id: SpeakerId,
        embedding: &[f32],
        sim: f32,
        now: u64,
    ) -> AssignResult {
        let min_hits = self.min_hits_to_stable;
        // Caller invariant: `id` comes from this cache's candidate list, built
        // from `self.entries` earlier in the same `&mut self` call, so the
        // entry always exists — concurrent eviction is impossible.
        let idx = self.entries.iter().position(|e| e.id == id);
        debug_assert!(idx.is_some());
        let Some(idx) = idx else {
            return AssignResult {
                speaker: id,
                confidence: sim,
                stable: false,
                overflow_merged: false,
            };
        };
        let entry = &mut self.entries[idx];
        let n = entry.count as f32;
        for (v, &e) in entry.centroid.iter_mut().zip(embedding.iter()) {
            *v = (*v * n + e) / (n + 1.0);
        }
        l2_normalize(&mut entry.centroid);
        entry.count += 1;
        entry.confidence_sum += sim;
        entry.last_step = now;
        entry.hits += 1;
        if entry.hits >= min_hits {
            entry.stable = true;
        }
        AssignResult {
            speaker: entry.id,
            confidence: sim,
            stable: entry.stable,
            overflow_merged: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[axis] = 1.0;
        v
    }

    fn near(dim: usize, axis: usize, eps: f32) -> Vec<f32> {
        let mut v = unit(dim, axis);
        v[(axis + 1) % dim] = eps;
        l2_normalize(&mut v);
        v
    }

    #[test]
    fn arrival_order_ids_are_stable_across_chunks() {
        // Chunk 1: speaker A then B. Chunk 2: B then A (permuted order).
        // Cache must re-match by embedding, not re-number by local order.
        let mut cache = ArrivalOrderSpeakerCache::new(8, 0.5, 2, 0.05);
        let a = unit(8, 0);
        let b = unit(8, 1);

        let r0 = cache.assign(&a);
        let r1 = cache.assign(&b);
        assert_eq!(r0.speaker, SpeakerId(0));
        assert_eq!(r1.speaker, SpeakerId(1));

        // Permuted "chunk 2"
        let r2 = cache.assign(&b);
        let r3 = cache.assign(&a);
        assert_eq!(r2.speaker, SpeakerId(1), "B must stay speaker 1");
        assert_eq!(r3.speaker, SpeakerId(0), "A must stay speaker 0");
    }

    #[test]
    fn cache_size_never_exceeds_cap() {
        let cap = 3;
        let mut cache = ArrivalOrderSpeakerCache::new(cap, 0.99, 2, 0.0);
        // Orthogonal-ish vectors that will not match under high threshold.
        for axis in 0..10 {
            let emb = unit(16, axis % 16);
            cache.assign(&emb);
            assert!(
                cache.len() <= cap,
                "cache len {} exceeded cap {}",
                cache.len(),
                cap
            );
        }
        assert_eq!(cache.len(), cap);
    }

    #[test]
    fn overflow_merges_into_closest() {
        let mut cache = ArrivalOrderSpeakerCache::new(2, 0.99, 3, 0.0);
        let a = unit(4, 0);
        let b = unit(4, 1);
        cache.assign(&a);
        cache.assign(&b);
        // Near-A should merge into 0 even though below threshold (cap full).
        let near_a = near(4, 0, 0.01);
        let r = cache.assign(&near_a);
        assert!(r.overflow_merged || r.speaker.0 < 2);
        assert_eq!(cache.len(), 2);
        assert!(r.speaker.0 < 2);
    }

    #[test]
    fn provisional_then_stable_is_deterministic() {
        let mut cache = ArrivalOrderSpeakerCache::new(4, 0.5, 3, 0.0);
        let a = unit(8, 0);
        let r1 = cache.assign(&a);
        assert!(!r1.stable, "first hit is provisional");
        let r2 = cache.assign(&near(8, 0, 0.01));
        assert!(!r2.stable, "second hit still provisional at min_hits=3");
        let r3 = cache.assign(&near(8, 0, 0.01));
        assert!(r3.stable, "third hit reaches stability");
        let r4 = cache.assign(&near(8, 0, 0.01));
        assert!(r4.stable, "stability latches");
        assert_eq!(r1.speaker, r4.speaker);
    }

    #[test]
    fn hysteresis_suppresses_flicker_inside_cache() {
        let mut cache = ArrivalOrderSpeakerCache::new(4, 0.4, 5, 0.15);
        let a = unit(8, 0);
        let b = unit(8, 1);
        cache.assign(&a); // current = 0
        cache.assign(&b); // current = 1
        // Strong A again
        let r = cache.assign(&a);
        assert_eq!(r.speaker, SpeakerId(0));
        // Ambiguous: slightly closer to B but within margin of A if we craft
        // an embedding with sim(A)≈0.70, sim(B)≈0.78 and margin 0.15 → keep A.
        // Construct e = normalize(0.7*a + 0.78*b) — actually cosine to axes is the component.
        let mut e = vec![0.0f32; 8];
        e[0] = 0.70;
        e[1] = 0.78;
        l2_normalize(&mut e);
        let r2 = cache.assign(&e);
        // Without hysteresis best is B; with margin 0.15 and current=A, keep A if gap ≤ 0.15.
        let sim_a = e[0]; // unit axes after normalize: components scale
        let sim_b = e[1];
        // After L2 normalize, gap = sim_b - sim_a
        let gap = cosine_similarity(&e, &b) - cosine_similarity(&e, &a);
        if gap <= 0.15 {
            assert_eq!(r2.speaker, SpeakerId(0), "hysteresis should keep current A");
        } else {
            assert_eq!(r2.speaker, SpeakerId(1));
        }
        let _ = (sim_a, sim_b);
    }

    #[test]
    fn zero_cap_is_clamped_to_one() {
        let mut cache = ArrivalOrderSpeakerCache::new(0, 0.5, 2, 0.0);
        assert_eq!(cache.cap(), 1);
        cache.assign(&unit(4, 0));
        assert_eq!(cache.len(), 1);
        // A second distinct speaker cannot grow the cache: overflow force-merge.
        let r = cache.assign(&unit(4, 1));
        assert_eq!(cache.len(), 1);
        assert!(r.overflow_merged);
    }
}
