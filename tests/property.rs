use polyvoice::utils::cosine_similarity;
use proptest::prelude::*;

proptest! {
    /// Property: cosine similarity is always in [-1.0, 1.0].
    #[test]
    fn cosine_similarity_range(a in prop::collection::vec(-1.0f32..=1.0, 256),
                                 b in prop::collection::vec(-1.0f32..=1.0, 256)) {
        let sim = cosine_similarity(&a, &b);
        prop_assert!((-1.0 - 1e-5..=1.0 + 1e-5).contains(&sim),
            "cosine_similarity out of range: {}", sim);
    }

    /// Property: cosine similarity of identical vectors is 1.0 (after normalization).
    #[test]
    fn cosine_similarity_identity(v in prop::collection::vec(-1.0f32..=1.0, 256)) {
        let sim = cosine_similarity(&v, &v);
        prop_assert!((sim - 1.0).abs() < 1e-3,
            "cosine_similarity of identical vectors should be ~1.0, got {}", sim);
    }

    /// Property: cosine similarity is symmetric.
    #[test]
    fn cosine_similarity_symmetric(a in prop::collection::vec(-1.0f32..=1.0, 256),
                                    b in prop::collection::vec(-1.0f32..=1.0, 256)) {
        let sim_ab = cosine_similarity(&a, &b);
        let sim_ba = cosine_similarity(&b, &a);
        prop_assert!((sim_ab - sim_ba).abs() < 1e-5,
            "cosine_similarity is not symmetric: {} vs {}", sim_ab, sim_ba);
    }

    /// Property: orthogonal vectors have similarity near 0.
    #[test]
    fn cosine_similarity_orthogonal(i in 0usize..256, j in 0usize..256) {
        prop_assume!(i != j);
        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        a[i] = 1.0;
        b[j] = 1.0;
        let sim = cosine_similarity(&a, &b);
        prop_assert!(sim.abs() < 1e-5,
            "orthogonal vectors should have similarity ~0, got {}", sim);
    }

    /// Property: cluster assigns same speaker to identical embeddings.
    #[test]
    fn cluster_assign_idempotent(val in -1.0f32..=1.0f32) {
        use polyvoice::{SpeakerCluster, DiarizationConfig};
        let mut cluster = SpeakerCluster::new(DiarizationConfig::default());
        let mut emb = vec![val; 256];
        polyvoice::utils::l2_normalize(&mut emb);
        let (id1, _) = cluster.assign(&emb);
        let (id2, _) = cluster.assign(&emb);
        prop_assert_eq!(id1, id2, "identical embeddings should get same speaker id");
    }

    /// Property: cluster never exceeds max_speakers.
    #[test]
    fn cluster_max_speakers_boundary(max_speakers in 1usize..=16) {
        use polyvoice::{SpeakerCluster, DiarizationConfig};
        let config = DiarizationConfig {
            max_speakers,
            ..DiarizationConfig::default()
        };
        let mut cluster = SpeakerCluster::new(config);
        for i in 0..(max_speakers + 5) {
            let mut emb = vec![0.0f32; 256];
            emb[i % 256] = 1.0;
            cluster.assign(&emb);
        }
        prop_assert!(
            cluster.num_speakers() <= max_speakers,
            "num_speakers {} > max_speakers {}",
            cluster.num_speakers(),
            max_speakers
        );
    }

    /// Property: non-overlapping segments produce no overlaps.
    #[test]
    fn overlap_no_false_positives(
        starts in prop::collection::vec(0.0f64..100.0, 5),
        durations in prop::collection::vec(0.1f64..5.0, 5),
    ) {
        use polyvoice::{detect_overlaps, Segment, SpeakerId, TimeRange};
        let mut segments = Vec::new();
        let mut t = 0.0f64;
        for (start, dur) in starts.iter().zip(durations.iter()) {
            let s = t + start.abs();
            let e = s + dur.abs();
            segments.push(Segment {
                time: TimeRange { start: s, end: e },
                speaker: Some(SpeakerId(segments.len() as u32)),
                confidence: None,
            });
            t = e;
        }
        let overlaps = detect_overlaps(&segments);
        prop_assert!(overlaps.is_empty(), "non-overlapping segments should produce no overlaps");
    }

    /// Property: fbank output shape invariant.
    #[test]
    fn fbank_shape_invariant(extra_samples in 0usize..16000) {
        use polyvoice::features::{compute_fbank, FbankConfig};
        let config = FbankConfig::default();
        let samples = vec![0.0f32; config.win_length + extra_samples];
        let fb = compute_fbank(&samples, &config).unwrap();
        if !fb.is_empty() {
            prop_assert!(
                fb.iter().all(|f| f.len() == config.n_mels),
                "every frame must have n_mels={} coefficients", config.n_mels
            );
        }
    }
}
