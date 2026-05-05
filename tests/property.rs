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
}
