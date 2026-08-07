use super::*;

fn tone_1s() -> Vec<f32> {
    vec![0.1_f32; 16_000]
}

#[test]
fn dim_is_reported() {
    let e = DummyExtractor::new(192);
    assert_eq!(e.dim(), 192);
}

#[test]
fn embed_returns_l2_normalized_vector_of_dim() {
    let e = DummyExtractor::new(256);
    let v = e.embed(&tone_1s()).unwrap();
    assert_eq!(v.len(), 256);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "expected unit vector, |v|={norm}"
    );
}

#[test]
fn successive_embeds_differ() {
    let e = DummyExtractor::new(64);
    let a = e.embed(&tone_1s()).unwrap();
    let b = e.embed(&tone_1s()).unwrap();
    assert_ne!(a, b, "the internal seed advances between calls");
}

#[test]
fn fresh_instances_reproduce_the_same_sequence() {
    let e1 = DummyExtractor::new(32);
    let e2 = DummyExtractor::new(32);
    for _ in 0..3 {
        assert_eq!(e1.embed(&[]).unwrap(), e2.embed(&[]).unwrap());
    }
}

#[test]
fn zero_dim_extractor_returns_empty_embedding() {
    let e = DummyExtractor::new(0);
    assert_eq!(e.dim(), 0);
    assert!(e.embed(&tone_1s()).unwrap().is_empty());
}

#[test]
fn default_batch_embeds_each_input() {
    let e = DummyExtractor::new(16);
    let audio = tone_1s();
    let inputs: Vec<&[f32]> = vec![&audio, &audio];
    let out = e.embed_batch(&inputs).unwrap();
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|v| v.len() == 16));
}
