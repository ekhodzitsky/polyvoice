use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Counts how many times `embed` was called.
struct CountingEmbedder {
    counter: Arc<AtomicUsize>,
    dim: usize,
}

impl Embedder for CountingEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        Ok(vec![0.0; self.dim])
    }
}

fn make_pool(n: usize) -> (EmbedderPool<CountingEmbedder>, Arc<AtomicUsize>) {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut embedders = Vec::with_capacity(n);
    for _ in 0..n {
        embedders.push(CountingEmbedder {
            counter: counter.clone(),
            dim: 192,
        });
    }
    let pool = EmbedderPool::new(embedders).unwrap();
    (pool, counter)
}

#[test]
fn pool_with_single_embedder_round_trip() {
    let (pool, counter) = make_pool(1);
    let result = pool.embed(&[0.0_f32; 100]).unwrap();
    assert_eq!(result.len(), 192);
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}

#[test]
fn pool_dim_is_consistent() {
    let (pool, _) = make_pool(4);
    assert_eq!(pool.dim(), 192);
}

#[test]
fn pool_serial_embed_increments_counter_per_call() {
    let (pool, counter) = make_pool(2);
    for _ in 0..5 {
        pool.embed(&[0.0_f32; 100]).unwrap();
    }
    assert_eq!(counter.load(Ordering::SeqCst), 5);
}

#[test]
fn pool_with_zero_embedders_errors() {
    let pool: EmbedderPool<CountingEmbedder> = EmbedderPool::new(Vec::new()).unwrap();
    let err = pool
        .embed(&[0.0_f32; 100])
        .expect_err("empty pool must fail");
    assert!(
        matches!(err, EmbedderError::ResourceExhausted { .. }),
        "empty pool is resource exhaustion, got {err}"
    );
    assert!(err.is_resource_exhausted());
}

#[test]
fn pool_rejects_mismatched_embedder_dims() {
    let counter = Arc::new(AtomicUsize::new(0));
    let embedders = vec![
        CountingEmbedder {
            counter: counter.clone(),
            dim: 192,
        },
        CountingEmbedder {
            counter: counter.clone(),
            dim: 256,
        },
    ];
    let err = match EmbedderPool::new(embedders) {
        Err(e) => e,
        Ok(_) => panic!("mismatched dims must fail"),
    };
    assert!(
        matches!(
            err,
            EmbedderError::DimMismatch {
                expected: 192,
                actual: 256
            }
        ),
        "expected DimMismatch(192, 256), got {err}"
    );
}

#[test]
fn resource_exhausted_classifier() {
    let typed = EmbedderError::ResourceExhausted {
        detail: "speaker sessions busy".into(),
    };
    assert!(typed.is_resource_exhausted());

    let legacy_string = EmbedderError::InferenceFailed {
        detail: "onnx session pool exhausted".into(),
    };
    assert!(legacy_string.is_resource_exhausted());

    let other = EmbedderError::DimMismatch {
        expected: 1,
        actual: 2,
    };
    assert!(!other.is_resource_exhausted());
}

#[test]
fn pool_propagates_inner_embedder_error() {
    struct FailingEmbedder;

    impl Embedder for FailingEmbedder {
        fn dim(&self) -> usize {
            8
        }
        fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
            Err(EmbedderError::InferenceFailed {
                detail: "boom".into(),
            })
        }
    }

    let pool = EmbedderPool::new(vec![FailingEmbedder]).unwrap();
    let err = pool
        .embed(&[0.0_f32; 16])
        .expect_err("inner error surfaces");
    assert!(
        matches!(err, EmbedderError::InferenceFailed { ref detail } if detail == "boom"),
        "expected the inner InferenceFailed error, got {err}"
    );
}

#[test]
fn pipeline_and_streaming_error_helpers() {
    use crate::pipeline::LegacyPipelineError;
    use crate::streaming::StreamingError;

    let emb = EmbedderError::ResourceExhausted {
        detail: "busy".into(),
    };
    let pe = LegacyPipelineError::Embedding(emb.clone());
    let se = StreamingError::Embedding(emb);
    assert!(pe.is_resource_exhausted());
    assert!(se.is_resource_exhausted());
    let non_embedding = LegacyPipelineError::AudioTooLong {
        actual_secs: 2.0,
        max_secs: 1.0,
    };
    assert!(!non_embedding.is_resource_exhausted());
}
