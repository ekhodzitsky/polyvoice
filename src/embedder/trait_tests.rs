use super::*;

/// In-memory dummy used by trait tests.
struct ConstantEmbedder {
    values: Vec<f32>,
}

impl Embedder for ConstantEmbedder {
    fn dim(&self) -> usize {
        self.values.len()
    }
    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        Ok(self.values.clone())
    }
}

#[test]
fn embedder_trait_object_is_dyn_compatible() {
    let e = ConstantEmbedder {
        values: vec![0.1, 0.2, 0.3],
    };
    let _b: Box<dyn Embedder> = Box::new(e);
}

#[test]
fn embedder_default_batch_is_serial() {
    let e = ConstantEmbedder {
        values: vec![0.5; 4],
    };
    let inputs: Vec<&[f32]> = vec![&[][..], &[][..], &[][..]];
    let out = e.embed_batch(&inputs).unwrap();
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|v| v.len() == 4 && v[0] == 0.5));
}

#[test]
fn embedder_dim_matches_output() {
    let e = ConstantEmbedder {
        values: vec![1.0; 192],
    };
    assert_eq!(e.dim(), 192);
    assert_eq!(e.embed(&[]).unwrap().len(), 192);
}

#[test]
fn embedder_error_audio_too_short_displays() {
    let err = EmbedderError::AudioTooShort {
        actual_secs: 0.05,
        min_secs: 0.25,
    };
    let msg = format!("{err}");
    assert!(msg.contains("0.05"));
    assert!(msg.contains("0.25"));
}
