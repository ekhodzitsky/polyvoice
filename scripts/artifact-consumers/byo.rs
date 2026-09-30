use polyvoice::{
    DiarizationConfig, Embedder, EmbedderError, EnergyVad, VadConfig,
    pipeline::{LegacyPipeline, LegacyPipelineError},
};

struct ConstantEmbedder;
impl Embedder for ConstantEmbedder {
    fn dim(&self) -> usize {
        4
    }
    fn embed(&self, _audio: &[f32]) -> Result<Vec<f32>, EmbedderError> {
        Ok(vec![1.0, 0.0, 0.0, 0.0])
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vad_config = VadConfig::default();
    let frame_size = vad_config.frame_size;
    let pipeline = LegacyPipeline::new(DiarizationConfig::default(), vad_config);
    for samples in [vec![], vec![0.0; 100], vec![0.0; 16000]] {
        let mut vad = EnergyVad::try_new(-40.0, 16000, frame_size)?;
        let result = pipeline.run(&samples, &ConstantEmbedder, &mut vad)?;
        assert_eq!(result.num_speakers, 0);
        assert!(result.turns.is_empty());
    }
    assert!(EnergyVad::try_new(-40.0, 16000, 0).is_err());
    let speech: Vec<_> = (0..16000 * 4)
        .map(|i| (i as f32 * 300.0 * std::f32::consts::TAU / 16000.0).sin() * 0.5)
        .collect();
    let mut vad = EnergyVad::try_new(-40.0, 16000, frame_size)?;
    let result = pipeline.run(&speech, &ConstantEmbedder, &mut vad)?;
    assert_eq!(result.num_speakers, 1);
    assert!(!result.turns.is_empty());
    let mixed: Vec<_> = speech
        .iter()
        .enumerate()
        .map(|(i, sample)| {
            (sample + (i as f32 * 700.0 * std::f32::consts::TAU / 16000.0).sin() * 0.5) / 2.0
        })
        .collect();
    let mut vad = EnergyVad::try_new(-40.0, 16000, frame_size)?;
    assert!(
        !pipeline
            .run(&mixed, &ConstantEmbedder, &mut vad)?
            .turns
            .is_empty()
    );
    let mut hour = vec![0.0; 16000 * 3600];
    hour[..speech.len()].copy_from_slice(&speech);
    let end = hour.len();
    hour[end - speech.len()..].copy_from_slice(&speech);
    let mut vad = EnergyVad::try_new(-40.0, 16000, frame_size)?;
    let result = pipeline.run(&hour, &ConstantEmbedder, &mut vad)?;
    assert!(result.turns.iter().any(|t| t.time.start < 4.0));
    assert!(result.turns.iter().any(|t| t.time.end > 3596.0));
    hour.push(0.0);
    assert!(matches!(
        pipeline.run(&hour, &ConstantEmbedder, &mut vad),
        Err(LegacyPipelineError::AudioTooLong { .. })
    ));
    println!(
        "{}",
        serde_json::json!([
            "empty",
            "short",
            "silence",
            "invalid-vad",
            "speech",
            "overlap",
            "hour",
            "too-long"
        ])
    );
    Ok(())
}
