use polyvoice::{ModelRegistry, Pipeline, PipelineConfig, PipelineError, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let scenario = &args[3];
    #[cfg(feature = "local")]
    let registry = ModelRegistry::with_local_dir(&args[1])?;
    #[cfg(not(feature = "local"))]
    let registry = ModelRegistry::with_cache_dir(&args[1])?;
    let mut config = PipelineConfig::default();
    config.vbx_plda_dir = Some((&args[1]).into());
    let pipeline = Pipeline::builder()
        .config(config)
        .with_models_from(registry)
        .build()?;
    let samples = if scenario == "too-long" {
        vec![0.0; polyvoice::pipeline_v2::MAX_AUDIO_SAMPLES + 1]
    } else {
        polyvoice::wav::load_audio(std::path::Path::new(&args[2]))?.0
    };
    let rate = SampleRate::new(if scenario == "invalid-rate" {
        8000
    } else {
        16000
    })
    .ok_or("invalid rate")?;
    match pipeline.run(&samples, rate) {
        Err(PipelineError::Segmentation(
            polyvoice::segmentation::SegmentationError::AudioTooShort { .. },
        )) if scenario == "empty" || scenario == "short" => {
            println!("{}", serde_json::json!({"error": scenario}));
        }
        Err(PipelineError::AudioTooLong { .. }) if scenario == "too-long" => {
            println!("{}", serde_json::json!({"error": scenario}));
        }
        Err(PipelineError::UnsupportedSampleRate { .. }) if scenario == "invalid-rate" => {
            println!("{}", serde_json::json!({"error": scenario}));
        }
        Err(error) => return Err(error.into()),
        Ok(result) => {
            if matches!(
                scenario.as_str(),
                "too-long" | "invalid-rate" | "empty" | "short"
            ) {
                return Err("invalid input accepted".into());
            }
            println!("{}", serde_json::to_string(&result)?);
        }
    }
    Ok(())
}
