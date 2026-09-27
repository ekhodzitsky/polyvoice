use polyvoice::{ModelRegistry, Pipeline, PipelineConfig, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
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
    let (samples, rate) = polyvoice::wav::load_audio(std::path::Path::new(&args[2]))?;
    let result = pipeline.run(&samples, SampleRate::new(rate).ok_or("invalid rate")?)?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
