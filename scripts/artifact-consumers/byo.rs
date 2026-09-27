use polyvoice::{EnergyVad, SampleRate, VoiceActivityDetector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rate = SampleRate::new(16000).ok_or("invalid rate")?;
    let mut vad = EnergyVad::try_new(-40.0, rate.get(), 400)?;
    let probabilities = vad.process(&vec![0.0; 16000])?;
    assert!(!probabilities.is_empty());
    assert!(probabilities.iter().all(|p| *p < 0.5));
    Ok(())
}
