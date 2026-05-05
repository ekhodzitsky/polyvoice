//! Offline diarization example.
//!
//! Run with:
//!     cargo run --example offline

use polyvoice::{DiarizationConfig, DummyExtractor, OfflineDiarizer};

fn main() {
    let sample_rate = 16000;
    let seconds = 10;
    let samples = vec![0.0f32; sample_rate * seconds];

    let config = DiarizationConfig::default();
    let diarizer = OfflineDiarizer::new(config);
    let extractor = DummyExtractor::new(256);

    let result = diarizer.run(&samples, &extractor).expect("diarization failed");

    println!("Detected {} speaker(s):", result.num_speakers);
    for turn in &result.turns {
        println!(
            "  {}: {:.2}s - {:.2}s",
            turn.speaker, turn.time.start, turn.time.end
        );
    }
}
