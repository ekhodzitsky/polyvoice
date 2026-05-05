//! Streaming diarization example.
//!
//! Simulates feeding audio in 1-second chunks.
//!
//! Run with:
//!     cargo run --example online

use polyvoice::{DiarizationConfig, DummyExtractor, OnlineDiarizer};

fn main() {
    let sample_rate = 16000;
    let total_seconds = 10;
    let samples = vec![0.0f32; sample_rate * total_seconds];

    let config = DiarizationConfig::default();
    let mut diarizer = OnlineDiarizer::new(config);
    let extractor = DummyExtractor::new(256);

    // Feed in 1-second chunks.
    let chunk_size = sample_rate;
    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        let segments = diarizer.feed(chunk, &extractor).expect("feed failed");
        if !segments.is_empty() {
            println!("Chunk {}: {} new segment(s)", i, segments.len());
        }
    }

    // Flush remaining audio.
    if let Some(final_seg) = diarizer.flush(&extractor).expect("flush failed") {
        println!(
            "Final segment: {:.2}s - {:.2}s",
            final_seg.time.start, final_seg.time.end
        );
    }

    println!("Total speakers detected: {}", diarizer.num_speakers());
}
