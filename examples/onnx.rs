//! ONNX extractor example (requires `--features onnx`).
//!
//! Run with:
//!     cargo run --example onnx --features onnx -- /path/to/model.onnx

#[cfg(feature = "onnx")]
fn main() {
    use polyvoice::{DiarizationConfig, EcapaTdnnExtractor, OfflineDiarizer};
    use std::path::Path;

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: cargo run --example onnx --features onnx -- <model.onnx>");

    let sample_rate = 16000;
    let seconds = 10;
    let samples = vec![0.0f32; sample_rate * seconds];

    let config = DiarizationConfig::default();
    let diarizer = OfflineDiarizer::new(config);
    let extractor = EcapaTdnnExtractor::new(
        Path::new(&model_path),
        192, // embedding dimension
        4,   // pool size
    )
    .expect("failed to load ONNX model");

    let result = diarizer.run(&samples, &extractor).expect("diarization failed");
    println!("Detected {} speaker(s)", result.num_speakers);
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("This example requires the `onnx` feature.");
    eprintln!("Run with: cargo run --example onnx --features onnx -- <model.onnx>");
    std::process::exit(1);
}
