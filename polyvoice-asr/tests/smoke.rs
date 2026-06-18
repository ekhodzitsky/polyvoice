//! Gated smoke test — real Parakeet TDT inference.
//!
//! Skips unless `POLYVOICE_ASR_MODEL_DIR` points at a directory holding the TDT
//! ONNX files (`encoder-model.onnx` + `encoder-model.onnx.data`,
//! `decoder_joint-model.onnx`, `vocab.txt` — see the crate README). This mirrors
//! the other model-gated ONNX tests in the workspace: it always compiles, but
//! only exercises inference when a model is provided.

use polyvoice::Asr;
use polyvoice::types::SampleRate;
use polyvoice_asr::ParakeetAsr;

#[test]
fn smoke_transcribe_when_model_present() {
    let dir = match std::env::var("POLYVOICE_ASR_MODEL_DIR") {
        Ok(d) => d,
        Err(_) => {
            eprintln!("skip: set POLYVOICE_ASR_MODEL_DIR to run the real ASR smoke test");
            return;
        }
    };

    let asr = ParakeetAsr::from_dir(&dir).expect("load Parakeet TDT model");
    let sr = SampleRate::new(16_000).expect("valid sample rate");
    // One second of silence: exercises load + transcribe + the Vec<Word> mapping
    // without asserting on transcript content (silence may yield zero words).
    let audio = vec![0.0f32; 16_000];
    let words = asr
        .transcribe(&audio, sr)
        .expect("transcribe must not error");
    eprintln!("smoke: produced {} words", words.len());
    // Timestamps, when present, must be ordered and non-negative.
    for w in &words {
        assert!(
            w.time.start >= 0.0 && w.time.end >= w.time.start,
            "bad word span"
        );
    }
}
