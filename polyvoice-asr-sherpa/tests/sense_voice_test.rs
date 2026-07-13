//! Integration: real SenseVoice inference through the core Asr trait.
//! Needs the model bundle under data/sherpa-models/ (see README) — ignored
//! by default, mirroring the repo's model-gated test convention.

use polyvoice::Asr;
use polyvoice::types::SampleRate;
use polyvoice_asr_sherpa::SenseVoiceAsr;
use std::path::Path;

#[test]
#[ignore = "requires the SenseVoice bundle under ../data/sherpa-models/"]
fn sense_voice_transcribes_english_clip_via_trait_object() {
    let dir = Path::new("../data/sherpa-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17");
    let wav = Path::new("../tests/data/e2e-smoke/audio/fuzfh.wav");
    if !dir.is_dir() || !wav.is_file() {
        panic!(
            "model bundle or clip missing; see README (dir={}, wav={})",
            dir.display(),
            wav.display()
        );
    }

    let asr =
        SenseVoiceAsr::from_files(dir.join("model.int8.onnx"), dir.join("tokens.txt"), "auto")
            .expect("load SenseVoice");
    // Boxed as dyn Asr so object-safety is exercised for real.
    let asr: Box<dyn Asr> = Box::new(asr);

    let (samples, sr_hz) = polyvoice::wav::read_wav(wav).expect("read wav");
    let sr = SampleRate::new(sr_hz).expect("sample rate");
    let words = asr.transcribe(&samples, sr).expect("transcribe");

    assert!(
        words.len() >= 10,
        "26s English clip must yield many words, got {}",
        words.len()
    );
    assert!(
        words.windows(2).all(|w| w[0].time.start <= w[1].time.start),
        "word starts must be monotonic"
    );
    assert!(
        words.iter().all(|w| w.time.end > w.time.start),
        "every word must have positive duration"
    );
    let clip_secs = samples.len() as f64 / sr_hz as f64;
    assert!(
        words.iter().all(|w| w.time.end <= clip_secs + 0.5),
        "timestamps must stay within the clip"
    );
}

#[test]
fn missing_model_files_error_cleanly_without_touching_sherpa() {
    // The path check runs before any sherpa call, so the error is ModelIo.
    let err = match SenseVoiceAsr::from_files(
        "/nonexistent/model.onnx",
        "/nonexistent/tokens.txt",
        "auto",
    ) {
        Ok(_) => panic!("missing files must error"),
        Err(e) => e,
    };
    assert!(matches!(err, polyvoice::asr::AsrError::ModelIo { .. }));
}
