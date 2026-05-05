//! Real-audio benchmark suite for polyvoice.
//!
//! Measures end-to-end latency of offline diarization and ECAPA-TDNN
//! preprocessing on synthetic multi-speaker audio (two speakers,
//! alternating segments).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polyvoice::{DiarizationConfig, DummyExtractor, OfflineDiarizer};
use polyvoice::features::{FbankConfig, FbankExtractor};

/// Generate a 10-second synthetic waveform with two alternating speakers.
///
/// Speaker A (0–3 s, 6–8 s) is a 200 Hz sine wave.
/// Speaker B (3–6 s, 8–10 s) is a 400 Hz sine wave.
fn generate_two_speaker_audio(duration_secs: usize) -> Vec<f32> {
    let sample_rate = 16000;
    let total_samples = sample_rate * duration_secs;
    let mut samples = vec![0.0f32; total_samples];

    for (i, sample) in samples.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        let freq = if (0.0..3.0).contains(&t) || (6.0..8.0).contains(&t) {
            200.0f32
        } else {
            400.0f32
        };
        *sample = 0.5 * (2.0 * std::f32::consts::PI * freq * t).sin();
    }
    samples
}

fn bench_offline_diarization(c: &mut Criterion) {
    let config = DiarizationConfig::default();
    let diarizer = OfflineDiarizer::new(config);
    let extractor = DummyExtractor::new(256);
    let samples = generate_two_speaker_audio(10);

    c.bench_function("offline_diarization_10s", |b| {
        b.iter(|| {
            let result = diarizer.run(black_box(&samples), &extractor).unwrap();
            black_box(result);
        });
    });
}

fn bench_ecapa_fbank(c: &mut Criterion) {
    let samples = generate_two_speaker_audio(10);
    let config = FbankConfig::default();
    let extractor = FbankExtractor::new(config);

    c.bench_function("ecapa_fbank_10s", |b| {
        b.iter(|| {
            let fb = extractor.extract(black_box(&samples)).unwrap();
            black_box(fb);
        });
    });
}

criterion_group!(benches, bench_offline_diarization, bench_ecapa_fbank);
criterion_main!(benches);
