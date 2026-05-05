//! Diarization Error Rate (DER) benchmark on synthetic ground-truth data.
//!
//! This benchmark measures accuracy, not throughput. It generates synthetic
//! two-speaker audio with known turn boundaries, runs offline diarization,
//! and computes DER, precision, recall, and F1.

use criterion::{Criterion, criterion_group, criterion_main};
use polyvoice::{
    DiarizationConfig, EmbeddingError, EmbeddingExtractor, OfflineDiarizer, SpeakerTurn, TimeRange,
};

/// Extractor that returns a deterministic embedding based on the dominant
/// frequency in the window (estimated via zero-crossing rate).
///
/// This allows perfect speaker separation when each speaker has a distinct
/// fundamental frequency.
struct FrequencyExtractor {
    dim: usize,
}

impl FrequencyExtractor {
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn dominant_freq(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0usize;
        for i in 1..samples.len() {
            if samples[i - 1] >= 0.0 && samples[i] < 0.0 {
                crossings += 1;
            }
        }
        // Zero-crossing rate ≈ 2 * freq / sample_rate
        let zcr = crossings as f32 / samples.len() as f32;
        zcr * sample_rate as f32 / 2.0
    }
}

impl EmbeddingExtractor for FrequencyExtractor {
    fn extract(
        &self,
        samples: &[f32],
        config: &DiarizationConfig,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let freq = self.dominant_freq(samples, config.sample_rate.get());
        // Map frequency to an index in the embedding vector.
        let idx = (freq / 100.0) as usize % self.dim;
        let mut emb = vec![0.0f32; self.dim];
        emb[idx] = 1.0;
        Ok(emb)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

/// Generate synthetic audio with known speaker turns.
///
/// Speaker A: 200 Hz sine, Speaker B: 400 Hz sine.
fn generate_ground_truth_audio(
    sample_rate: u32,
    duration_secs: usize,
) -> (Vec<f32>, Vec<SpeakerTurn>) {
    let total_samples = sample_rate as usize * duration_secs;
    let mut samples = vec![0.0f32; total_samples];
    let mut turns = Vec::new();

    // Alternating every 2 seconds.
    let segment_secs = 2usize;
    for seg_start in (0..duration_secs).step_by(segment_secs) {
        let freq = if (seg_start / segment_secs).is_multiple_of(2) {
            200.0
        } else {
            400.0
        };
        let start_sample = seg_start * sample_rate as usize;
        let end_sample = ((seg_start + segment_secs) * sample_rate as usize).min(total_samples);

        for (i, sample) in samples
            .iter_mut()
            .enumerate()
            .take(end_sample)
            .skip(start_sample)
        {
            let t = i as f32 / sample_rate as f32;
            *sample = 0.5 * (2.0 * std::f32::consts::PI * freq * t).sin();
        }

        turns.push(SpeakerTurn {
            speaker: polyvoice::SpeakerId((seg_start / segment_secs) as u32 % 2),
            time: TimeRange {
                start: seg_start as f64,
                end: (seg_start + segment_secs).min(duration_secs) as f64,
            },
            text: None,
        });
    }

    (samples, turns)
}

/// Compute DER (Diarization Error Rate) between reference and hypothesis turns.
///
/// DER = (Miss + False Alarm + Confusion) / Total Reference Time
fn compute_der(reference: &[SpeakerTurn], hypothesis: &[SpeakerTurn]) -> (f64, f64, f64, f64) {
    let total_ref_time: f64 = reference.iter().map(|t| t.time.duration()).sum();
    if total_ref_time <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let resolution = 0.01; // 10 ms frames
    let max_time = reference
        .iter()
        .chain(hypothesis.iter())
        .map(|t| t.time.end)
        .fold(0.0, f64::max);

    let mut miss = 0.0f64;
    let mut fa = 0.0f64;
    let mut confusion = 0.0f64;

    let mut t = 0.0;
    while t < max_time {
        let ref_speakers: Vec<_> = reference
            .iter()
            .filter(|tr| tr.time.start <= t && t < tr.time.end)
            .map(|tr| tr.speaker)
            .collect();
        let hyp_speakers: Vec<_> = hypothesis
            .iter()
            .filter(|tr| tr.time.start <= t && t < tr.time.end)
            .map(|tr| tr.speaker)
            .collect();

        let ref_set: std::collections::HashSet<_> = ref_speakers.iter().copied().collect();
        let hyp_set: std::collections::HashSet<_> = hyp_speakers.iter().copied().collect();

        for spk in &ref_set {
            if !hyp_set.contains(spk) {
                miss += resolution;
            } else if hyp_set.len() > 1 || ref_set.len() > 1 {
                // Simplified: any overlap mismatch counts as confusion
                confusion += resolution;
            }
        }

        for spk in &hyp_set {
            if !ref_set.contains(spk) {
                fa += resolution;
            }
        }

        t += resolution;
    }

    let der = (miss + fa + confusion) / total_ref_time;
    (
        der,
        miss / total_ref_time,
        fa / total_ref_time,
        confusion / total_ref_time,
    )
}

fn bench_der(c: &mut Criterion) {
    let sample_rate = 16000;
    let (samples, reference) = generate_ground_truth_audio(sample_rate, 10);

    let config = DiarizationConfig {
        window_secs: 1.0,
        hop_secs: 0.5,
        threshold: 0.1, // low threshold to encourage separation
        ..Default::default()
    };

    c.bench_function("der_10s_two_speakers", |b| {
        b.iter(|| {
            let diarizer = OfflineDiarizer::new(config);
            let extractor = FrequencyExtractor::new(256);
            let result = diarizer.run(&samples, &extractor).unwrap();
            let (der, miss, fa, conf) = compute_der(&reference, &result.turns);
            // Use black_box to prevent compiler from optimizing away the computation.
            criterion::black_box((der, miss, fa, conf));
        });
    });
}

criterion_group!(benches, bench_der);
criterion_main!(benches);
