//! DER (Diarization Error Rate) evaluation.
//!
//! Usage:
//!   cargo bench --bench der_ami --features onnx
//!
//! Requires POLYVOICE_MODEL_DIR to be set for real model benchmarks.

use criterion::{Criterion, criterion_group, criterion_main};
use polyvoice::der::compute_der_from_rttm;
use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};

/// Convert bench hypothesis tuples to SpeakerTurns for the unified DER surface.
fn to_speaker_turns(hypothesis: &[(f64, f64, u32)]) -> Vec<SpeakerTurn> {
    hypothesis
        .iter()
        .map(|&(start, end, spk)| SpeakerTurn {
            speaker: SpeakerId(spk),
            time: TimeRange { start, end },
            text: None,
        })
        .collect()
}

fn bench_der_synthetic(c: &mut Criterion) {
    let reference: Vec<(f64, f64, &str)> = vec![(0.0, 3.0, "A"), (3.5, 6.0, "B"), (6.5, 10.0, "A")];

    let perfect_hyp: Vec<(f64, f64, u32)> = vec![(0.0, 3.0, 0), (3.5, 6.0, 1), (6.5, 10.0, 0)];

    let perfect_turns = to_speaker_turns(&perfect_hyp);
    let result = compute_der_from_rttm(&reference, &perfect_turns, 0.0);
    eprintln!("Synthetic DER (perfect): {:.1}%", result.der * 100.0);
    assert!(
        result.der < 0.05,
        "perfect hypothesis should have near-zero DER"
    );

    let imperfect_hyp: Vec<(f64, f64, u32)> = vec![
        (0.0, 3.0, 0),
        (3.5, 6.0, 0), // wrong speaker
        (6.5, 10.0, 0),
    ];

    let imperfect_turns = to_speaker_turns(&imperfect_hyp);
    let result2 = compute_der_from_rttm(&reference, &imperfect_turns, 0.0);
    eprintln!("Synthetic DER (confused): {:.1}%", result2.der * 100.0);
    assert!(
        result2.der > 0.1,
        "confused hypothesis should have significant DER"
    );

    c.bench_function("der_synthetic", |b| {
        b.iter(|| compute_der_from_rttm(&reference, &perfect_turns, 0.25));
    });
}

criterion_group!(benches, bench_der_synthetic);
criterion_main!(benches);
