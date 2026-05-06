//! DER (Diarization Error Rate) evaluation.
//!
//! Usage:
//!   cargo bench --bench der_ami --features onnx
//!
//! Requires POLYVOICE_MODEL_DIR to be set for real model benchmarks.

use criterion::{Criterion, criterion_group, criterion_main};

/// Compute DER between reference and hypothesis annotations.
///
/// DER = (false_alarm + missed_speech + speaker_confusion) / total_reference_speech
///
/// Uses a simplified frame-based approach at 100ms resolution.
fn compute_der(reference: &[(f64, f64, &str)], hypothesis: &[(f64, f64, u32)], collar: f64) -> f64 {
    if reference.is_empty() {
        return 0.0;
    }

    let max_time = reference
        .iter()
        .map(|r| r.1)
        .chain(hypothesis.iter().map(|h| h.1))
        .fold(0.0f64, f64::max);

    let resolution = 0.1;
    let n_frames = (max_time / resolution).ceil() as usize;

    let mut ref_frames: Vec<Option<&str>> = vec![None; n_frames];
    for &(start, end, speaker) in reference {
        let s = ((start + collar) / resolution).ceil() as usize;
        let e = ((end - collar) / resolution).floor() as usize;
        for frame in ref_frames.iter_mut().take(e.min(n_frames)).skip(s) {
            *frame = Some(speaker);
        }
    }

    let mut hyp_frames: Vec<Option<u32>> = vec![None; n_frames];
    for &(start, end, speaker) in hypothesis {
        let s = (start / resolution).ceil() as usize;
        let e = (end / resolution).floor() as usize;
        for frame in hyp_frames.iter_mut().take(e.min(n_frames)).skip(s) {
            *frame = Some(speaker);
        }
    }

    // Count co-occurrences for greedy speaker mapping.
    let mut overlap = std::collections::HashMap::new();
    for i in 0..n_frames {
        if let (Some(r), Some(h)) = (ref_frames[i], hyp_frames[i]) {
            *overlap.entry((r, h)).or_insert(0usize) += 1;
        }
    }

    // Greedy mapping (good enough for evaluation).
    let mut mapping: std::collections::HashMap<u32, &str> = std::collections::HashMap::new();
    let mut used_ref: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut pairs: Vec<((&str, u32), usize)> = overlap.into_iter().collect();
    pairs.sort_by_key(|a| std::cmp::Reverse(a.1));
    for ((r, h), _) in pairs {
        if !mapping.contains_key(&h) && !used_ref.contains(r) {
            mapping.insert(h, r);
            used_ref.insert(r);
        }
    }

    let mut total_ref = 0usize;
    let mut missed = 0usize;
    let mut false_alarm = 0usize;
    let mut confusion = 0usize;

    for i in 0..n_frames {
        match (ref_frames[i], hyp_frames[i]) {
            (Some(_), None) => {
                total_ref += 1;
                missed += 1;
            }
            (None, Some(_)) => {
                false_alarm += 1;
            }
            (Some(r), Some(h)) => {
                total_ref += 1;
                if mapping.get(&h) != Some(&r) {
                    confusion += 1;
                }
            }
            (None, None) => {}
        }
    }

    if total_ref == 0 {
        return 0.0;
    }

    (missed + false_alarm + confusion) as f64 / total_ref as f64
}

fn bench_der_synthetic(c: &mut Criterion) {
    let reference: Vec<(f64, f64, &str)> = vec![(0.0, 3.0, "A"), (3.5, 6.0, "B"), (6.5, 10.0, "A")];

    let perfect_hyp: Vec<(f64, f64, u32)> = vec![(0.0, 3.0, 0), (3.5, 6.0, 1), (6.5, 10.0, 0)];

    let der = compute_der(&reference, &perfect_hyp, 0.0);
    eprintln!("Synthetic DER (perfect): {:.1}%", der * 100.0);
    assert!(der < 0.05, "perfect hypothesis should have near-zero DER");

    let imperfect_hyp: Vec<(f64, f64, u32)> = vec![
        (0.0, 3.0, 0),
        (3.5, 6.0, 0), // wrong speaker
        (6.5, 10.0, 0),
    ];

    let der2 = compute_der(&reference, &imperfect_hyp, 0.0);
    eprintln!("Synthetic DER (confused): {:.1}%", der2 * 100.0);
    assert!(
        der2 > 0.1,
        "confused hypothesis should have significant DER"
    );

    c.bench_function("der_synthetic", |b| {
        b.iter(|| compute_der(&reference, &perfect_hyp, 0.25));
    });
}

criterion_group!(benches, bench_der_synthetic);
criterion_main!(benches);
