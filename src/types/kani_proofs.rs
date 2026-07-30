//! Kani formal proofs for `types` invariants.
//!
//! Run with: `cargo kani`

use super::{Confidence, SampleRate, TimeRange};

#[kani::proof]
fn confidence_new_invariant() {
    let value: f32 = kani::any();
    kani::assume(value.is_finite());

    if let Some(c) = Confidence::new(value) {
        assert!(c.get() >= 0.0);
        assert!(c.get() <= 1.0);
    }
}

#[kani::proof]
fn sample_rate_new_invariant() {
    let value: u32 = kani::any();

    if let Some(sr) = SampleRate::new(value) {
        assert!(sr.get() >= 8000);
        assert!(sr.get() <= 192000);
    }
}

#[kani::proof]
fn time_range_duration_non_negative() {
    let start: f64 = kani::any();
    let end: f64 = kani::any();
    kani::assume(start.is_finite());
    kani::assume(end.is_finite());

    let tr = TimeRange { start, end };
    let dur: f64 = tr.duration();
    assert!(dur >= 0.0);
    assert!(tr.start <= tr.end || dur == 0.0);
}
