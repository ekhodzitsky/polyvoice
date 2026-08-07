use super::*;

#[test]
fn no_overlap_regions_pass_through() {
    let audio = vec![1.0_f32; 16_000];
    let masked = apply_overlap_mask(&audio, &[], 16_000);
    assert_eq!(masked, audio);
}

#[test]
fn single_overlap_region_is_zeroed() {
    let audio = vec![1.0_f32; 16_000];
    let masked = apply_overlap_mask(&audio, &[(0.5, 0.7)], 16_000);
    for (i, &v) in masked.iter().enumerate() {
        if (8000..11200).contains(&i) {
            assert_eq!(v, 0.0, "sample {i} should be zeroed");
        } else {
            assert_eq!(v, 1.0, "sample {i} should pass through");
        }
    }
}

#[test]
fn empty_input_returns_empty() {
    let masked = apply_overlap_mask(&[], &[(0.0, 1.0)], 16_000);
    assert!(masked.is_empty());
}

#[test]
fn out_of_bounds_overlap_is_clamped() {
    let audio = vec![1.0_f32; 100];
    let masked = apply_overlap_mask(&audio, &[(0.5, 1.0)], 16_000);
    assert_eq!(masked, audio, "out-of-bounds overlap is a no-op");
}

#[test]
fn negative_overlap_start_is_clamped_to_zero() {
    let audio = vec![1.0_f32; 16_000];
    let masked = apply_overlap_mask(&audio, &[(-1.0, 0.5)], 16_000);
    for &v in masked.iter().take(8000) {
        assert_eq!(v, 0.0);
    }
    for &v in masked.iter().skip(8000) {
        assert_eq!(v, 1.0);
    }
}

#[test]
fn multiple_overlap_regions_all_zeroed() {
    let audio = vec![1.0_f32; 16_000];
    let masked = apply_overlap_mask(&audio, &[(0.1, 0.2), (0.5, 0.6), (0.9, 1.0)], 16_000);
    let zero_ranges = [(1600..3200), (8000..9600), (14_400..16_000)];
    for (i, &v) in masked.iter().enumerate() {
        let in_zero = zero_ranges.iter().any(|r| r.contains(&i));
        if in_zero {
            assert_eq!(v, 0.0, "sample {i} should be zeroed");
        } else {
            assert_eq!(v, 1.0, "sample {i} should pass through");
        }
    }
}

#[test]
fn invalid_overlap_with_end_before_start_is_no_op() {
    let audio = vec![1.0_f32; 16_000];
    let masked = apply_overlap_mask(&audio, &[(0.7, 0.5)], 16_000);
    assert_eq!(masked, audio, "end<start is silently skipped");
}

#[test]
fn non_finite_overlap_bounds_are_skipped() {
    let audio = vec![1.0_f32; 16_000];
    let masked = apply_overlap_mask(
        &audio,
        &[(f32::NAN, 0.5), (0.1, f32::INFINITY), (0.2, f32::NAN)],
        16_000,
    );
    assert_eq!(masked, audio, "NaN/infinite bounds are silently skipped");
}

#[test]
fn zero_length_overlap_is_no_op() {
    let audio = vec![1.0_f32; 16_000];
    let masked = apply_overlap_mask(&audio, &[(0.5, 0.5)], 16_000);
    assert_eq!(masked, audio, "end==start is silently skipped");
}

#[test]
fn overlap_extending_past_end_is_clamped_to_len() {
    let audio = vec![1.0_f32; 100];
    let masked = apply_overlap_mask(&audio, &[(0.0, 10.0)], 16_000);
    assert!(
        masked.iter().all(|&v| v == 0.0),
        "region past the end zeroes through the final sample"
    );
    assert_eq!(masked.len(), audio.len());
}

#[test]
fn overlap_starting_past_end_is_no_op() {
    let audio = vec![1.0_f32; 100];
    let masked = apply_overlap_mask(&audio, &[(5.0, 6.0)], 16_000);
    assert_eq!(masked, audio, "region fully past the end is a no-op");
}
