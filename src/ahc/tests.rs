use super::*;

#[test]
fn test_agglomerative_cluster_basic() {
    // Two clear clusters.
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.1, 0.9, 0.0],
    ];
    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert_eq!(labels.len(), 4);
    assert_eq!(labels.iter().copied().max(), Some(1));
    // First two should be same cluster, last two should be same cluster.
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
    assert_ne!(labels[0], labels[2]);
}

#[test]
fn test_agglomerative_cluster_empty() {
    let labels = agglomerative_cluster(&[], 0.5);
    assert!(labels.is_empty());
}

/// Scored AHC: when a singleton absorbs a strictly heavier cluster, the
/// merged cluster's dominant member must come from the heavier
/// (merged-into) side — its cohort stats best approximate the weighted
/// centroid.
#[cfg(feature = "clusterer")]
#[test]
fn dominant_member_comes_from_the_heavier_merged_into_side() {
    use std::cell::RefCell;

    /// Returns scripted scores per dominant-member pair and records the
    /// member indices of every call (centroid values are ignored).
    struct TableScorer {
        scores: HashMap<(usize, usize), f32>,
        calls: RefCell<Vec<(usize, usize)>>,
    }
    impl AhcScorer for TableScorer {
        fn score(&self, _a: &[f32], ma: usize, _b: &[f32], mb: usize) -> f32 {
            self.calls.borrow_mut().push((ma, mb));
            self.scores[&(ma, mb)]
        }
    }

    // Merge script: (1,2) at 0.9 merges first (equal sizes — the merge
    // target keeps its own dominant member, 1). The refresh then scores
    // the {1,2} cluster against singleton 0 at 0.8, so 0 absorbs {1,2}:
    // the merged-into side is strictly heavier, and the merged cluster
    // must inherit dominant member 1, not 0.
    let scores: HashMap<(usize, usize), f32> = [
        ((0, 1), 0.5),
        ((0, 2), 0.5),
        ((0, 3), 0.0),
        ((1, 2), 0.9),
        ((1, 3), 0.1),
        ((2, 3), 0.1),
        ((1, 0), 0.8), // refreshed score of {1,2} against 0
    ]
    .into_iter()
    .collect();
    let scorer = TableScorer {
        scores,
        calls: RefCell::new(Vec::new()),
    };
    let embeddings = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.1, 0.9],
        vec![0.8, 0.2],
    ];
    let labels = agglomerative_cluster_scored(&embeddings, 0.05, 0, &scorer);
    assert!(
        labels.iter().all(|&l| l == labels[0]),
        "all scripted scores clear the threshold: {labels:?}"
    );
    // The last call is the refresh of the merged {0,1,2} cluster against
    // singleton 3: member 1 (inherited from the heavier side), not 0.
    // Without the heavier-side inheritance this would read (0, 3).
    assert_eq!(scorer.calls.borrow().last(), Some(&(1, 3)));
}

// --- duration-based pruning ---

fn ax(a: usize) -> Vec<f32> {
    let mut v = vec![0.02f32, 0.02, 0.02];
    v[a] = 1.0;
    v
}
fn tr(s: f64, e: f64) -> TimeRange {
    TimeRange { start: s, end: e }
}
fn ndistinct(labels: &[usize]) -> usize {
    let set: std::collections::HashSet<usize> = labels.iter().copied().collect();
    set.len()
}

#[test]
fn prune_by_duration_dissolves_brief_cluster() {
    // cluster 0: ~3 s on axis 0; cluster 1: one 0.5 s window on axis 1.
    let embeddings = vec![ax(0), ax(0), ax(0), ax(0), ax(1)];
    let times = vec![
        tr(0.0, 1.0),
        tr(0.75, 1.75),
        tr(1.5, 2.5),
        tr(2.25, 3.25),
        tr(5.0, 5.5),
    ];
    let out = prune_small_clusters_by_duration(&times, &embeddings, vec![0, 0, 0, 0, 1], 1.5);
    assert_eq!(out.len(), 5);
    assert_eq!(ndistinct(&out), 1, "the brief 0.5 s cluster is dissolved");
}

#[test]
fn prune_by_duration_keeps_few_but_long_speaker() {
    // cluster 1 has only TWO windows but 4 s of speech — it survives duration
    // pruning, whereas the member-count rule (min 4) wrongly dissolves it.
    let embeddings = vec![ax(0), ax(0), ax(0), ax(0), ax(0), ax(0), ax(1), ax(1)];
    let times = vec![
        tr(0.0, 1.0),
        tr(0.75, 1.75),
        tr(1.5, 2.5),
        tr(2.25, 3.25),
        tr(3.0, 4.0),
        tr(3.75, 4.75),
        tr(10.0, 12.0),
        tr(12.0, 14.0),
    ];
    let labels = vec![0, 0, 0, 0, 0, 0, 1, 1];
    let dur = prune_small_clusters_by_duration(&times, &embeddings, labels.clone(), 1.5);
    assert_eq!(
        ndistinct(&dur),
        2,
        "few-but-long cluster survives duration prune"
    );
    assert_ne!(dur[0], dur[6]);
    // Contrast: the count rule over-prunes the same 2-member cluster.
    let cnt = prune_small_clusters(&embeddings, labels, 4);
    assert_eq!(
        ndistinct(&cnt),
        1,
        "count rule over-prunes the long-but-few cluster"
    );
}

#[test]
fn prune_by_duration_zero_is_passthrough() {
    let embeddings = vec![ax(0), ax(1)];
    let times = vec![tr(0.0, 0.1), tr(1.0, 1.1)];
    let labels = vec![0, 1];
    assert_eq!(
        prune_small_clusters_by_duration(&times, &embeddings, labels.clone(), 0.0),
        labels
    );
}

#[test]
fn prune_by_duration_all_short_keeps_one() {
    let embeddings = vec![ax(0), ax(1), ax(2)];
    let times = vec![tr(0.0, 0.2), tr(1.0, 1.2), tr(2.0, 2.2)];
    let out = prune_small_clusters_by_duration(&times, &embeddings, vec![0, 1, 2], 5.0);
    assert_eq!(out.len(), 3);
    assert_eq!(ndistinct(&out), 1, "all-short collapses to one survivor");
}

#[test]
fn test_agglomerative_cluster_single() {
    let embeddings = vec![vec![1.0, 0.0, 0.0]];
    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert_eq!(labels, vec![0]);
}

#[test]
fn test_agglomerative_cluster_auto_max_clusters_caps_count() {
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.1, 0.9, 0.0],
    ];
    let (labels, _th) = agglomerative_cluster_auto_max_clusters(&embeddings, 2);
    let unique: std::collections::HashSet<usize> = labels.iter().copied().collect();
    assert_eq!(
        unique.len(),
        2,
        "max_clusters=2 must produce exactly 2 clusters"
    );
}

#[test]
fn test_agglomerative_cluster_mismatched_dimensions() {
    let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1]];
    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert_eq!(labels, vec![0, 0]);
}

#[test]
fn cluster_ids_are_canonical_and_shuffle_invariant() {
    // A 3-member cluster near [1,0,0] and a 2-member cluster near [0,1,0].
    let a = vec![1.0, 0.0, 0.0];
    let a2 = vec![0.95, 0.05, 0.0];
    let a3 = vec![0.9, 0.1, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let b2 = vec![0.05, 0.95, 0.0];

    // Canonical ordering: the larger cluster (3 members) must get id 0,
    // regardless of input order — descending size, tie-break min index.
    let base = vec![a.clone(), a2.clone(), a3.clone(), b.clone(), b2.clone()];
    let l1 = agglomerative_cluster(&base, 0.5);
    assert_eq!(l1, vec![0, 0, 0, 1, 1], "big cluster must be id 0");

    // Shuffled copy of the SAME points: the 3-member cluster (a-points at
    // shuffled indices 1,3,4) must still be id 0. The old first-appearance
    // relabel would have made the first-seen cluster id 0 instead.
    let shuffled = vec![b2.clone(), a3.clone(), b.clone(), a.clone(), a2.clone()];
    let l2 = agglomerative_cluster(&shuffled, 0.5);
    assert_eq!(l2[1], 0, "a3 is in the big cluster -> id 0");
    assert_eq!(l2[3], 0, "a is in the big cluster -> id 0");
    assert_eq!(l2[4], 0, "a2 is in the big cluster -> id 0");
    assert_eq!(l2[0], 1, "b2 is in the small cluster -> id 1");
    assert_eq!(l2[2], 1, "b is in the small cluster -> id 1");
}

#[test]
fn cahc_asc_refuses_to_merge_two_established_clusters() {
    // Two well-separated speakers (3 members each). With a very low
    // threshold classic AHC would still merge them; MinMembers(3) stops
    // before that final merge because both sides are already established.
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.98, 0.05, 0.0],
        vec![0.97, 0.0, 0.05],
        vec![0.0, 1.0, 0.0],
        vec![0.05, 0.98, 0.0],
        vec![0.0, 0.97, 0.05],
    ];
    let classic = agglomerative_cluster(&embeddings, 0.0);
    assert_eq!(
        ndistinct(&classic),
        1,
        "threshold 0 must glue everything without ASC"
    );
    let asc = agglomerative_cluster_asc(&embeddings, 0.0, 0, AscStop::MinMembers(3), None);
    assert_eq!(
        ndistinct(&asc),
        2,
        "cAHC-ASC must keep two established speakers separate"
    );
    assert_eq!(asc[0], asc[1]);
    assert_eq!(asc[0], asc[2]);
    assert_eq!(asc[3], asc[4]);
    assert_eq!(asc[3], asc[5]);
    assert_ne!(asc[0], asc[3]);
}

#[test]
fn cahc_asc_min_secs_uses_durations() {
    // Same geometry as above, but establish by total duration (each speaker
    // has ~3 s of speech across three 1 s windows).
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.98, 0.05, 0.0],
        vec![0.97, 0.0, 0.05],
        vec![0.0, 1.0, 0.0],
        vec![0.05, 0.98, 0.0],
        vec![0.0, 0.97, 0.05],
    ];
    let times = vec![
        tr(0.0, 1.0),
        tr(1.0, 2.0),
        tr(2.0, 3.0),
        tr(10.0, 11.0),
        tr(11.0, 12.0),
        tr(12.0, 13.0),
    ];
    let asc = agglomerative_cluster_asc(&embeddings, 0.0, 0, AscStop::MinSecs(2.5), Some(&times));
    assert_eq!(ndistinct(&asc), 2);
    assert_ne!(asc[0], asc[3]);
}

#[test]
fn cahc_asc_off_matches_classic() {
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.1, 0.9, 0.0],
    ];
    let classic = agglomerative_cluster_max_clusters(&embeddings, 0.5, 0);
    let asc = agglomerative_cluster_asc(&embeddings, 0.5, 0, AscStop::Off, None);
    assert_eq!(classic, asc);
}

#[test]
fn cahc_asc_min_secs_ignored_without_matching_time_ranges() {
    // MinSecs with no time ranges (or a length mismatch) degrades to Off:
    // threshold 0 then glues everything, exactly like classic AHC.
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.98, 0.05, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.05, 0.98, 0.0],
    ];
    let classic = agglomerative_cluster(&embeddings, 0.0);
    assert_eq!(ndistinct(&classic), 1);

    let no_times = agglomerative_cluster_asc(&embeddings, 0.0, 0, AscStop::MinSecs(2.0), None);
    assert_eq!(no_times, classic, "missing time ranges → stop ignored");

    let short = vec![tr(0.0, 1.0), tr(1.0, 2.0)];
    let mismatched =
        agglomerative_cluster_asc(&embeddings, 0.0, 0, AscStop::MinSecs(2.0), Some(&short));
    assert_eq!(mismatched, classic, "length mismatch → stop ignored");
}

#[test]
fn cahc_asc_degenerate_stops_are_off() {
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.1, 0.9, 0.0],
    ];
    let classic = agglomerative_cluster(&embeddings, 0.0);
    assert_eq!(ndistinct(&classic), 1);
    for stop in [
        AscStop::MinMembers(0),
        AscStop::MinSecs(0.0),
        AscStop::MinSecs(-1.5),
    ] {
        let asc = agglomerative_cluster_asc(&embeddings, 0.0, 0, stop, None);
        assert_eq!(asc, classic, "{stop:?} must behave as Off");
    }
}

#[test]
fn cahc_asc_ceiling_yields_to_established_clusters() {
    // Two tight pairs, each established after one merge. max_clusters = 1
    // wants more merging, but every remaining pair is two established
    // clusters — the ASC stop wins over the ceiling.
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.98, 0.05, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.05, 0.98, 0.0],
    ];
    let asc = agglomerative_cluster_asc(&embeddings, 0.0, 1, AscStop::MinMembers(2), None);
    assert_eq!(
        ndistinct(&asc),
        2,
        "ceiling must not glue two established clusters"
    );
    assert_eq!(asc[0], asc[1]);
    assert_eq!(asc[2], asc[3]);
    assert_ne!(asc[0], asc[2]);
}

#[test]
fn cahc_asc_stop_holds_even_at_unbounded_threshold() {
    // threshold = -inf merges anything allowed; the only blocked merge is
    // the established-vs-established pair, which must still be refused.
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.98, 0.05, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.05, 0.98, 0.0],
    ];
    let asc = agglomerative_cluster_asc(
        &embeddings,
        f32::NEG_INFINITY,
        0,
        AscStop::MinMembers(2),
        None,
    );
    assert_eq!(ndistinct(&asc), 2);
    // Sanity: without the stop, -inf glues everything into one cluster.
    let classic = agglomerative_cluster(&embeddings, f32::NEG_INFINITY);
    assert_eq!(ndistinct(&classic), 1);
}

#[test]
fn cluster_durations_sums_disjoint_spans_separately() {
    // Disjoint spans in one cluster must not be merged into a single span:
    // total = (1.0) + (1.0) = 2.0, not the 4.0 outer envelope.
    let times = vec![tr(0.0, 1.0), tr(3.0, 4.0), tr(0.5, 2.0)];
    let labels = vec![0, 0, 1];
    let durs = cluster_durations(&times, &labels);
    assert!((durs[&0] - 2.0).abs() < 1e-12);
    assert!((durs[&1] - 1.5).abs() < 1e-12);
}

#[test]
fn auto_max_clusters_empty_input() {
    let (labels, th) = agglomerative_cluster_auto_max_clusters(&[], 3);
    assert!(labels.is_empty());
    assert_eq!(th, 0.0);
}

#[test]
fn auto_max_clusters_mismatched_dimensions_fall_back_to_one_cluster() {
    let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1]];
    let (labels, th) = agglomerative_cluster_auto_max_clusters(&embeddings, 2);
    assert_eq!(labels, vec![0, 0]);
    assert_eq!(th, 0.0);
}

// --- custom-scorer (AS-norm seam) tests ---

/// Scorer driven by a fixed member-keyed score table, recording every
/// `(member_a, member_b)` pair it is called with, so tests can observe
/// which dominant members the merge loop scores after each merge.
#[cfg(feature = "clusterer")]
struct TableScorer {
    table: HashMap<(usize, usize), f32>,
    calls: std::cell::RefCell<Vec<(usize, usize)>>,
}

#[cfg(feature = "clusterer")]
impl TableScorer {
    fn from_pairs(pairs: &[((usize, usize), f32)]) -> Self {
        Self {
            table: pairs.iter().copied().collect(),
            calls: std::cell::RefCell::new(Vec::new()),
        }
    }
}

#[cfg(feature = "clusterer")]
impl AhcScorer for TableScorer {
    fn score(&self, _a: &[f32], ma: usize, _b: &[f32], mb: usize) -> f32 {
        self.calls.borrow_mut().push((ma, mb));
        self.table
            .get(&(ma, mb))
            .or_else(|| self.table.get(&(mb, ma)))
            .copied()
            .unwrap_or(f32::NEG_INFINITY)
    }
}

#[cfg(feature = "clusterer")]
#[test]
fn scored_ahc_uses_scorer_for_matrix_and_post_merge_refresh() {
    // Four embeddings; geometry irrelevant — the table drives every score.
    let embeddings = vec![
        vec![1.0, 0.0],
        vec![0.9, 0.1],
        vec![0.0, 1.0],
        vec![0.1, 0.9],
    ];
    // s(0,1) = 0.99 merges first; then {0,1} (dominant 0, heavier side)
    // merges with 2 at 0.9; 3 stays out at 0.1. Threshold 0.85.
    let scorer = TableScorer::from_pairs(&[
        ((0, 1), 0.99),
        ((0, 2), 0.9),
        ((1, 2), 0.9),
        ((0, 3), 0.1),
        ((1, 3), 0.1),
        ((2, 3), 0.1),
    ]);
    let labels = agglomerative_cluster_scored(&embeddings, 0.85, 0, &scorer);
    assert_eq!(labels, vec![0, 0, 0, 1], "table-driven merges");
    let calls = scorer.calls.borrow();
    assert_eq!(
        &calls[..6],
        &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        "initial matrix scored once per pair, upper triangle"
    );
    // After merging 1 into 0 (tie 1v1 keeps best_i's dominant) the refresh
    // re-scores centroid 0 via its dominant member 0 — not raw cosine.
    assert_eq!(calls[6], (0, 2), "post-merge refresh uses dominant member");
    assert_eq!(calls[7], (0, 3));
    // After the second merge ({0,1} heavier than {2}), the dominant member
    // stays 0 — the heavier side's member, not the freshly merged 2.
    assert_eq!(
        calls[8],
        (0, 3),
        "dominant member of an unequal merge comes from the heavier side"
    );
    assert_eq!(calls.len(), 9, "no rescoring beyond the merge refreshes");
}

#[cfg(feature = "clusterer")]
#[test]
fn scored_ahc_constant_scorer_controls_merging() {
    let embeddings = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];
    struct Const(f32);
    impl AhcScorer for Const {
        fn score(&self, _a: &[f32], _ma: usize, _b: &[f32], _mb: usize) -> f32 {
            self.0
        }
    }
    // All pairs at 1.0 ≥ 0.5: everything merges despite the geometry.
    let merged = agglomerative_cluster_scored(&embeddings, 0.5, 0, &Const(1.0));
    assert_eq!(merged, vec![0, 0, 0]);
    // All pairs at 0.0 < 0.5: nothing merges.
    let split = agglomerative_cluster_scored(&embeddings, 0.5, 0, &Const(0.0));
    assert_eq!(split, vec![0, 1, 2]);
}

#[cfg(feature = "clusterer")]
#[test]
fn scored_ahc_degenerate_inputs() {
    let empty: Vec<Vec<f32>> = Vec::new();
    assert!(agglomerative_cluster_scored(&empty, 0.5, 0, &CosineScorer).is_empty());
    // Mixed dimensions fall back to a single cluster, like the cosine path.
    let mixed = vec![vec![1.0, 0.0], vec![0.9]];
    assert_eq!(
        agglomerative_cluster_scored(&mixed, 0.5, 0, &CosineScorer),
        vec![0, 0]
    );
}

#[cfg(feature = "clusterer")]
#[test]
fn scored_ahc_with_cosine_scorer_matches_classic() {
    // The cosine scorer through the scored path must reproduce plain AHC.
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.1, 0.9, 0.0],
    ];
    let classic = agglomerative_cluster_max_clusters(&embeddings, 0.5, 0);
    let scored = agglomerative_cluster_scored(&embeddings, 0.5, 0, &CosineScorer);
    assert_eq!(classic, scored);
}
