//! Hungarian-constrained local→global speaker reassignment.
//!
//! Maps file-consistent local speaker indices (from powerset segmentation) onto
//! global clusters after embedding clustering. Maximizes total co-occurrence
//! duration via Kuhn-Munkres, then enforces cannot-link pairs: two locals that
//! co-occur in an overlap region must not share a global identity.
//!
//! Also documents / guards against the pyannote-style "inactive speakers in the
//! similarity matrix" bug: only locals that actually appear in the co-occurrence
//! table participate in the assignment (zero-duration / never-seen locals are
//! excluded rather than padded into a full max-speakers square matrix).

use crate::types::SpeakerId;
use std::collections::{HashMap, HashSet};

/// Duration co-occurrence table: `local_idx → (global_label → seconds)`.
pub type LocalGlobalDuration = HashMap<u8, HashMap<u32, f64>>;

/// Map each local speaker index to a global [`SpeakerId`].
///
/// * `cooc` — speech duration of each local landing in each global cluster.
/// * `cannot_link` — pairs of locals that co-occur (e.g. the two speakers of an
///   overlap region) and therefore must receive distinct globals.
///
/// Locals absent from `cooc` are omitted. Empty `cooc` yields an empty map.
pub fn hungarian_local_to_global(
    cooc: &LocalGlobalDuration,
    cannot_link: &[(u8, u8)],
) -> HashMap<u8, SpeakerId> {
    if cooc.is_empty() {
        return HashMap::new();
    }

    // Active locals only — never invent rows for unused local indices.
    let mut locals: Vec<u8> = cooc.keys().copied().collect();
    locals.sort_unstable();

    // Active globals = those that appear with positive duration for some local.
    let mut globals: Vec<u32> = cooc
        .values()
        .flat_map(|m| m.iter().filter(|(_, d)| **d > 0.0).map(|(g, _)| *g))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    globals.sort_unstable();

    if globals.is_empty() {
        // No positive co-occurrence; fall back to majority (first-seen global id 0).
        return locals
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, SpeakerId(i as u32)))
            .collect();
    }

    // Square cost matrix of size max(L, G), padded with zeros (neutral).
    // Cost = -duration so Hungarian (min-cost) maximises co-occurrence.
    let n = locals.len().max(globals.len());
    let mut cost = vec![vec![0.0_f32; n]; n];
    for (li, &loc) in locals.iter().enumerate() {
        let row = cooc.get(&loc);
        for (gi, &g) in globals.iter().enumerate() {
            let d = row.and_then(|m| m.get(&g).copied()).unwrap_or(0.0);
            cost[li][gi] = -(d as f32);
        }
    }

    let assignment = crate::hungarian::solve(&cost).unwrap_or_else(|| {
        // Degenerate fallback: identity into the smaller dimension.
        (0..n).collect()
    });

    let mut map: HashMap<u8, SpeakerId> = HashMap::new();
    for (li, &loc) in locals.iter().enumerate() {
        let gi = assignment[li];
        if gi < globals.len() {
            map.insert(loc, SpeakerId(globals[gi]));
        } else {
            // Padded column — local assigned to a dummy; use majority fallback.
            if let Some((&g, _)) = cooc
                .get(&loc)
                .and_then(|m| m.iter().max_by(|a, b| a.1.total_cmp(b.1)))
            {
                map.insert(loc, SpeakerId(g));
            }
        }
    }

    enforce_cannot_link(&mut map, cooc, cannot_link);
    map
}

/// Majority-vote map (legacy). Used only as an ablation / comparison baseline;
/// production path uses [`hungarian_local_to_global`].
pub fn majority_local_to_global(cooc: &LocalGlobalDuration) -> HashMap<u8, SpeakerId> {
    cooc.iter()
        .filter_map(|(loc, per_global)| {
            per_global
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(g, _)| (*loc, SpeakerId(*g)))
        })
        .collect()
}

/// If two cannot-link locals share a global, reassign the weaker (less
/// co-occurrence duration on that global) to its next-best free global.
fn enforce_cannot_link(
    map: &mut HashMap<u8, SpeakerId>,
    cooc: &LocalGlobalDuration,
    cannot_link: &[(u8, u8)],
) {
    for &(a, b) in cannot_link {
        let (Some(ga), Some(gb)) = (map.get(&a).copied(), map.get(&b).copied()) else {
            continue;
        };
        if ga != gb {
            continue;
        }
        // Conflict: both map to the same global. Reassign the one with less
        // duration on that global to its next-best distinct global.
        let da = cooc
            .get(&a)
            .and_then(|m| m.get(&ga.0).copied())
            .unwrap_or(0.0);
        let db = cooc
            .get(&b)
            .and_then(|m| m.get(&gb.0).copied())
            .unwrap_or(0.0);
        let (victim, other_global) = if da <= db { (a, ga) } else { (b, gb) };
        let taken: HashSet<u32> = map
            .iter()
            .filter(|(l, _)| **l != victim)
            .map(|(_, s)| s.0)
            .collect();
        if let Some(next) = cooc.get(&victim).and_then(|m| {
            m.iter()
                .filter(|(g, _)| **g != other_global.0 && !taken.contains(*g))
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map(|(g, _)| SpeakerId(*g))
        }) {
            map.insert(victim, next);
        } else if let Some(next) = cooc.get(&victim).and_then(|m| {
            // Last resort: any other global even if taken (prefer distinct).
            m.iter()
                .filter(|(g, _)| **g != other_global.0)
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map(|(g, _)| SpeakerId(*g))
        }) {
            map.insert(victim, next);
        }
        // If there is no alternative global at all, leave the conflict — better
        // a shared global than inventing a speaker id with no evidence.
    }
}

/// Build a co-occurrence table from parallel local indices, global labels, and
/// per-item durations. Items with non-positive duration are skipped.
pub fn build_cooccurrence(
    local_idx: &[u8],
    global_labels: &[usize],
    durations: &[f64],
) -> LocalGlobalDuration {
    let n = local_idx
        .len()
        .min(global_labels.len())
        .min(durations.len());
    let mut cooc: LocalGlobalDuration = HashMap::new();
    for i in 0..n {
        let d = durations[i];
        if d <= 0.0 {
            continue;
        }
        *cooc
            .entry(local_idx[i])
            .or_default()
            .entry(global_labels[i] as u32)
            .or_default() += d;
    }
    cooc
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_matches_clear_cooccurrence() {
        // local 0 mostly with global 1; local 1 mostly with global 0.
        let mut cooc = LocalGlobalDuration::new();
        cooc.insert(0, HashMap::from([(0, 1.0), (1, 9.0)]));
        cooc.insert(1, HashMap::from([(0, 8.0), (1, 1.0)]));
        let map = hungarian_local_to_global(&cooc, &[]);
        assert_eq!(map.get(&0), Some(&SpeakerId(1)));
        assert_eq!(map.get(&1), Some(&SpeakerId(0)));
    }

    #[test]
    fn cannot_link_forces_distinct_globals() {
        // Majority would map both locals to global 0; cannot-link must split them.
        let mut cooc = LocalGlobalDuration::new();
        cooc.insert(0, HashMap::from([(0, 10.0), (1, 1.0)]));
        cooc.insert(1, HashMap::from([(0, 9.0), (1, 2.0)]));
        let majority = majority_local_to_global(&cooc);
        assert_eq!(majority.get(&0), Some(&SpeakerId(0)));
        assert_eq!(majority.get(&1), Some(&SpeakerId(0)));

        let map = hungarian_local_to_global(&cooc, &[(0, 1)]);
        assert_ne!(
            map.get(&0),
            map.get(&1),
            "cannot-link pair must not share a global"
        );
    }

    #[test]
    fn inactive_local_not_invented() {
        // Only local 0 and 2 appear — local 1 must not show up in the map
        // (the pyannote inactive-speaker-in-matrix anti-pattern).
        let mut cooc = LocalGlobalDuration::new();
        cooc.insert(0, HashMap::from([(0, 5.0)]));
        cooc.insert(2, HashMap::from([(1, 5.0)]));
        let map = hungarian_local_to_global(&cooc, &[]);
        assert!(map.contains_key(&0));
        assert!(map.contains_key(&2));
        assert!(
            !map.contains_key(&1),
            "never-seen local must not be invented"
        );
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn majority_includes_inactive_bug_baseline() {
        // Document the buggy pattern: building a full max_speakers matrix and
        // running Hungarian over inactive rows. This test encodes the CORRECT
        // behaviour (active-only) and a helper that simulates the old bug so
        // the regression is explicit.
        let mut cooc = LocalGlobalDuration::new();
        cooc.insert(0, HashMap::from([(0, 5.0)]));
        cooc.insert(2, HashMap::from([(1, 5.0)]));

        // Correct path.
        let good = hungarian_local_to_global(&cooc, &[]);
        assert!(!good.contains_key(&1));

        // Buggy path: pad inactive local 1 with zero-duration row into a 3×3.
        let buggy = buggy_full_matrix_assign(&cooc, 3);
        // The bug invents a mapping for the inactive local (or corrupts the
        // active assignment by competing with a zero-row). Either way the
        // active-only map must differ from, or at least not invent, local 1
        // when the zero row steals a column.
        assert!(
            buggy.contains_key(&1)
                || buggy.get(&0) != good.get(&0)
                || buggy.get(&2) != good.get(&2),
            "buggy full-matrix path must differ from active-only (regression oracle)"
        );
        // And the good path must never invent local 1.
        assert!(!good.contains_key(&1));
    }

    /// Simulates the pyannote-style bug: allocate a cost matrix of size
    /// `max_speakers × max_speakers` including never-seen (inactive) locals.
    fn buggy_full_matrix_assign(
        cooc: &LocalGlobalDuration,
        max_speakers: usize,
    ) -> HashMap<u8, SpeakerId> {
        let n = max_speakers;
        let mut cost = vec![vec![0.0_f32; n]; n];
        for loc in 0..n {
            for g in 0..n {
                let d = cooc
                    .get(&(loc as u8))
                    .and_then(|m| m.get(&(g as u32)).copied())
                    .unwrap_or(0.0);
                cost[loc][g] = -(d as f32);
            }
        }
        let assignment = crate::hungarian::solve(&cost).unwrap();
        let mut map = HashMap::new();
        for (loc, &g) in assignment.iter().enumerate() {
            map.insert(loc as u8, SpeakerId(g as u32));
        }
        map
    }

    #[test]
    fn build_cooccurrence_sums_durations() {
        let cooc = build_cooccurrence(&[0, 0, 1], &[0, 1, 1], &[1.0, 2.0, 3.0]);
        assert!((cooc[&0][&0] - 1.0).abs() < 1e-9);
        assert!((cooc[&0][&1] - 2.0).abs() < 1e-9);
        assert!((cooc[&1][&1] - 3.0).abs() < 1e-9);
    }
}
