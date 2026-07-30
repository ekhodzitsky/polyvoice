//! Kuhn-Munkres minimum-cost assignment for square cost matrices.
//!
//! Pure Rust, wasm32-clean. Consumers: `der` (optimal speaker mapping for
//! frame DER and WDER, both via [`map_max_cooccurrence`]), the `segmentation`
//! aggregator (aligning local speaker indices between overlapping windows),
//! and `clusterer::assign` (local-to-global speaker id assignment).

use std::collections::HashMap;

/// { true }
/// `pub fn solve(cost: &[Vec<f32>]) -> Option<Vec<usize>>`
/// { ret.as_ref().map_or(true, |v| v.len() == cost.len()) }
/// Solve the assignment problem for an N×N cost matrix.
///
/// Returns a `Vec<usize>` of length N where `result[i]` is the column assigned to row `i`.
/// Each column is assigned to exactly one row. The total cost
/// `sum(cost[i][result[i]])` is minimized.
///
/// **Requires:** `cost` is square (every row has length `cost.len()`).
/// **Returns** `None` if `cost` is not square. An empty matrix returns `Some(vec![])`.
///
/// Implementation: classic Kuhn-Munkres in O(N³) using row/column potentials
/// (u/v) and shortest-path augmentation. Index 0 is reserved as a sentinel,
/// so internal arrays are length N+1.
pub fn solve(cost: &[Vec<f32>]) -> Option<Vec<usize>> {
    let n = cost.len();
    if n == 0 {
        return Some(Vec::new());
    }
    if cost.iter().any(|row| row.len() != n) {
        return None;
    }

    let inf = f32::INFINITY;
    let mut u = vec![0.0_f32; n + 1];
    let mut v = vec![0.0_f32; n + 1];
    // p[j] = row assigned to column j (0 = unassigned, sentinel)
    let mut p = vec![0_usize; n + 1];
    // way[j] = column predecessor in augmenting path
    let mut way = vec![0_usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0_usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0_usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            // Update potentials
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Reconstruct: walk back via `way` and fix `p`
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut result = vec![0_usize; n];
    for j in 1..=n {
        if p[j] > 0 {
            result[p[j] - 1] = j - 1;
        }
    }
    Some(result)
}

/// { true }
/// `pub(crate) fn map_max_cooccurrence(cooccurrence: &HashMap<(u32, u32), u64>) -> HashMap<u32, u32>`
/// { ret.len() <= cooccurrence.len() }
/// Optimal 1-to-1 mapping from hypothesis speaker IDs to reference speaker IDs,
/// maximizing total co-occurrence.
///
/// `cooccurrence` maps `(hyp_id, ref_id)` pairs to their co-occurrence count
/// (frames for DER, scored words for WDER). Distinct ids are sorted for
/// deterministic output, the square cost matrix uses `-count` costs (padding
/// cells stay 0.0), and the assignment is filtered to real pairs: the solver
/// may route leftover rows/columns through zero-cost padding cells, so pairs
/// with no actual co-occurrence are dropped. Counts cast to f32 are exact
/// below ~16.7M (f32 has a 24-bit mantissa); callers cap their grids well
/// under that. An empty map yields an empty mapping.
pub(crate) fn map_max_cooccurrence(cooccurrence: &HashMap<(u32, u32), u64>) -> HashMap<u32, u32> {
    if cooccurrence.is_empty() {
        return HashMap::new();
    }

    // Distinct hyp ids (rows) and ref ids (cols), sorted for deterministic output.
    let mut hyp_ids: Vec<u32> = cooccurrence.keys().map(|&(h, _)| h).collect();
    hyp_ids.sort_unstable();
    hyp_ids.dedup();
    let mut ref_ids: Vec<u32> = cooccurrence.keys().map(|&(_, r)| r).collect();
    ref_ids.sort_unstable();
    ref_ids.dedup();

    // Square cost matrix: cost = -co-occurrence so minimizing cost maximizes
    // agreement; padding cells stay 0.0.
    let n = hyp_ids.len().max(ref_ids.len());
    let mut cost = vec![vec![0.0_f32; n]; n];
    for (&(h, r), &count) in cooccurrence {
        if let (Ok(i), Ok(j)) = (hyp_ids.binary_search(&h), ref_ids.binary_search(&r)) {
            cost[i][j] = -(count as f32);
        }
    }

    let assignment = match solve(&cost) {
        Some(a) => a,
        None => return HashMap::new(),
    };

    let mut mapping: HashMap<u32, u32> = HashMap::new();
    for (row, &col) in assignment.iter().enumerate() {
        // Map only real (non-padding) speakers that actually co-occur — the
        // solver may pair leftover rows/cols through zero-cost padding cells.
        if let (Some(&h), Some(&r)) = (hyp_ids.get(row), ref_ids.get(col))
            && cooccurrence.get(&(h, r)).copied().unwrap_or(0) > 0
        {
            mapping.insert(h, r);
        }
    }

    mapping
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_matrix_returns_empty_assignment() {
        let cost: Vec<Vec<f32>> = Vec::new();
        let assignment = solve(&cost).expect("empty matrix is valid");
        assert!(assignment.is_empty());
    }

    #[test]
    fn one_by_one_matrix_returns_self() {
        let cost = vec![vec![3.5_f32]];
        let assignment = solve(&cost).expect("1x1 valid");
        assert_eq!(assignment, vec![0]);
    }

    #[test]
    fn diagonal_zero_matrix_returns_identity() {
        let n = 3;
        let mut cost = vec![vec![10.0_f32; n]; n];
        for (i, row) in cost.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        let assignment = solve(&cost).expect("3x3 valid");
        assert_eq!(assignment, vec![0, 1, 2]);
    }

    #[test]
    fn anti_diagonal_zero_matrix_returns_reverse_permutation() {
        let cost = vec![
            vec![10.0_f32, 10.0, 0.0],
            vec![10.0, 0.0, 10.0],
            vec![0.0, 10.0, 10.0],
        ];
        let assignment = solve(&cost).expect("3x3 valid");
        assert_eq!(assignment, vec![2, 1, 0]);
    }

    #[test]
    fn permutation_matrix_recovered() {
        let cost = vec![
            vec![5.0_f32, 0.0, 5.0],
            vec![5.0, 5.0, 0.0],
            vec![0.0, 5.0, 5.0],
        ];
        let assignment = solve(&cost).expect("3x3 valid");
        assert_eq!(assignment, vec![1, 2, 0]);
    }

    #[test]
    fn rejects_non_square_matrix() {
        let cost = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        assert!(solve(&cost).is_none());
    }

    #[test]
    fn handles_negative_costs() {
        let cost = vec![vec![-1.0_f32, -3.0], vec![-2.0, -5.0]];
        // Best: row 0 → col 0 (-1) + row 1 → col 1 (-5) = -6
        let assignment = solve(&cost).expect("2x2 valid");
        assert_eq!(assignment, vec![0, 1]);
    }

    #[test]
    fn cost_matrix_with_repeated_rows_still_assigns_unique_columns() {
        let cost = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];
        let assignment = solve(&cost).expect("3x3 valid");
        let mut sorted = assignment.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2], "must be a permutation");
    }

    #[test]
    fn map_max_cooccurrence_empty_input_maps_to_empty() {
        let cooccurrence: HashMap<(u32, u32), u64> = HashMap::new();
        assert!(map_max_cooccurrence(&cooccurrence).is_empty());
    }

    #[test]
    fn map_max_cooccurrence_prefers_optimal_over_greedy() {
        // (hyp 0, ref 0)=10, (hyp 0, ref 1)=9, (hyp 1, ref 0)=8.
        // Greedy would take 0->0 (10); optimal takes 0->1 (9) + 1->0 (8) = 17.
        let mut cooccurrence: HashMap<(u32, u32), u64> = HashMap::new();
        cooccurrence.insert((0, 0), 10);
        cooccurrence.insert((0, 1), 9);
        cooccurrence.insert((1, 0), 8);
        let mapping = map_max_cooccurrence(&cooccurrence);
        assert_eq!(mapping.get(&0), Some(&1));
        assert_eq!(mapping.get(&1), Some(&0));
    }

    #[test]
    fn map_max_cooccurrence_drops_padding_only_pairs() {
        // Three hyp ids but a single ref id: only one hyp can win the ref
        // column; the other two must not be mapped through padding cells.
        let mut cooccurrence: HashMap<(u32, u32), u64> = HashMap::new();
        cooccurrence.insert((0, 7), 5);
        cooccurrence.insert((1, 7), 3);
        cooccurrence.insert((2, 7), 1);
        let mapping = map_max_cooccurrence(&cooccurrence);
        assert_eq!(mapping.len(), 1);
        assert_eq!(mapping.get(&0), Some(&7));
    }
}
