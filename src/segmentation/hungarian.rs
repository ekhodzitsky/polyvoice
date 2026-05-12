//! Kuhn-Munkres minimum-cost assignment for square cost matrices.
//!
//! Pure Rust, wasm32-clean. Used by the segmentation aggregator to
//! align local speaker indices between overlapping windows.

/// { TODO: precondition }
/// `pub fn solve(cost: &[Vec<f32>]) -> Option<Vec<usize>>`
/// { TODO: postcondition }
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
#[allow(dead_code)]
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
}
