use super::*;

fn unit(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    v[axis] = 1.0;
    v
}

#[test]
fn compute_centroids_l2_normalized() {
    let embeddings = vec![unit(3, 0), unit(3, 0), unit(3, 1), unit(3, 1)];
    let labels = vec![0, 0, 1, 1];
    let centroids = compute_centroids(&embeddings, &labels);
    assert_eq!(centroids.len(), 2);
    for c in &centroids {
        let n: f32 = c.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (n - 1.0).abs() < 1e-3,
            "centroid not L2-normalized: norm={n}"
        );
    }
}

#[test]
fn compute_centroids_drops_empty_clusters() {
    // Labels skip from 0 to 2; cluster 1 has no members.
    let embeddings = vec![unit(3, 0), unit(3, 1), unit(3, 1)];
    let labels = vec![0, 2, 2];
    let centroids = compute_centroids(&embeddings, &labels);
    assert_eq!(centroids.len(), 2);
    let speakers: Vec<u32> = centroids.iter().map(|c| c.speaker.0).collect();
    assert_eq!(speakers, vec![0, 2]);
}

#[test]
fn compute_centroids_sorted_by_speaker_id() {
    let embeddings = vec![unit(3, 0), unit(3, 1), unit(3, 2)];
    let labels = vec![5, 1, 3];
    let centroids = compute_centroids(&embeddings, &labels);
    let speakers: Vec<u32> = centroids.iter().map(|c| c.speaker.0).collect();
    assert_eq!(speakers, vec![1, 3, 5]);
}

#[test]
fn compute_centroids_empty_input_returns_empty() {
    let centroids = compute_centroids(&[], &[]);
    assert!(centroids.is_empty());
}

#[test]
fn compute_centroids_label_mismatch_returns_empty() {
    // Mismatched lengths: caller bug, conservative empty return rather than panic.
    let centroids = compute_centroids(&[unit(3, 0)], &[0, 1]);
    assert!(centroids.is_empty());
}
