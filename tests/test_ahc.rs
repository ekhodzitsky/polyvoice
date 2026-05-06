use polyvoice::ahc::agglomerative_cluster;
use polyvoice::utils::l2_normalize;

fn unit_vec(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[axis] = 1.0;
    v
}

fn noisy_vec(dim: usize, axis: usize, noise: f32) -> Vec<f32> {
    let mut v = unit_vec(dim, axis);
    v[(axis + 1) % dim] = noise;
    l2_normalize(&mut v);
    v
}

#[test]
fn test_ahc_two_speakers() {
    let embeddings = vec![
        unit_vec(256, 0),
        unit_vec(256, 1),
        noisy_vec(256, 0, 0.05),
        noisy_vec(256, 1, 0.05),
    ];

    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert_eq!(labels.len(), 4);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[1], labels[3]);
    assert_ne!(labels[0], labels[1]);
}

#[test]
fn test_ahc_single_speaker() {
    let mut embeddings = Vec::new();
    for i in 0..5 {
        embeddings.push(noisy_vec(256, 0, 0.01 * i as f32));
    }

    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert!(labels.iter().all(|&l| l == labels[0]));
}

#[test]
fn test_ahc_all_different() {
    let embeddings = vec![unit_vec(256, 0), unit_vec(256, 1), unit_vec(256, 2)];

    let labels = agglomerative_cluster(&embeddings, 0.99);
    assert_ne!(labels[0], labels[1]);
    assert_ne!(labels[1], labels[2]);
    assert_ne!(labels[0], labels[2]);
}

#[test]
fn test_ahc_empty() {
    let embeddings: Vec<Vec<f32>> = vec![];
    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert!(labels.is_empty());
}

#[test]
fn test_ahc_single_embedding() {
    let embeddings = vec![unit_vec(256, 0)];
    let labels = agglomerative_cluster(&embeddings, 0.5);
    assert_eq!(labels, vec![0]);
}
