#![no_main]

use libfuzzer_sys::fuzz_target;
use polyvoice::{DiarizationConfig, SpeakerCluster};

fuzz_target!(|data: &[u8]| {
    // Convert bytes to f32 embedding (each 4 bytes).
    let embedding: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let config = DiarizationConfig::default();
    let mut cluster = SpeakerCluster::new(config);

    // Must not panic for any input (including empty embedding).
    let _ = cluster.assign(&embedding);
});
