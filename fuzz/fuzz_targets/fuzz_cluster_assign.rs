#![no_main]

use libfuzzer_sys::fuzz_target;
use polyvoice::streaming::ArrivalOrderSpeakerCache;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to f32 embedding (each 4 bytes).
    let embedding: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    // Production online centroid path (replaces removed SpeakerCluster).
    let mut cache = ArrivalOrderSpeakerCache::new(
        8,    // cap
        0.45, // match_threshold
        2,    // min_hits_to_stable
        0.05, // prefer_current_margin
    );

    // Must not panic for any input (including empty embedding).
    let _ = cache.assign(&embedding);
});
