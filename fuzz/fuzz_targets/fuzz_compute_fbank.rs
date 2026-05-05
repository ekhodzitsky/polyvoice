#![no_main]

use libfuzzer_sys::fuzz_target;
use polyvoice::features::{FbankConfig, FbankExtractor};

fuzz_target!(|data: &[u8]| {
    // Convert bytes to f32 samples (each 4 bytes).
    let samples: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let config = FbankConfig::default();
    let extractor = FbankExtractor::new(config);
    // Must not panic for any input.
    let _ = extractor.extract(&samples);
});
