#![no_main]

use libfuzzer_sys::fuzz_target;
use polyvoice::{segment_speech, DiarizationConfig, EnergyVad, VadConfig};

fuzz_target!(|data: &[u8]| {
    let samples: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let mut vad = EnergyVad::new(-40.0, 16000, 512);
    let config = DiarizationConfig::default();
    let vad_config = VadConfig::default();

    // Must not panic for any input.
    let _ = segment_speech(&mut vad, &samples, &config, &vad_config);
});
