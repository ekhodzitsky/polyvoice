#![no_main]

use libfuzzer_sys::fuzz_target;
use polyvoice::overlap::detect_overlaps;
use polyvoice::{Segment, SpeakerId, TimeRange};

fuzz_target!(|data: &[u8]| {
    // Each segment is encoded as 24 bytes: start(f64), end(f64), speaker_id(u32, padded).
    let mut segments = Vec::new();
    for chunk in data.chunks_exact(24) {
        let start = f64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let end = f64::from_le_bytes([
            chunk[8], chunk[9], chunk[10], chunk[11],
            chunk[12], chunk[13], chunk[14], chunk[15],
        ]);
        let speaker_id = u32::from_le_bytes([chunk[16], chunk[17], chunk[18], chunk[19]]);
        segments.push(Segment {
            time: TimeRange { start, end },
            speaker: Some(SpeakerId(speaker_id)),
            confidence: None,
        });
    }

    // Must not panic for any input.
    let _ = detect_overlaps(&segments);
});
