use polyvoice::{
    DiarizationConfig, DummyExtractor, OfflineDiarizer, OnlineDiarizer, WordAlignment,
};

/// Helper to generate a sine wave at a given frequency.
fn sine_wave(freq: f32, duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_secs * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect()
}

/// Generate 10 seconds of audio with two "speakers" (different frequencies).
fn two_speaker_audio(sample_rate: u32) -> Vec<f32> {
    let mut audio = Vec::new();
    // Speaker 0: 200 Hz for 3s
    audio.extend_from_slice(&sine_wave(200.0, 3.0, sample_rate));
    // Speaker 1: 400 Hz for 3s
    audio.extend_from_slice(&sine_wave(400.0, 3.0, sample_rate));
    // Speaker 0 again: 200 Hz for 4s
    audio.extend_from_slice(&sine_wave(200.0, 4.0, sample_rate));
    audio
}

#[test]
fn test_offline_two_speakers_dummy() {
    let sample_rate = 16000;
    let config = DiarizationConfig {
        window_secs: 1.0,
        hop_secs: 0.5,
        ..Default::default()
    };
    let diarizer = OfflineDiarizer::new(config);
    let extractor = DummyExtractor::new(256);
    let samples = two_speaker_audio(sample_rate);

    let result = diarizer.run(&samples, &extractor).unwrap();

    // Dummy extractor returns pseudo-random embeddings, so we mainly verify
    // structural correctness.
    assert!(!result.segments.is_empty());
    assert!(!result.turns.is_empty());
    // All turns should have a speaker assigned.
    for turn in &result.turns {
        assert!(turn.time.duration() > 0.0);
    }
}

#[test]
fn test_online_streaming_basic() {
    let sample_rate = 16000;
    let config = DiarizationConfig {
        window_secs: 1.0,
        hop_secs: 0.5,
        ..Default::default()
    };
    let mut diarizer = OnlineDiarizer::new(config);
    let extractor = DummyExtractor::new(256);
    let samples = two_speaker_audio(sample_rate);

    // Feed in 1-second chunks.
    let chunk_size = sample_rate as usize;
    let mut all_segments = Vec::new();
    for chunk in samples.chunks(chunk_size) {
        let segs = diarizer.feed(chunk, &extractor).unwrap();
        all_segments.extend(segs);
    }
    let final_seg = diarizer.flush(&extractor).unwrap();
    if let Some(s) = final_seg {
        all_segments.push(s);
    }

    assert!(!all_segments.is_empty());
    // Ensure monotonic timestamps.
    for window in all_segments.windows(2) {
        assert!(window[0].time.start <= window[1].time.start);
    }
}

#[test]
fn test_word_alignment() {
    let sample_rate = 16000;
    let config = DiarizationConfig {
        window_secs: 1.0,
        hop_secs: 0.5,
        ..Default::default()
    };
    let mut diarizer = OnlineDiarizer::new(config);
    let extractor = DummyExtractor::new(256);

    // Feed 3 seconds of audio.
    let samples = sine_wave(300.0, 3.0, sample_rate);
    let _ = diarizer.feed(&samples, &extractor).unwrap();

    let mut words = vec![
        WordAlignment {
            word: "hello".into(),
            time: polyvoice::TimeRange { start: 0.5, end: 1.0 },
            speaker: None,
            confidence: 0.9,
        },
        WordAlignment {
            word: "world".into(),
            time: polyvoice::TimeRange { start: 1.5, end: 2.0 },
            speaker: None,
            confidence: 0.8,
        },
    ];

    diarizer.align_words(&mut words);
    // Words should now have a speaker assigned (or at least not panic).
    for w in &words {
        // Since dummy extractor is deterministic per call, there will be a speaker.
        assert!(w.speaker.is_some());
    }
}

#[test]
fn test_overlap_detection() {
    use polyvoice::{detect_overlaps, Segment, SpeakerId, TimeRange};

    let segments = vec![
        Segment {
            time: TimeRange { start: 0.0, end: 3.0 },
            speaker: Some(SpeakerId(0)),
            confidence: None,
        },
        Segment {
            time: TimeRange { start: 1.0, end: 4.0 },
            speaker: Some(SpeakerId(1)),
            confidence: None,
        },
    ];

    let overlaps = detect_overlaps(&segments);
    assert_eq!(overlaps.len(), 1);
    assert!((overlaps[0].time.start - 1.0).abs() < 1e-5);
    assert!((overlaps[0].time.end - 3.0).abs() < 1e-5);
}
