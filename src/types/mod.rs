//! Core types for speaker diarization.
//!
//! These types are shared across the offline pipeline, online diarizer, and
//! evaluation code. Start with [`DiarizationResult`] and [`SpeakerId`].

mod confidence;
mod config;
mod ids;
mod measures;
mod profile;
mod result;

pub use confidence::{
    CONFIDENCE_SIM_MIDPOINT, CONFIDENCE_SIM_STEEPNESS, confidence_from_distance,
    confidence_from_similarity, confidence_from_similarity_params, mean_speaker_embeddings,
    segment_confidences_from_embeddings,
};
pub use config::{
    ClusterConfig, ConfigError, DEFAULT_AHC_THRESHOLD, DiarizationConfig, SpeechFilterConfig,
    WindowConfig,
};
pub use ids::{SpeakerId, SpeakerIdRemap};
pub use measures::{Confidence, SampleRate, TimeRange};
pub use profile::{Profile, ProfileParseError};
pub use result::{
    AudioMeta, DiarizationResult, Provenance, Segment, SpeakerSummary, SpeakerTurn, Transcript,
    Word, WordAlignment, exclusive_turns, remap_segments, remap_turns,
};

#[cfg(test)]
use result::EXCLUSIVE_FRAME_SECS;

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod speaker_id_remap_tests {
    use super::*;

    #[test]
    fn from_mapping_accepts_unique_old_ids() {
        let mapping = vec![
            (SpeakerId(0), SpeakerId(0)),
            (SpeakerId(1), SpeakerId(0)),
            (SpeakerId(2), SpeakerId(1)),
        ];
        let remap = SpeakerIdRemap::from_mapping(mapping).unwrap();
        assert_eq!(remap.len(), 3);
        assert_eq!(remap.remap(SpeakerId(0)), SpeakerId(0));
        assert_eq!(remap.remap(SpeakerId(1)), SpeakerId(0));
        assert_eq!(remap.remap(SpeakerId(2)), SpeakerId(1));
        assert_eq!(remap.remap(SpeakerId(99)), SpeakerId(99));
    }

    #[test]
    fn from_mapping_rejects_duplicate_old_ids() {
        let mapping = vec![(SpeakerId(0), SpeakerId(1)), (SpeakerId(0), SpeakerId(2))];
        assert!(SpeakerIdRemap::from_mapping(mapping).is_none());
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn mobile_profile_uses_cam_pp_dim() {
        assert_eq!(Profile::Mobile.embedding_dim(), 512);
    }

    #[test]
    fn balanced_profile_uses_resnet34_dim() {
        assert_eq!(Profile::Balanced.embedding_dim(), 256);
    }

    #[test]
    fn custom_profile_dim_is_unresolved() {
        assert_eq!(Profile::Custom.embedding_dim(), 0);
    }

    #[test]
    fn default_thresholds_match_spec() {
        // Profile default thresholds are part of the public contract.
        assert!((Profile::Mobile.default_threshold() - 0.55).abs() < 1e-6);
        assert!((Profile::Balanced.default_threshold() - 0.45).abs() < 1e-6);
        assert!((Profile::Custom.default_threshold() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn manifest_id_for_each_variant() {
        assert_eq!(Profile::Mobile.manifest_id(), "mobile");
        assert_eq!(Profile::Balanced.manifest_id(), "balanced");
        assert_eq!(Profile::Custom.manifest_id(), "custom");
    }

    #[test]
    fn from_str_parses_kebab_and_lowercase() {
        assert_eq!("mobile".parse::<Profile>().unwrap(), Profile::Mobile);
        assert_eq!("Mobile".parse::<Profile>().unwrap(), Profile::Mobile);
        assert_eq!("balanced".parse::<Profile>().unwrap(), Profile::Balanced);
        assert!("nope".parse::<Profile>().is_err());
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod diarization_result_tests {
    use super::*;

    fn turn(id: u32, start: f64, end: f64) -> SpeakerTurn {
        SpeakerTurn::new(SpeakerId(id), TimeRange { start, end })
    }

    #[test]
    fn new_stamps_schema_version_and_provenance_version() {
        let r = DiarizationResult::new(vec![], vec![], 0);
        assert_eq!(r.schema_version, "diarization-result-v1");
        assert_eq!(r.provenance.version, env!("CARGO_PKG_VERSION"));
        assert!(r.speakers.is_empty());
    }

    #[test]
    fn speakers_rollup_matches_turns_with_dual_id() {
        let turns = vec![turn(0, 0.0, 2.0), turn(1, 2.0, 5.0), turn(0, 6.0, 7.0)];
        let r = DiarizationResult::new(vec![], turns, 2);
        assert_eq!(r.speakers.len(), 2);
        // Dual representation: numeric id AND canonical string label.
        assert_eq!(r.speakers[0].id, 0);
        assert_eq!(r.speakers[0].label, "SPEAKER_00");
        assert_eq!(r.speakers[0].turn_count, 2);
        assert!((r.speakers[0].total_speech_s - 3.0).abs() < 1e-9); // 2.0 + 1.0
        assert_eq!(r.speakers[1].id, 1);
        assert_eq!(r.speakers[1].label, "SPEAKER_01");
        assert_eq!(r.speakers[1].turn_count, 1);
        assert!((r.speakers[1].total_speech_s - 3.0).abs() < 1e-9);
    }

    #[test]
    fn old_json_without_metadata_deserializes() {
        // JSON shaped like the pre-v1 result (no metadata fields).
        let json = r#"{"segments":[],"turns":[],"num_speakers":0}"#;
        let r: DiarizationResult = serde_json::from_str(json).unwrap();
        assert_eq!(r.num_speakers, 0);
        assert_eq!(r.schema_version, "diarization-result-v1"); // serde default
        assert_eq!(r.audio, AudioMeta::default());
        assert_eq!(r.provenance, Provenance::default());
        assert!(r.speakers.is_empty());
    }

    #[test]
    fn round_trips_through_json_with_builders() {
        let r = DiarizationResult::new(vec![], vec![turn(0, 0.0, 1.0)], 1)
            .with_audio(12.5, 16000)
            .with_provenance(Provenance {
                profile: "balanced".to_owned(),
                ..Provenance::default()
            });
        let json = serde_json::to_string(&r).unwrap();
        let back: DiarizationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
        assert_eq!(back.audio.sample_rate, 16000);
        assert_eq!(back.provenance.profile, "balanced");
        // version preserved by the builder when the supplied one is empty.
        assert_eq!(back.provenance.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn word_and_transcript_round_trip() {
        let t = Transcript {
            words: vec![
                Word {
                    word: "hello".into(),
                    time: TimeRange {
                        start: 0.0,
                        end: 0.4,
                    },
                    confidence: 0.95,
                },
                Word {
                    word: "world".into(),
                    time: TimeRange {
                        start: 0.4,
                        end: 0.9,
                    },
                    confidence: 0.88,
                },
            ],
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
        assert_eq!(back.words.len(), 2);
        assert_eq!(back.words[0].word, "hello");
        assert_eq!(Transcript::default().words.len(), 0);
    }

    #[test]
    fn exclusive_collapses_overlap_to_one_speaker_per_frame() {
        // Overlap on [2, 4): both spk0 and spk1 active. spk0 turn is longer (0-4)
        // than spk1 (2-6), so exclusive should keep spk0 on the overlap.
        let turns = vec![turn(0, 0.0, 4.0), turn(1, 2.0, 6.0)];
        let ex = exclusive_turns(&turns);
        // Frame check: never two speakers.
        assert_exclusive_one_speaker(&ex, 6.0);
        // Speech coverage equals the union [0, 6).
        let speech: f64 = ex.iter().map(|t| t.time.duration()).sum();
        assert!(
            (speech - 6.0).abs() < 0.02,
            "exclusive speech coverage should match union, got {speech}"
        );
    }

    #[test]
    fn exclusive_no_overlap_is_identity_up_to_frame_quantize() {
        let turns = vec![turn(0, 0.0, 2.0), turn(1, 2.0, 4.0)];
        let ex = exclusive_turns(&turns);
        assert_exclusive_one_speaker(&ex, 4.0);
        assert_eq!(ex.len(), 2);
        assert_eq!(ex[0].speaker, SpeakerId(0));
        assert_eq!(ex[1].speaker, SpeakerId(1));
    }

    #[test]
    fn with_exclusive_populates_field_without_touching_turns() {
        let turns = vec![turn(0, 0.0, 3.0), turn(1, 2.0, 5.0)];
        let r = DiarizationResult::new(vec![], turns.clone(), 2).with_exclusive();
        assert_eq!(r.turns, turns);
        assert!(!r.exclusive_turns.is_empty());
        assert_exclusive_one_speaker(&r.exclusive_turns, 5.0);
        // Empty exclusive_turns is omitted from JSON.
        let bare = DiarizationResult::new(vec![], turns, 2);
        let json = serde_json::to_string(&bare).unwrap();
        assert!(!json.contains("exclusive_turns"));
        let with = bare.with_exclusive();
        let json2 = serde_json::to_string(&with).unwrap();
        assert!(json2.contains("exclusive_turns"));
    }

    #[test]
    fn confidence_from_similarity_is_monotone() {
        let sims = [-1.0f32, -0.5, 0.0, 0.3, 0.5, 0.7, 0.9, 1.0];
        let mut prev = -1.0f32;
        for &s in &sims {
            let c = confidence_from_similarity(s);
            assert!(
                (0.0..=1.0).contains(&c),
                "conf {c} out of range for sim {s}"
            );
            assert!(
                c + 1e-6 >= prev,
                "not monotone: sim {s} conf {c} < prev {prev}"
            );
            prev = c;
        }
        // Larger distance → lower confidence.
        let d_small = confidence_from_distance(0.1);
        let d_large = confidence_from_distance(0.8);
        assert!(d_small > d_large, "{d_small} should beat {d_large}");
    }

    #[test]
    fn mean_speaker_embeddings_are_l2_normalized_and_deterministic() {
        let labels = [SpeakerId(0), SpeakerId(0), SpeakerId(1), SpeakerId(1)];
        let embeddings = vec![
            vec![3.0, 0.0],
            vec![0.0, 4.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
        ];
        let a = mean_speaker_embeddings(&labels, &embeddings);
        let b = mean_speaker_embeddings(&labels, &embeddings);
        assert_eq!(a, b, "must be deterministic");
        assert_eq!(a.len(), 2);
        for (_, emb) in &a {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
        }
        // Speaker 1 mean is [1,0] already unit.
        assert!((a[1].1[0] - 1.0).abs() < 1e-5);
        // Attach to result.
        let r = DiarizationResult::new(vec![], vec![turn(0, 0.0, 1.0), turn(1, 1.0, 2.0)], 2)
            .with_speaker_embeddings(&a);
        assert!(r.speakers[0].embedding.is_some());
        assert!(r.speakers[1].embedding.is_some());
        let confs = segment_confidences_from_embeddings(&labels, &embeddings);
        assert_eq!(confs.len(), 4);
        assert!(confs.iter().all(|&c| (0.0..=1.0).contains(&c)));
    }

    fn assert_exclusive_one_speaker(turns: &[SpeakerTurn], max_time: f64) {
        let n = ((max_time / EXCLUSIVE_FRAME_SECS).ceil() as usize) + 1;
        let mut counts = vec![0u32; n];
        for t in turns {
            let s = (t.time.start / EXCLUSIVE_FRAME_SECS) as usize;
            let e = (t.time.end / EXCLUSIVE_FRAME_SECS).ceil() as usize;
            for c in counts.iter_mut().take(e.min(n)).skip(s) {
                *c += 1;
            }
        }
        assert!(
            counts.iter().all(|&c| c <= 1),
            "exclusive timeline has dual-speaker frames"
        );
    }
}

#[cfg(kani)]
mod kani_proofs;
