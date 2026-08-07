use super::*;

/// In-memory dummy used by trait conformance tests.
struct ConstantResegmenter {
    out: Vec<SpeakerTurn>,
}

impl Resegmenter for ConstantResegmenter {
    fn resegment(&self, _inputs: ResegmentInputs<'_>) -> Result<Vec<SpeakerTurn>, ResegmentError> {
        Ok(self.out.clone())
    }
}

fn turn(start: f64, end: f64, spk: u32) -> SpeakerTurn {
    SpeakerTurn {
        speaker: SpeakerId(spk),
        time: TimeRange { start, end },
        text: None,
        stable: true,
    }
}

#[test]
fn resegmenter_trait_object_is_dyn_compatible() {
    let r = ConstantResegmenter {
        out: vec![turn(0.0, 1.0, 0)],
    };
    let _b: Box<dyn Resegmenter> = Box::new(r);
}

#[test]
fn resegmenter_returns_owned_turns() {
    let r = ConstantResegmenter {
        out: vec![turn(0.0, 1.0, 0), turn(1.0, 2.0, 1)],
    };
    let inputs = ResegmentInputs {
        primary_turns: &[],
        speaker_centroids: &[],
        overlap_regions: &[],
    };
    let out = r.resegment(inputs).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].speaker, SpeakerId(0));
}

#[test]
fn error_centroid_dim_mismatch_displays() {
    let err = ResegmentError::CentroidDimMismatch {
        index: 1,
        expected: 192,
        actual: 256,
    };
    let msg = format!("{err}");
    assert!(msg.contains("192"));
    assert!(msg.contains("256"));
    assert!(msg.contains("index 1"));
}

#[test]
fn error_overlap_dim_mismatch_displays() {
    let err = ResegmentError::OverlapDimMismatch {
        index: 0,
        expected: 192,
        actual: 64,
    };
    let msg = format!("{err}");
    assert!(msg.contains("192"));
    assert!(msg.contains("64"));
}

#[test]
fn error_missing_primary_centroid_displays() {
    let err = ResegmentError::MissingPrimaryCentroid {
        index: 2,
        primary: SpeakerId(7),
    };
    let msg = format!("{err}");
    assert!(msg.contains('2'));
    assert!(msg.contains('7'));
}
