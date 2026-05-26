//! Snapshot tests for RTTM parsing and serialization stability.

#![allow(clippy::unwrap_used)]

use polyvoice::rttm::{parse_rttm_file, to_speaker_turns};
use std::io::Write;

#[test]
fn snapshot_rttm_roundtrip() {
    let rttm = "SPEAKER test 1 0.0 1.0 <NA> <NA> 0 <NA> <NA>\n\
                  SPEAKER test 1 1.5 2.0 <NA> <NA> 1 <NA> <NA>\n";
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(rttm.as_bytes()).unwrap();
    let segs = parse_rttm_file(tmp.path()).unwrap();
    let (turns, _map) = to_speaker_turns(&segs);
    insta::assert_snapshot!(format!("{turns:#?}"));
}
