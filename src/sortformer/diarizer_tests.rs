use super::*;
use crate::onnx::{InferenceError, InferenceTensor, NamedTensor, TensorData};
use crate::sortformer::PostProcessConfig;

/// Mock that returns zeros shaped like Sortformer outputs.
struct ZeroMock {
    input_names: Vec<String>,
    output_names: Vec<String>,
    chunk_len: usize,
}

impl ZeroMock {
    fn new(chunk_len: usize) -> Self {
        Self {
            input_names: vec![
                "chunk".into(),
                "chunk_lengths".into(),
                "spkcache".into(),
                "spkcache_lengths".into(),
                "fifo".into(),
                "fifo_lengths".into(),
            ],
            output_names: vec![OUT_PREDS.into(), OUT_EMBS.into()],
            chunk_len,
        }
    }
}

impl InferenceRuntime for ZeroMock {
    fn input_names(&self) -> &[String] {
        &self.input_names
    }
    fn output_names(&self) -> &[String] {
        &self.output_names
    }
    fn run(&mut self, inputs: &[NamedTensor<'_>]) -> Result<Vec<InferenceTensor>, InferenceError> {
        // Derive spkcache/fifo lengths from inputs so shapes stay consistent.
        let mut spk = 0usize;
        let mut fifo = 0usize;
        for nt in inputs {
            match nt.name {
                "spkcache_lengths" => {
                    if let TensorData::I64(v) = &nt.tensor.data {
                        spk = *v.first().unwrap_or(&0) as usize;
                    }
                }
                "fifo_lengths" => {
                    if let TensorData::I64(v) = &nt.tensor.data {
                        fifo = *v.first().unwrap_or(&0) as usize;
                    }
                }
                _ => {}
            }
        }
        // Emit chunk_len model frames of predictions after prefix.
        let total = spk + fifo + self.chunk_len;
        let preds = InferenceTensor::f32(
            vec![1, total, MAX_SPEAKERS],
            vec![0.0; total * MAX_SPEAKERS],
        );
        let embs = InferenceTensor::f32(
            vec![1, self.chunk_len, EMB_DIM],
            vec![0.0; self.chunk_len * EMB_DIM],
        );
        Ok(vec![preds, embs])
    }
    fn run_ordered(
        &mut self,
        _inputs: &[&InferenceTensor],
    ) -> Result<Vec<InferenceTensor>, InferenceError> {
        Err(InferenceError::Run("not used".into()))
    }
}

#[test]
fn mock_diarize_empty_audio() {
    let cfg = SortformerConfig::default();
    let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    let turns = d.diarize(&[]).unwrap();
    assert!(turns.is_empty());
}

#[test]
fn mock_diarize_silence_yields_no_turns() {
    // Zero mock → all probs 0 → no onsets.
    let cfg = SortformerConfig::default();
    let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    let audio = vec![0.0f32; SAMPLE_RATE as usize]; // 1 s
    let turns = d.diarize(&audio).unwrap();
    assert!(turns.is_empty(), "silence must not produce speakers");
}

#[test]
fn mock_chunk_preserves_state_across_calls() {
    let cfg = SortformerConfig {
        chunk_len: 4,
        fifo_len: 4,
        spkcache_len: 8,
        right_context: 1,
        ..Default::default()
    };
    let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    // ~1 s of audio
    let audio = vec![0.01f32; SAMPLE_RATE as usize];
    let _ = d.diarize_chunk(&audio).unwrap();
    // After one chunk FIFO may have grown; second call should not panic.
    let _ = d.diarize_chunk(&audio).unwrap();
    assert!(d.fifo_frames <= d.config.fifo_len || d.spkcache_frames > 0);
}

#[test]
fn from_runtime_rejects_via_config_validate() {
    let err = SortformerConfig::default()
        .with_max_speakers(8)
        .expect_err("cap");
    assert!(matches!(
        err,
        SortformerError::MaxSpeakersExceeded { requested: 8 }
    ));
}

/// Small streaming geometry so chunk math stays fast in tests.
fn small_cfg() -> SortformerConfig {
    SortformerConfig {
        chunk_len: 4,
        fifo_len: 4,
        spkcache_len: 8,
        right_context: 1,
        ..Default::default()
    }
}

/// Post-processing that passes scripted 0/0.9 probabilities straight
/// through (no median smoothing, no duration/gap filtering).
fn passthrough_post() -> PostProcessConfig {
    PostProcessConfig {
        onset: 0.5,
        offset: 0.4,
        pad_onset: 0.0,
        pad_offset: 0.0,
        min_duration_on: 0.0,
        min_duration_off: 0.0,
        median_window: 1,
    }
}

/// Scripted mock with configurable output shapes, fill values, output
/// names, and failure injection for state-machine and error-path tests.
struct ScriptedMock {
    input_names: Vec<String>,
    output_names: Vec<String>,
    preds_chunk_frames: usize,
    emb_frames: usize,
    preds_fill: f32,
    preds_last_dim: Option<usize>,
    embs_last_dim: Option<usize>,
    preds_i64: bool,
    fail_on_run: Option<usize>,
    runs: usize,
}

impl ScriptedMock {
    fn new(cfg: &SortformerConfig) -> Self {
        Self {
            input_names: vec![
                "chunk".into(),
                "chunk_lengths".into(),
                "spkcache".into(),
                "spkcache_lengths".into(),
                "fifo".into(),
                "fifo_lengths".into(),
            ],
            output_names: vec![OUT_PREDS.into(), OUT_EMBS.into()],
            preds_chunk_frames: cfg.chunk_len,
            emb_frames: cfg.chunk_len,
            preds_fill: 0.0,
            preds_last_dim: None,
            embs_last_dim: None,
            preds_i64: false,
            fail_on_run: None,
            runs: 0,
        }
    }
}

impl InferenceRuntime for ScriptedMock {
    fn input_names(&self) -> &[String] {
        &self.input_names
    }
    fn output_names(&self) -> &[String] {
        &self.output_names
    }
    fn run(&mut self, inputs: &[NamedTensor<'_>]) -> Result<Vec<InferenceTensor>, InferenceError> {
        self.runs += 1;
        if self.fail_on_run == Some(self.runs) {
            return Err(InferenceError::Run("scripted failure".into()));
        }
        let mut spk = 0usize;
        let mut fifo = 0usize;
        for nt in inputs {
            match nt.name {
                "spkcache_lengths" => {
                    if let TensorData::I64(v) = &nt.tensor.data {
                        spk = *v.first().unwrap_or(&0) as usize;
                    }
                }
                "fifo_lengths" => {
                    if let TensorData::I64(v) = &nt.tensor.data {
                        fifo = *v.first().unwrap_or(&0) as usize;
                    }
                }
                _ => {}
            }
        }
        let total = spk + fifo + self.preds_chunk_frames;
        let preds_dim = self.preds_last_dim.unwrap_or(MAX_SPEAKERS);
        let preds = if self.preds_i64 {
            InferenceTensor::i64(vec![1, total, preds_dim], vec![0; total * preds_dim])
        } else {
            let mut data = vec![0.0f32; total * preds_dim];
            if preds_dim == MAX_SPEAKERS {
                for t in (spk + fifo)..total {
                    for s in 0..MAX_SPEAKERS {
                        data[t * MAX_SPEAKERS + s] = self.preds_fill;
                    }
                }
            }
            InferenceTensor::f32(vec![1, total, preds_dim], data)
        };
        let emb_dim = self.embs_last_dim.unwrap_or(EMB_DIM);
        let embs = InferenceTensor::f32(
            vec![1, self.emb_frames, emb_dim],
            vec![0.0; self.emb_frames * emb_dim],
        );
        Ok(vec![preds, embs])
    }
    fn run_ordered(
        &mut self,
        _inputs: &[&InferenceTensor],
    ) -> Result<Vec<InferenceTensor>, InferenceError> {
        Err(InferenceError::Run("not used".into()))
    }
}

// --- pure helpers -------------------------------------------------------

#[test]
fn reclaim_f32_roundtrip_and_i64_fallback() {
    let t = InferenceTensor::f32(vec![2], vec![1.0, 2.0]);
    assert_eq!(reclaim_f32(t), vec![1.0, 2.0]);
    let t = InferenceTensor::i64(vec![1], vec![7]);
    assert!(reclaim_f32(t).is_empty());
}

#[test]
fn clip_turns_to_audio_trims_and_drops_overruns() {
    let mut turns = vec![
        SpeakerTurn {
            speaker: SpeakerId(0),
            time: TimeRange {
                start: 0.0,
                end: 5.0,
            },
            text: None,
            stable: true,
        },
        SpeakerTurn {
            speaker: SpeakerId(1),
            time: TimeRange {
                start: 3.0,
                end: 4.0,
            },
            text: None,
            stable: true,
        },
    ];
    clip_turns_to_audio(&mut turns, 2 * SAMPLE_RATE as usize); // 2 s of audio
    assert_eq!(turns.len(), 1);
    assert!((turns[0].time.end - 2.0).abs() < 1e-9);
}

#[test]
fn pad_or_slice_mel_pads_short_and_truncates_long() {
    let mel: Vec<f32> = (0..3 * N_MELS).map(|i| i as f32).collect();
    let padded = pad_or_slice_mel(&mel, 3, 5);
    assert_eq!(padded.len(), 5 * N_MELS);
    assert_eq!(padded[..3 * N_MELS], mel[..]);
    assert!(padded[3 * N_MELS..].iter().all(|&v| v == 0.0));

    let long: Vec<f32> = (0..7 * N_MELS).map(|i| i as f32).collect();
    let sliced = pad_or_slice_mel(&long, 7, 5);
    assert_eq!(sliced.len(), 5 * N_MELS);
    assert_eq!(sliced[..], long[..5 * N_MELS]);

    // Zero frames: all padding.
    let zero = pad_or_slice_mel(&[], 0, 2);
    assert!(zero.iter().all(|&v| v == 0.0));
}

#[test]
fn map_outputs_prefers_matching_names() {
    let names = vec![OUT_PREDS.to_owned(), OUT_EMBS.to_owned()];
    let outputs = vec![
        InferenceTensor::f32(vec![1], vec![1.0]),
        InferenceTensor::f32(vec![1], vec![2.0]),
    ];
    let map = map_outputs(&names, outputs).unwrap();
    assert!(map.contains_key(OUT_PREDS));
    assert!(map.contains_key(OUT_EMBS));
}

#[test]
fn map_outputs_positional_fallback_without_names() {
    let outputs = vec![
        InferenceTensor::f32(vec![1], vec![1.0]),
        InferenceTensor::f32(vec![1], vec![2.0]),
    ];
    let map = map_outputs(&[], outputs).unwrap();
    assert!(map.contains_key(OUT_PREDS));
    assert!(map.contains_key(OUT_EMBS));
}

#[test]
fn map_outputs_positional_fallback_on_name_count_mismatch() {
    let names = vec!["only_one".to_owned()];
    let outputs = vec![
        InferenceTensor::f32(vec![1], vec![1.0]),
        InferenceTensor::f32(vec![1], vec![2.0]),
    ];
    let map = map_outputs(&names, outputs).unwrap();
    assert!(map.contains_key(OUT_PREDS));
    assert!(map.contains_key(OUT_EMBS));
}

#[test]
fn map_outputs_single_output_maps_only_preds() {
    let outputs = vec![InferenceTensor::f32(vec![1], vec![1.0])];
    let map = map_outputs(&[], outputs).unwrap();
    assert!(map.contains_key(OUT_PREDS));
    assert!(!map.contains_key(OUT_EMBS));
}

#[test]
fn preds_shape_frames_accepts_batched_and_flat() {
    let batched = InferenceTensor::f32(vec![1, 7, MAX_SPEAKERS], vec![0.0; 7 * MAX_SPEAKERS]);
    assert_eq!(preds_shape_frames(&batched).unwrap(), 7);
    let flat = InferenceTensor::f32(vec![5, MAX_SPEAKERS], vec![0.0; 5 * MAX_SPEAKERS]);
    assert_eq!(preds_shape_frames(&flat).unwrap(), 5);
    let bad = InferenceTensor::f32(vec![1, 5, 3], vec![0.0; 15]);
    let err = preds_shape_frames(&bad).unwrap_err();
    assert!(matches!(err, SortformerError::Shape(_)));
    let bad_rank = InferenceTensor::f32(vec![5], vec![0.0; 5]);
    assert!(preds_shape_frames(&bad_rank).is_err());
}

#[test]
fn embs_shape_frames_accepts_batched_and_flat() {
    let batched = InferenceTensor::f32(vec![1, 3, EMB_DIM], vec![0.0; 3 * EMB_DIM]);
    assert_eq!(embs_shape_frames(&batched).unwrap(), 3);
    let flat = InferenceTensor::f32(vec![2, EMB_DIM], vec![0.0; 2 * EMB_DIM]);
    assert_eq!(embs_shape_frames(&flat).unwrap(), 2);
    let bad = InferenceTensor::f32(vec![1, 3, 16], vec![0.0; 48]);
    assert!(matches!(
        embs_shape_frames(&bad).unwrap_err(),
        SortformerError::Shape(_)
    ));
}

#[test]
fn log_pred_scores_rank_by_probability_and_stay_finite() {
    let mut preds = vec![0.0f32; MAX_SPEAKERS];
    preds[0] = 0.9;
    preds[1] = 0.3;
    // Zeros exercise the threshold clamping path.
    let scores = get_log_pred_scores(&preds, 1);
    assert!(scores.iter().all(|v| v.is_finite()));
    assert!(scores[0] > scores[1]);
    assert!(scores[1] > scores[2]);
}

#[test]
fn disable_low_scores_blanks_nonspeech_and_saturated_negatives() {
    let n_frames = 2;
    let mut preds = vec![0.0f32; n_frames * MAX_SPEAKERS];
    // Speaker 0: speech in both frames.
    preds[0] = 0.6;
    preds[MAX_SPEAKERS] = 0.6;
    // Speaker 1: below the speech threshold in both frames.
    preds[1] = 0.1;
    preds[MAX_SPEAKERS + 1] = 0.1;
    let mut scores = vec![f32::NEG_INFINITY; n_frames * MAX_SPEAKERS];
    scores[0] = -1.0; // negative but speaker has one positive frame
    scores[MAX_SPEAKERS] = 1.0; // the positive frame
    scores[1] = 5.0; // positive but not speech
    scores[MAX_SPEAKERS + 1] = 5.0;
    let out = disable_low_scores(&preds, scores, n_frames, 1);
    // Non-speech frames are always disabled.
    assert_eq!(out[1], f32::NEG_INFINITY);
    assert_eq!(out[MAX_SPEAKERS + 1], f32::NEG_INFINITY);
    // Negative score is disabled once the speaker meets min_pos via another frame.
    assert_eq!(out[0], f32::NEG_INFINITY);
    // The positive speech frame survives.
    assert_eq!(out[MAX_SPEAKERS], 1.0);
}

#[test]
fn disable_low_scores_keeps_negatives_below_min_pos_quota() {
    let n_frames = 1;
    let mut preds = vec![0.0f32; n_frames * MAX_SPEAKERS];
    preds[0] = 0.6;
    let mut scores = vec![f32::NEG_INFINITY; n_frames * MAX_SPEAKERS];
    scores[0] = -1.0; // negative, and no positive frames for the speaker
    let out = disable_low_scores(&preds, scores, n_frames, 1);
    assert_eq!(out[0], -1.0, "below min_pos quota the score survives");
}

#[test]
fn boost_topk_scores_boosts_only_finite_top_entries() {
    let n_frames = 3;
    let mut scores = vec![f32::NEG_INFINITY; n_frames * MAX_SPEAKERS];
    scores[0] = 1.0;
    scores[MAX_SPEAKERS] = 2.0;
    scores[2 * MAX_SPEAKERS] = 3.0;
    let out = boost_topk_scores(scores, n_frames, 1, 2.0);
    // Top entry (frame 2, score 3.0) boosted by -scale*ln(0.5).
    let expect = 3.0 - 2.0 * 0.5f32.ln();
    assert!((out[2 * MAX_SPEAKERS] - expect).abs() < 1e-5);
    // Other finite scores untouched, -inf untouched.
    assert_eq!(out[0], 1.0);
    assert_eq!(out[MAX_SPEAKERS], 2.0);
    assert_eq!(out[1], f32::NEG_INFINITY);
}

#[test]
fn topk_indices_marks_neg_inf_and_sil_frames_disabled() {
    // 2 real frames + 1 silence placeholder row (+inf), one -inf score.
    let n_frames = 3;
    let n_no_sil = 2;
    let mut scores = vec![1.0f32; n_frames * MAX_SPEAKERS];
    scores[0] = f32::NEG_INFINITY;
    // Silence placeholder row.
    for s in 0..MAX_SPEAKERS {
        scores[2 * MAX_SPEAKERS + s] = f32::INFINITY;
    }
    let (indices, disabled) = get_topk_indices(&scores, n_frames, n_no_sil, 4);
    assert_eq!(indices.len(), 4);
    assert_eq!(disabled.len(), 4);
    // +inf sil rows are selected first but land beyond n_no_sil → disabled.
    assert!(disabled.iter().any(|&d| d));
    // All returned frame indices for enabled entries are in range.
    for (&idx, &dis) in indices.iter().zip(disabled.iter()) {
        if !dis {
            assert!(idx < n_no_sil);
        }
    }
}

#[test]
fn topk_indices_pads_to_spkcache_len() {
    // Fewer finite entries than spkcache_len exercises the padding loop.
    let n_frames = 1;
    let scores = vec![f32::NEG_INFINITY; n_frames * MAX_SPEAKERS];
    let (indices, disabled) = get_topk_indices(&scores, n_frames, n_frames, 10);
    assert_eq!(indices.len(), 10);
    assert!(disabled.iter().all(|&d| d));
    assert!(indices.iter().all(|&i| i == 0));
}

// --- median filter / binarize -------------------------------------------

#[test]
fn median_filter_passthrough_for_small_windows_and_empty() {
    let mut cfg = small_cfg();
    cfg.post.median_window = 1;
    let d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg.clone());
    let preds = vec![0.1, 0.2, 0.3, 0.4];
    assert_eq!(d.median_filter(&preds), preds);
    assert!(d.median_filter(&[]).is_empty());

    cfg.post.median_window = 0;
    let d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    assert_eq!(d.median_filter(&preds), preds);
}

#[test]
fn median_filter_removes_single_frame_spike() {
    let mut cfg = small_cfg();
    cfg.post.median_window = 3;
    let d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    let n_frames = 5;
    let mut preds = vec![0.0f32; n_frames * MAX_SPEAKERS];
    preds[2 * MAX_SPEAKERS] = 1.0; // speaker 0 spike at frame 2
    let filtered = d.median_filter(&preds);
    assert_eq!(
        filtered[2 * MAX_SPEAKERS],
        0.0,
        "spike must be smoothed out"
    );
}

#[test]
fn binarize_merges_close_gaps_and_drops_short_segments() {
    let cfg = SortformerConfig {
        post: PostProcessConfig {
            onset: 0.5,
            offset: 0.4,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.1,  // needs ≥ 2 frames
            min_duration_off: 0.2, // 1-frame gaps merge
            median_window: 1,
        },
        ..Default::default()
    };
    let d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    let n_frames = 8;
    let mut preds = vec![0.0f32; n_frames * MAX_SPEAKERS];
    // Speaker 0: frames 0-1 on, frame 2 off, frames 3-4 on → merged.
    for t in [0usize, 1, 3, 4] {
        preds[t * MAX_SPEAKERS] = 0.9;
    }
    // Speaker 1: single on-frame → below min_duration_on → dropped.
    preds[6 * MAX_SPEAKERS + 1] = 0.9;
    let turns = d.binarize(&preds);
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].speaker, SpeakerId(0));
    assert!((turns[0].time.start - 0.0).abs() < 1e-9);
    assert!((turns[0].time.end - 5.0 * FRAME_DURATION_SECS as f64).abs() < 1e-9);
}

#[test]
fn binarize_closes_segment_open_at_stream_end() {
    let cfg = SortformerConfig {
        post: passthrough_post(),
        ..Default::default()
    };
    let d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    let n_frames = 4;
    let mut preds = vec![0.0f32; n_frames * MAX_SPEAKERS];
    preds[2 * MAX_SPEAKERS] = 0.9;
    preds[3 * MAX_SPEAKERS] = 0.9; // still on at the end
    let turns = d.binarize(&preds);
    assert_eq!(turns.len(), 1);
    assert!((turns[0].time.start - 2.0 * FRAME_DURATION_SECS as f64).abs() < 1e-9);
    assert!((turns[0].time.end - 4.0 * FRAME_DURATION_SECS as f64).abs() < 1e-9);
}

#[test]
fn binarize_respects_max_speakers_cap() {
    let cfg = SortformerConfig {
        max_speakers: 1,
        post: passthrough_post(),
        ..Default::default()
    };
    let d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    let n_frames = 3;
    let mut preds = vec![0.0f32; n_frames * MAX_SPEAKERS];
    for t in 0..n_frames {
        preds[t * MAX_SPEAKERS] = 0.9; // speaker 0
        preds[t * MAX_SPEAKERS + 1] = 0.9; // speaker 1 must be ignored
    }
    let turns = d.binarize(&preds);
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].speaker, SpeakerId(0));
}

#[test]
fn binarize_applies_onset_offset_padding() {
    let cfg = SortformerConfig {
        post: PostProcessConfig {
            onset: 0.5,
            offset: 0.4,
            pad_onset: 0.1,
            pad_offset: 0.2,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
            median_window: 1,
        },
        ..Default::default()
    };
    let d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    let n_frames = 6;
    let mut preds = vec![0.0f32; n_frames * MAX_SPEAKERS];
    preds[2 * MAX_SPEAKERS] = 0.9;
    preds[3 * MAX_SPEAKERS] = 0.9;
    let turns = d.binarize(&preds);
    assert_eq!(turns.len(), 1);
    let expect_start = 2.0 * FRAME_DURATION_SECS as f64 - 0.1;
    let expect_end = 4.0 * FRAME_DURATION_SECS as f64 + 0.2;
    // f32 thresholds widened to f64 inside binarize → allow 1e-6 slack.
    assert!((turns[0].time.start - expect_start).abs() < 1e-6);
    assert!((turns[0].time.end - expect_end).abs() < 1e-6);
}

#[test]
fn binarize_empty_preds_yields_no_turns() {
    let cfg = small_cfg();
    let d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    assert!(d.binarize(&[]).is_empty());
}

// --- streaming state machine ---------------------------------------------

#[test]
fn streaming_update_rejects_wrong_chunk_mel_len() {
    let cfg = small_cfg();
    let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    let err = d.streaming_update(&[0.0f32; 10], 1).unwrap_err();
    assert!(matches!(err, SortformerError::Shape(_)));
}

#[test]
fn streaming_update_missing_preds_output() {
    let cfg = small_cfg();
    let mut mock = ScriptedMock::new(&cfg);
    mock.output_names = vec!["bogus".into(), OUT_EMBS.into()];
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize / 2];
    let err = d.diarize_chunk(&audio).unwrap_err();
    match err {
        SortformerError::MissingOutput { name, available } => {
            assert_eq!(name, OUT_PREDS);
            assert!(available.iter().any(|n| n == "bogus"));
        }
        other => panic!("unexpected: {other}"),
    }
}

#[test]
fn streaming_update_missing_embs_output() {
    let cfg = small_cfg();
    let mut mock = ScriptedMock::new(&cfg);
    mock.output_names = vec![OUT_PREDS.into(), "bogus".into()];
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize / 2];
    let err = d.diarize_chunk(&audio).unwrap_err();
    assert!(matches!(
        err,
        SortformerError::MissingOutput { name: OUT_EMBS, .. }
    ));
}

#[test]
fn streaming_update_bad_preds_shape_is_shape_error() {
    let cfg = small_cfg();
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_last_dim = Some(3);
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize / 2];
    let err = d.diarize_chunk(&audio).unwrap_err();
    assert!(matches!(err, SortformerError::Shape(_)));
}

#[test]
fn streaming_update_bad_embs_shape_is_shape_error() {
    let cfg = small_cfg();
    let mut mock = ScriptedMock::new(&cfg);
    mock.embs_last_dim = Some(16);
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize / 2];
    let err = d.diarize_chunk(&audio).unwrap_err();
    assert!(matches!(err, SortformerError::Shape(_)));
}

#[test]
fn streaming_update_preds_frames_shorter_than_prefix_plus_keep() {
    let cfg = small_cfg();
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_chunk_frames = 2; // keep is 4 → 0 + 4 > 2 total frames
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize / 2];
    let err = d.diarize_chunk(&audio).unwrap_err();
    assert!(matches!(err, SortformerError::Shape(_)));
}

#[test]
fn streaming_update_i64_preds_is_inference_error() {
    let cfg = small_cfg();
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_i64 = true;
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize / 2];
    let err = d.diarize_chunk(&audio).unwrap_err();
    assert!(matches!(err, SortformerError::Inference(_)));
}

#[test]
fn streaming_update_works_with_positional_output_fallback() {
    let cfg = small_cfg();
    let mut mock = ScriptedMock::new(&cfg);
    mock.output_names = Vec::new(); // force positional mapping
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize / 2];
    let turns = d.diarize_chunk(&audio).unwrap();
    assert!(turns.is_empty(), "zero preds → no turns");
}

#[test]
fn run_failure_preserves_streaming_state_buffers() {
    let cfg = small_cfg();
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_fill = 0.9;
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize / 2];
    let _ = d.diarize_chunk(&audio).unwrap();
    let fifo_before = d.fifo_frames;
    let cache_before = d.spkcache_frames;
    assert!(fifo_before > 0);
    d.session.fail_on_run = Some(d.session.runs + 1);
    let err = d.diarize_chunk(&audio).unwrap_err();
    assert!(matches!(err, SortformerError::Inference(_)));
    // Buffers lent to the runtime are reclaimed even on error.
    assert_eq!(d.fifo.len(), fifo_before * EMB_DIM);
    assert_eq!(d.spkcache.len(), cache_before * EMB_DIM);
    assert_eq!(d.fifo_frames, fifo_before);
    assert_eq!(d.spkcache_frames, cache_before);
}

#[test]
fn diarize_with_active_speaker_produces_clipped_turns() {
    let cfg = SortformerConfig {
        // The scripted mock activates every sigmoid head; cap at one
        // speaker so only speaker 0 turns are expected.
        max_speakers: 1,
        post: passthrough_post(),
        ..small_cfg()
    };
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_fill = 0.9;
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize / 2]; // 0.5 s
    let turns = d.diarize(&audio).unwrap();
    assert!(!turns.is_empty());
    assert!(turns.iter().all(|t| t.speaker == SpeakerId(0)));
    assert!(turns.iter().all(|t| t.time.end <= 0.5 + 1e-9));
    assert!(turns.iter().all(|t| t.time.end > t.time.start));
}

#[test]
fn diarize_chunk_empty_input_yields_empty() {
    let cfg = small_cfg();
    let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    assert!(d.diarize_chunk(&[]).unwrap().is_empty());
}

#[test]
fn feed_buffers_until_window_then_emits_absolute_times() {
    let cfg = SortformerConfig {
        post: passthrough_post(),
        ..small_cfg()
    };
    let feed_samples = (cfg.chunk_len + cfg.right_context) * SUBSAMPLING * HOP_LENGTH;
    let chunk_dur = (cfg.chunk_len * SUBSAMPLING * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_fill = 0.9;
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    // Half a window: nothing can be emitted yet.
    let part = vec![0.01f32; feed_samples / 2];
    assert!(d.feed(&part).unwrap().is_empty());
    // Complete the window with some slack: exactly one window processed.
    let rest = vec![0.01f32; feed_samples];
    let turns = d.feed(&rest).unwrap();
    assert!(!turns.is_empty());
    assert!(turns.iter().all(|t| t.time.start >= 0.0));
    assert!(turns.iter().all(|t| t.time.end <= chunk_dur + 1e-9));
}

#[test]
fn flush_processes_remainder_with_elapsed_offset() {
    let cfg = SortformerConfig {
        post: passthrough_post(),
        ..small_cfg()
    };
    let feed_samples = (cfg.chunk_len + cfg.right_context) * SUBSAMPLING * HOP_LENGTH;
    let stride_samples = cfg.chunk_len * SUBSAMPLING * HOP_LENGTH;
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_fill = 0.9;
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    // One full window, then a partial one left in the buffer.
    let first = vec![0.01f32; feed_samples];
    assert!(!d.feed(&first).unwrap().is_empty());
    let tail = vec![0.01f32; feed_samples / 2];
    assert!(d.feed(&tail).unwrap().is_empty());
    let turns = d.flush().unwrap();
    assert!(!turns.is_empty());
    let offset = stride_samples as f64 / SAMPLE_RATE as f64;
    assert!(
        (turns[0].time.start - offset).abs() < 1e-6,
        "turn start {} should begin at elapsed offset {offset}",
        turns[0].time.start
    );
    // A second flush with an empty buffer is a no-op.
    assert!(d.flush().unwrap().is_empty());
}

#[test]
fn fifo_overflow_truncates_cache_when_per_spk_too_small() {
    // spkcache_len / MAX_SPEAKERS == 2 ≤ 3 → truncate branch.
    let cfg = SortformerConfig {
        post: passthrough_post(),
        ..small_cfg()
    };
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_fill = 0.9;
    let spkcache_len = cfg.spkcache_len;
    let fifo_len = cfg.fifo_len;
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize]; // 1 s → several chunks
    let _ = d.diarize_chunk(&audio).unwrap();
    let _ = d.diarize_chunk(&audio).unwrap();
    assert_eq!(d.spkcache_frames, spkcache_len);
    assert_eq!(d.spkcache.len(), spkcache_len * EMB_DIM);
    let preds = d.spkcache_preds.as_ref().expect("cache preds seeded");
    assert_eq!(preds.len(), spkcache_len * MAX_SPEAKERS);
    assert!(d.fifo_frames <= fifo_len);
}

#[test]
fn spkcache_overflow_compresses_cache_with_scoring() {
    // spkcache_len / MAX_SPEAKERS == 5 > 3 → full scoring/compression path.
    let cfg = SortformerConfig {
        chunk_len: 4,
        fifo_len: 4,
        spkcache_len: 20,
        right_context: 1,
        post: passthrough_post(),
        max_speakers: MAX_SPEAKERS,
    };
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_fill = 0.9;
    let spkcache_len = cfg.spkcache_len;
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize];
    let _ = d.diarize_chunk(&audio).unwrap();
    let _ = d.diarize_chunk(&audio).unwrap();
    assert_eq!(d.spkcache_frames, spkcache_len);
    assert_eq!(d.spkcache.len(), spkcache_len * EMB_DIM);
    let preds = d.spkcache_preds.as_ref().expect("cache preds present");
    assert_eq!(preds.len(), spkcache_len * MAX_SPEAKERS);
}

#[test]
fn compress_spkcache_without_preds_is_noop() {
    let cfg = small_cfg();
    let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    d.spkcache = vec![1.0f32; 4 * EMB_DIM];
    d.spkcache_frames = 4;
    d.spkcache_preds = None;
    d.compress_spkcache();
    assert_eq!(d.spkcache_frames, 4);
    assert_eq!(d.spkcache.len(), 4 * EMB_DIM);
}

#[test]
fn gather_spkcache_copies_frames_and_silence_placeholders() {
    let cfg = small_cfg();
    let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    // Two cache frames with distinct constant embeddings.
    d.spkcache = vec![1.0f32; EMB_DIM]
        .into_iter()
        .chain(vec![2.0f32; EMB_DIM])
        .collect();
    d.spkcache_frames = 2;
    d.spkcache_preds = Some(vec![0.5f32; 2 * MAX_SPEAKERS]);
    d.mean_sil_emb = vec![9.0f32; EMB_DIM];
    // out[0] = cache frame 1, out[1] = disabled → mean sil, out[2] = out-of-range idx.
    let indices = [1usize, 0, 5];
    let disabled = [false, true, false];
    let (embs, preds) = d.gather_spkcache(&indices, &disabled, 3);
    assert!(embs[..EMB_DIM].iter().all(|&v| v == 2.0));
    assert!(embs[EMB_DIM..2 * EMB_DIM].iter().all(|&v| v == 9.0));
    assert!(embs[2 * EMB_DIM..].iter().all(|&v| v == 0.0));
    assert!(preds[..MAX_SPEAKERS].iter().all(|&v| v == 0.5));
    assert!(
        preds[MAX_SPEAKERS..2 * MAX_SPEAKERS]
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn gather_spkcache_truncates_to_out_len() {
    let cfg = small_cfg();
    let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    d.spkcache = vec![1.0f32; 3 * EMB_DIM];
    d.spkcache_frames = 3;
    d.spkcache_preds = None; // exercises the empty cache_preds arm
    let indices = [0usize, 1, 2];
    let disabled = [false, false, false];
    let (embs, preds) = d.gather_spkcache(&indices, &disabled, 2);
    assert_eq!(embs.len(), 2 * EMB_DIM);
    assert_eq!(preds.len(), 2 * MAX_SPEAKERS);
}

#[test]
fn update_silence_profile_averages_only_silent_frames() {
    let cfg = small_cfg();
    let mut d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    let embs = vec![1.0f32; 2 * EMB_DIM];
    let mut preds = vec![0.0f32; 2 * MAX_SPEAKERS];
    preds[MAX_SPEAKERS] = 1.0; // frame 1 is speech
    d.update_silence_profile(&embs, &preds, 2);
    assert_eq!(d.n_sil_frames, 1);
    assert!(d.mean_sil_emb.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    // A second silent frame with different embedding updates the running mean.
    let embs2 = vec![3.0f32; EMB_DIM];
    let preds2 = vec![0.0f32; MAX_SPEAKERS];
    d.update_silence_profile(&embs2, &preds2, 1);
    assert_eq!(d.n_sil_frames, 2);
    assert!(d.mean_sil_emb.iter().all(|&v| (v - 2.0).abs() < 1e-6));
}

// --- accessors / lifecycle -----------------------------------------------

#[test]
fn config_and_latency_accessors_reflect_config() {
    let cfg = small_cfg();
    let expected_latency = cfg.latency_secs();
    let d = SortformerDiarizer::from_runtime(ZeroMock::new(cfg.chunk_len), cfg);
    assert_eq!(d.config().chunk_len, 4);
    assert!((d.latency_secs() - expected_latency).abs() < 1e-6);
}

#[test]
fn reset_clears_all_streaming_state() {
    let cfg = SortformerConfig {
        post: passthrough_post(),
        ..small_cfg()
    };
    let mut mock = ScriptedMock::new(&cfg);
    mock.preds_fill = 0.9;
    let mut d = SortformerDiarizer::from_runtime(mock, cfg);
    let audio = vec![0.01f32; SAMPLE_RATE as usize];
    let _ = d.diarize_chunk(&audio).unwrap();
    d.audio_buffer.extend_from_slice(&[0.1; 100]);
    d.elapsed_samples = 42;
    d.reset();
    assert!(d.spkcache.is_empty());
    assert_eq!(d.spkcache_frames, 0);
    assert!(d.spkcache_preds.is_none());
    assert!(d.fifo.is_empty());
    assert_eq!(d.fifo_frames, 0);
    assert!(d.fifo_preds.is_empty());
    assert!(d.mean_sil_emb.iter().all(|&v| v == 0.0));
    assert_eq!(d.n_sil_frames, 0);
    assert!(d.audio_buffer.is_empty());
    assert_eq!(d.elapsed_samples, 0);
}

#[test]
fn from_path_with_config_validates_before_any_io() {
    let cfg = SortformerConfig {
        max_speakers: 0,
        ..Default::default()
    };
    let err = SortformerDiarizer::from_path_with_config("/nonexistent/model.onnx", cfg)
        .err()
        .expect("must reject invalid config");
    assert!(matches!(
        err,
        SortformerError::MaxSpeakersExceeded { requested: 0 }
    ));
}

#[test]
fn from_path_missing_model_file_is_load_error() {
    let err = SortformerDiarizer::from_path("/nonexistent/sortformer.onnx")
        .err()
        .expect("missing file must fail");
    assert!(matches!(err, SortformerError::Load(_)));
}
