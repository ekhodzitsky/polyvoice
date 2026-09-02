//! Shared helpers for the integration-test suite.
//!
//! Single source of truth for the pieces that used to be copy-pasted across
//! the DER test family: reference-turn loading from RTTM, skip-if-missing
//! data gates, the typed view over `tests/der_baseline.json`, and the DER
//! gates themselves. Every integration-test binary that declares
//! `mod common;` compiles this module in; helpers a given binary does not
//! call stay unused, hence the module-level dead-code allowance.

#![allow(dead_code)]

use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{SpeakerId, SpeakerTurn, TimeRange};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Write a mono PCM16 WAVE file from `[-1, 1]` samples.
pub fn write_pcm16_mono(path: &Path, sample_rate: u32, samples: &[f32]) {
    let pcm = ryf::f32_to_s16le(samples);
    ryf::write_s16(path, &pcm, sample_rate).expect("write wav");
}

/// Fixed 10-file VoxConverse-test subset used by the fast DER gates.
pub const VOXCONVERSE_SUBSET_10: &[&str] = &[
    "aepyx", "aggyz", "aiqwk", "aorju", "auzru", "bgvvt", "bidnq", "bjruf", "bmsyn", "bpzsc",
];

/// Load reference speaker turns for `stem` from an RTTM file.
///
/// Single lookup semantics for every DER test: parse, group by file id, then
/// look up `stem` — with the AMI convention that audio named
/// `EN2002a.Mix-Headset.wav` is keyed as `EN2002a` in the RTTM.
pub fn load_ref_turns(rttm_path: &Path, stem: &str) -> Vec<SpeakerTurn> {
    let raw = parse_rttm_file(rttm_path).expect("parse rttm");
    let grouped = group_by_file(&raw);
    // AMI files use basename like EN2002a.Mix-Headset.wav but RTTM key is EN2002a
    let rttm_key = if stem.contains(".Mix-Headset") {
        stem.trim_end_matches(".Mix-Headset")
    } else {
        stem
    };
    let segs: Vec<_> = grouped
        .get(rttm_key)
        .map(|v| v.iter().map(|s| (*s).clone()).collect())
        .unwrap_or_default();
    let (turns, _map) = to_speaker_turns(&segs);
    turns
}

/// Release-gate signal: `true` when `POLYVOICE_REQUIRE_DATA=1` is set.
fn require_data() -> bool {
    std::env::var("POLYVOICE_REQUIRE_DATA")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Data gate for tests that need downloaded fixtures: returns `true` when
/// `path` exists and the caller may proceed.
///
/// When the data is missing, behavior depends on `POLYVOICE_REQUIRE_DATA`:
/// - `1` (the release gate exports it): hard failure — a partial
///   cache/download miss can never green-light a release without actually
///   running the gate;
/// - unset (local dev): note the skip on stderr and return `false`, so the
///   caller can skip the test (or the file) with soft-skip ergonomics.
pub fn require_wav(path: &Path) -> bool {
    if path.exists() {
        return true;
    }
    assert!(
        !require_data(),
        "release gate requires test data but it is missing: {}",
        path.display()
    );
    eprintln!(
        "{} not found — skipping (set POLYVOICE_REQUIRE_DATA=1 to require it)",
        path.display()
    );
    false
}

/// Typed view over `tests/der_baseline.json` — one struct for every test that
/// reads the file. `#[serde(default)]` on every field keeps the view total:
/// entries may carry null or absent values (placeholders, retired
/// experiments), and each test only touches the fields it gates on. All DER
/// values are percentages as recorded in the JSON; the gates convert to
/// 0..1 ratios.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct DerBaseline {
    pub schema: String,
    #[serde(rename = "crate_version")]
    pub crate_version: Option<String>,
    pub git_sha: String,
    pub command_line: String,
    pub voxconverse_test: DatasetBaseline,
    pub ami_test: DatasetBaseline,
    /// Full-split Linux CPU product gate (EP=cpu, powerset N=8).
    pub voxconverse_test_linux_cpu: DatasetBaseline,
    /// Full-split Linux CPU product gate (EP=cpu, powerset N=8).
    pub ami_test_linux_cpu: DatasetBaseline,
    /// Unmeasured fail-closed ceiling for Linux native Vox until filled.
    pub voxconverse_test_linux_cpu_native: DatasetBaseline,
    /// Unmeasured fail-closed ceiling for Linux native AMI until filled.
    pub ami_test_linux_cpu_native: DatasetBaseline,
    pub voxconverse_dev: DatasetBaseline,
    pub voxconverse_test_10files: DatasetBaseline,
    pub e2e_smoke: DatasetBaseline,
    pub ami_test_single: DatasetBaseline,
    pub v2_e2e_smoke: DatasetBaseline,
    pub hybrid_e2e_smoke: DatasetBaseline,
    pub hybrid_voxconverse_3file: DatasetBaseline,
    pub hybrid_voxconverse_10file: DatasetBaseline,
    /// v2-family AMI entry; hosts the overlap-excluded long-form floor.
    pub hybrid_ami_test_single: DatasetBaseline,
    pub voxconverse_test_legacy: DatasetBaseline,
    pub ami_test_legacy: DatasetBaseline,
    /// Cross-corpus gate: NOTSOFAR-1 dev-set-1, single far-field channel.
    pub notsofar_dev: DatasetBaseline,
    /// Fixed 3-meeting NOTSOFAR-1 subset used by the regression test.
    pub notsofar_dev_3file: DatasetBaseline,
}

/// One dataset entry in `tests/der_baseline.json`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct DatasetBaseline {
    pub files: Option<usize>,
    pub profile: String,
    pub pipeline: Option<String>,
    pub clusterer: Option<String>,
    /// `cli-native` / `cli-ort` when recorded.
    pub engine: Option<String>,
    /// Recorded EP when the baseline was measured (`cpu`, `CoreMl`, …).
    pub execution_provider: Option<String>,
    /// Powerset ONNX micro-batch size used for the measurement (e.g. 8).
    pub powerset_batch: Option<usize>,
    pub threshold: Option<f64>,
    pub min_cluster_size: Option<usize>,
    #[serde(rename = "der_collar_0_25")]
    pub der_collar_0_25: Option<f64>,
    /// No-collar (collar=0) DER baseline — the headline like-for-like metric,
    /// micro-averaged (frame-weighted) on multi-file sets. `None` (JSON null
    /// or absent) = not yet measured → the no-collar gate stays inactive for
    /// that dataset.
    pub der_no_collar: Option<f64>,
    pub der_collar_macro: Option<f64>,
    pub der_collar_micro: Option<f64>,
    pub der_no_collar_macro: Option<f64>,
    pub der_no_collar_micro: Option<f64>,
    /// Overlap-excluded (single-speaker-region) DER floor for long-form audio.
    /// `None` = not yet measured → the long-form floor gate stays inactive.
    pub der_single_speaker: Option<f64>,
    pub der_single_speaker_tolerance: Option<f64>,
    #[serde(rename = "_der_single_speaker_status")]
    pub der_single_speaker_status: Option<String>,
    /// Miss rate (%). Collar for this decomposition is `miss_fa_conf_collar_secs`
    /// when set (Linux CPU gate uses 0.25); otherwise historically collar 0.
    pub miss: Option<f64>,
    pub false_alarm: Option<f64>,
    pub confusion: Option<f64>,
    /// Collar seconds used for `miss` / `false_alarm` / `confusion` when known.
    pub miss_fa_conf_collar_secs: Option<f64>,
    pub rt_factor_avg: Option<f64>,
    pub speaker_count: Option<SpeakerCountBaseline>,
    pub tolerance: Option<f64>,
    pub model_versions: Option<ModelVersions>,
    #[serde(rename = "_status")]
    pub status: String,
    /// Provenance of a filled row. `None` (JSON `null` or absent) means the
    /// numbers are a copied ceiling / placeholder, not a measured artifact.
    #[serde(rename = "_filled_by")]
    pub filled_by: Option<String>,
}

/// Speaker-count accuracy breakdown attached to the full-corpus entries.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct SpeakerCountBaseline {
    pub exact: usize,
    pub plus_minus_1: usize,
    pub off_by_2_or_more: usize,
}

/// Model trio a baseline was measured with.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ModelVersions {
    pub segmenter: String,
    pub embedder: String,
    pub clusterer: Option<String>,
}

/// Path to the committed DER baseline shared by the gate tests.
pub fn der_baseline_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/der_baseline.json")
}

/// Load and parse a `der_baseline.json` file.
pub fn load_baseline(path: &Path) -> DerBaseline {
    let raw = std::fs::read_to_string(path).expect("read der_baseline.json");
    serde_json::from_str(&raw).expect("parse der_baseline.json")
}

/// Checked-in VBx PLDA fixtures used to exercise the default clusterer offline.
pub fn vbx_plda_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/vbx-plda")
}

/// Load the Balanced-profile ONNX embedder shared by the model-gated tests,
/// downloading models into the registry cache on first use.
#[cfg(all(feature = "onnx", feature = "download"))]
pub fn balanced_onnx_extractor() -> polyvoice::FbankOnnxExtractor {
    let registry = polyvoice::models::ModelRegistry::default().expect("registry");
    let models = registry
        .ensure_for_profile(polyvoice::types::Profile::Balanced)
        .expect("models");
    polyvoice::FbankOnnxExtractor::new(
        &models.embedder_path,
        polyvoice::types::Profile::Balanced.embedding_dim(),
        1,
        polyvoice::onnx::ExecutionProvider::Cpu,
    )
    .expect("embedder")
}

/// Gate a measured collar-0.25 DER (a 0..1 ratio) against the dataset baseline
/// (`der_collar_0_25`/`tolerance` are percent values in der_baseline.json).
pub fn gate_against_baseline(dataset: &str, measured: f64, baseline: &DatasetBaseline) {
    let expected = baseline
        .der_collar_0_25
        .unwrap_or_else(|| panic!("{dataset}: der_collar_0_25 missing in der_baseline.json"))
        / 100.0;
    let tolerance = baseline
        .tolerance
        .unwrap_or_else(|| panic!("{dataset}: tolerance missing in der_baseline.json"))
        / 100.0;
    assert!(
        measured <= expected + tolerance,
        "DER regression: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
        (expected + tolerance) * 100.0,
        measured * 100.0,
        expected * 100.0,
        tolerance * 100.0,
    );
}

/// Gate a measured no-collar DER (a 0..1 ratio) against the dataset baseline.
/// Inactive (prints the value to record) while `der_no_collar` is null in
/// der_baseline.json.
pub fn assert_no_collar(dataset: &str, measured: f64, baseline: &DatasetBaseline) {
    match baseline.der_no_collar {
        Some(expected_pct) => {
            let tolerance_pct = baseline
                .tolerance
                .unwrap_or_else(|| panic!("{dataset}: tolerance missing in der_baseline.json"));
            let bound = (expected_pct + tolerance_pct) / 100.0;
            assert!(
                measured <= bound,
                "no-collar DER regression on {dataset}: expected <= {:.2}%, got {:.2}% (baseline {:.2}% + tolerance {:.2}%)",
                bound * 100.0,
                measured * 100.0,
                expected_pct,
                tolerance_pct,
            );
        }
        None => println!(
            "{dataset}: no-collar baseline not yet measured — record {:.2}% as der_no_collar in tests/der_baseline.json to activate the gate",
            measured * 100.0
        ),
    }
}

/// AMI long-form regression gate for pipeline v2 on EN2002a.
///
/// Total DER is deliberately NOT gated: AMI EN2002a is ~79% overlapping
/// speech, and with single-speaker-per-frame output the miss term alone holds
/// DER near 88% whether diarization is healthy or collapsed — a DER ceiling
/// cannot tell the two apart. Gate instead on the signals that actually move
/// when diarization regresses: the speaker count must not collapse, clustering
/// confusion must stay low (post-fix it is ~11%), and the overlap-excluded
/// (single-speaker-region) DER must hold its recorded floor — unlike total
/// DER it DOES discriminate healthy vs collapsed diarization on high-overlap
/// audio. The floor activates only once a baseline is measured and recorded
/// in tests/der_baseline.json (`hybrid_ami_test_single.der_single_speaker`);
/// until then the printed value is the measurement to record.
pub fn gate_ami_longform(
    num_speakers: usize,
    confusion: f64,
    single_speaker_der: f64,
    baseline: &DatasetBaseline,
) {
    // The NaN-embedding collapse manifested as num_speakers=1 — every segment
    // merged into a single cluster.
    assert!(
        num_speakers >= 2,
        "pipeline_v2 collapsed to {num_speakers} speaker(s) on EN2002a (NaN-embedding regression?)"
    );
    assert!(
        confusion < 0.25,
        "pipeline_v2 clustering regressed on EN2002a: confusion={:.1}% exceeds 25%",
        confusion * 100.0
    );
    match (
        baseline.der_single_speaker,
        baseline.der_single_speaker_tolerance,
    ) {
        (Some(floor), Some(tol)) => {
            let ceiling = (floor + tol) / 100.0;
            assert!(
                single_speaker_der <= ceiling,
                "long-form floor regressed: overlap-excluded DER={:.2}% exceeds {:.2}% (baseline {:.2}% + tol {:.2}%)",
                single_speaker_der * 100.0,
                ceiling * 100.0,
                floor,
                tol,
            );
        }
        _ => println!(
            "  overlap-excluded DER baseline not yet measured — record {:.2}% in tests/der_baseline.json to activate the long-form floor",
            single_speaker_der * 100.0
        ),
    }
}

// ---------------------------------------------------------------------------
// Synthetic embeddings and turns
// ---------------------------------------------------------------------------

/// Unit vector along `axis` in `dim`-dimensional space.
pub fn unit_vec(dim: usize, axis: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[axis] = 1.0;
    v
}

/// A `SpeakerTurn` over `start..end` with no text, marked stable.
pub fn turn(start: f64, end: f64, speaker: u32) -> SpeakerTurn {
    SpeakerTurn {
        time: TimeRange { start, end },
        speaker: SpeakerId(speaker),
        text: None,
        stable: true,
    }
}

/// Alternating unit axes — deterministic two-speaker clustering without a model.
pub struct AxisEmbedder {
    dim: usize,
    flip: std::sync::atomic::AtomicUsize,
}

impl AxisEmbedder {
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 2);
        Self {
            dim,
            flip: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl polyvoice::Embedder for AxisEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, audio: &[f32]) -> Result<Vec<f32>, polyvoice::EmbedderError> {
        if audio.is_empty() {
            return Err(polyvoice::EmbedderError::AudioTooShort {
                actual_secs: 0.0,
                min_secs: 0.01,
            });
        }
        let n = self.flip.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut v = vec![0.0f32; self.dim];
        if n.is_multiple_of(2) {
            v[0] = 1.0;
        } else {
            v[1] = 1.0;
        }
        Ok(v)
    }
}

/// `secs` seconds of a 300 Hz sine at sample rate `sr`, amplitude 0.3.
pub fn speech_pcm(secs: f32, sr: u32) -> Vec<f32> {
    let n = (secs * sr as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CLI binaries
// ---------------------------------------------------------------------------

/// `polyvoice` binary under test, with backtraces off for stable output.
pub fn polyvoice_cmd() -> assert_cmd::Command {
    let mut cmd = assert_cmd::Command::cargo_bin("polyvoice").expect("polyvoice binary");
    cmd.env("RUST_BACKTRACE", "0");
    cmd
}
