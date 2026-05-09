//! Performance regression test for the legacy v0.5 pipeline.
//!
//! Tracks RTF (Real-Time Factor) and peak RSS memory over a fixed subset of
//! VoxConverse-test files. Run with:
//!
//! ```bash
//! cargo test --test perf_regression_test --features "onnx download" -- --ignored
//! ```

#![cfg(all(feature = "onnx", feature = "download"))]

use polyvoice::models::ModelRegistry;
use polyvoice::pipeline::Pipeline;
use polyvoice::types::{DiarizationConfig, Profile};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, SileroVad};
use std::path::Path;
use std::time::Instant;

/// Return the first 5 WAV files (alphabetically) from VoxConverse-test.
fn fixed_voxconverse_subset() -> Option<Vec<std::path::PathBuf>> {
    let dir = Path::new("data/voxconverse-test/audio");
    if !dir.is_dir() {
        return None;
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "wav"))
        .map(|e| e.path())
        .collect();
    entries.sort();
    let subset: Vec<_> = entries.into_iter().take(5).collect();
    if subset.is_empty() {
        None
    } else {
        Some(subset)
    }
}

/// Read peak resident set size in megabytes.
///
/// On Linux this reads VmHWM from /proc/self/status.
/// On macOS this always returns 0.0 (no dependency-free accurate API).
fn peak_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(kb_str) = line
                    .strip_prefix("VmHWM:")
                    .and_then(|s| s.split_whitespace().next())
                {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        // No dependency-free way to get accurate peak RSS on macOS.
        // Returning 0.0 skips the memory assertion on this platform.
    }
    0.0
}

#[derive(Debug, serde::Serialize)]
struct FileResult {
    file: String,
    audio_duration_secs: f64,
    elapsed_secs: f64,
    rtf: f64,
    peak_rss_mb: f64,
}

#[derive(Debug, serde::Serialize)]
struct PerfReport {
    files: Vec<FileResult>,
    avg_rtf: f64,
    max_rss_mb: f64,
}

#[ignore = "requires ONNX models + dataset"]
#[test]
fn perf_regression_legacy_pipeline() {
    let wav_paths = match fixed_voxconverse_subset() {
        Some(p) => p,
        None => panic!(
            "data/voxconverse-test/audio/ is empty — run scripts/download-voxconverse-test.sh first"
        ),
    };

    let registry = ModelRegistry::default()
        .expect("default ModelRegistry should resolve a writable cache dir");
    let models = registry.ensure_for_profile(Profile::Balanced).expect(
        "Balanced profile models should be available — \
             run `polyvoice download-models --profile balanced` first",
    );

    let embedding_dim = Profile::Balanced.embedding_dim();
    let extractor =
        FbankOnnxExtractor::new(&models.embedder_path, embedding_dim, 1).expect("load embedder");
    let mut vad = SileroVad::new(&models.segmenter_path, 512).expect("load vad");

    let config = DiarizationConfig::default();
    let vad_config = VadConfig::default();
    let pipeline = Pipeline::new(config, vad_config);

    let mut file_results = Vec::with_capacity(wav_paths.len());

    for wav_path in &wav_paths {
        let stem = wav_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        let (samples, sr_hz) = read_wav(wav_path).expect("WAV read failure");
        assert_eq!(sr_hz, 16000, "only 16 kHz WAVs supported");

        let audio_duration_secs = samples.len() as f64 / sr_hz as f64;

        let start = Instant::now();
        let _result = pipeline
            .run(&samples, &extractor, &mut vad)
            .expect("pipeline.run should succeed");
        let elapsed_secs = start.elapsed().as_secs_f64();

        let rtf = elapsed_secs / audio_duration_secs;
        let peak_rss = peak_rss_mb();

        file_results.push(FileResult {
            file: stem.to_string(),
            audio_duration_secs,
            elapsed_secs,
            rtf,
            peak_rss_mb: peak_rss,
        });
    }

    let avg_rtf = file_results.iter().map(|r| r.rtf).sum::<f64>() / file_results.len() as f64;
    let max_rss_mb = file_results
        .iter()
        .map(|r| r.peak_rss_mb)
        .fold(0.0, f64::max);

    let report = PerfReport {
        files: file_results,
        avg_rtf,
        max_rss_mb,
    };

    // Print JSON-friendly output for CI parsing.
    println!("{}", serde_json::to_string_pretty(&report).unwrap());

    // Thresholds.
    assert!(
        avg_rtf < 0.25,
        "average RTF too high: {:.3} (threshold 0.25)",
        avg_rtf
    );

    // Only assert memory on Linux where we have an accurate reading.
    #[cfg(target_os = "linux")]
    assert!(
        max_rss_mb < 500.0,
        "peak RSS too high: {:.1} MB (threshold 500 MB)",
        max_rss_mb
    );
}
