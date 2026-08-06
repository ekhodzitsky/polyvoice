//! One-off DER gate for a candidate embedder model on the legacy pipeline.
//! Usage: quant_der_check <model-path> [tag]

use polyvoice::der::compute_der;
use polyvoice::onnx::ExecutionProvider;
use polyvoice::pipeline::LegacyPipeline;
use polyvoice::rttm::{group_by_file, parse_rttm_file, to_speaker_turns};
use polyvoice::types::{DiarizationConfig, SpeakerTurn};
use polyvoice::vad::VadConfig;
use polyvoice::wav::read_wav;
use polyvoice::{FbankOnnxExtractor, SileroVad};
use std::path::Path;

const SUBSET_10: &[&str] = &[
    "aepyx", "aggyz", "aiqwk", "aorju", "auzru", "bgvvt", "bidnq", "bjruf", "bmsyn", "bpzsc",
];

fn load_ref(rttm_path: &Path, stem: &str) -> Vec<SpeakerTurn> {
    let segs = parse_rttm_file(rttm_path).expect("parse rttm");
    let grouped = group_by_file(&segs);
    let v: Vec<_> = grouped
        .get(stem)
        .or_else(|| stem.split('.').next().and_then(|s| grouped.get(s)))
        .map(|v| v.iter().map(|s| (*s).clone()).collect())
        .unwrap_or_default();
    let (turns, _map) = to_speaker_turns(&v);
    turns
}

fn run_one(
    ext: &FbankOnnxExtractor,
    vad: &mut SileroVad,
    wav: &Path,
    ref_turns: &[SpeakerTurn],
) -> (f64, f64, usize) {
    let (samples, sr) = read_wav(wav).expect("read wav");
    assert_eq!(sr, 16000);
    let pipeline = LegacyPipeline::new(DiarizationConfig::default(), VadConfig::default());
    let result = pipeline.run(&samples, ext, vad).expect("pipeline");
    let d25 = compute_der(ref_turns, &result.turns, 0.25);
    let d0 = compute_der(ref_turns, &result.turns, 0.0);
    (d25.der * 100.0, d0.der * 100.0, result.num_speakers)
}

fn main() -> anyhow::Result<()> {
    let model = std::env::args()
        .nth(1)
        .expect("usage: quant_der_check <model-path> [tag]");
    let tag = std::env::args().nth(2).unwrap_or_else(|| "model".into());
    let ext = FbankOnnxExtractor::new(Path::new(&model), 256, 1, ExecutionProvider::Cpu)?;
    let vad_path = Path::new("models/silero_vad.onnx");

    {
        let stem = "EN2002a.Mix-Headset";
        let wav = Path::new("data/ami-test-single/audio/EN2002a.Mix-Headset.wav");
        let rttm = Path::new("data/ami-test-single/rttm/EN2002a.Mix-Headset.rttm");
        let ref_turns = load_ref(rttm, stem);
        let mut vad = SileroVad::new(vad_path, 512)?;
        let t0 = std::time::Instant::now();
        let (d25, d0, n) = run_one(&ext, &mut vad, wav, &ref_turns);
        println!(
            "EN2002a [{tag}]: DER@0.25 = {d25:.2}%  DER@0 = {d0:.2}%  speakers = {n}  ({:.0}s)  [fp32 baseline: 34.62 / 42.90]",
            t0.elapsed().as_secs_f64()
        );
    }

    let mut sum = 0.0;
    for stem in SUBSET_10 {
        let wav = format!("data/voxconverse-test/audio/{stem}.wav");
        let rttm = format!("data/voxconverse-test/rttm/{stem}.rttm");
        let ref_turns = load_ref(Path::new(&rttm), stem);
        let mut vad = SileroVad::new(vad_path, 512)?;
        let (d25, _d0, n) = run_one(&ext, &mut vad, Path::new(&wav), &ref_turns);
        sum += d25;
        println!("{stem} [{tag}]: DER@0.25 = {d25:.2}%  speakers = {n}");
    }
    println!(
        "vox-10 macro [{tag}]: {:.2}%  [fp32: ~15.82 measured on the same subset]",
        sum / SUBSET_10.len() as f64
    );
    Ok(())
}
