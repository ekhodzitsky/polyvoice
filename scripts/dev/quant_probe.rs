//! One-off: speed + embedding-drift probe for quantized ResNet34 variants.
//! Reports per-call latency and per-window cosine(fp32, quantized) stats —
//! the direction-preservation metric that predicts clustering survival.

use polyvoice::FbankOnnxExtractor;
use polyvoice::embedder::Embedder;
use polyvoice::onnx::ExecutionProvider;
use polyvoice::utils::cosine_similarity;
use polyvoice::wav::read_wav;
use std::path::Path;
use std::time::Instant;

fn stats(name: &str, mut v: Vec<f32>) {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let mean = v.iter().sum::<f32>() / n as f32;
    println!(
        "  {name}: min={:.5} p5={:.5} p50={:.5} mean={:.5}",
        v[0],
        v[n / 20],
        v[n / 2],
        mean
    );
}

fn main() -> anyhow::Result<()> {
    let (samples, sr) = read_wav(Path::new("data/voxconverse-test/audio/aepyx.wav"))?;
    assert_eq!(sr, 16000);

    let models: Vec<(&str, &str)> = vec![
        ("fp32", "models/wespeaker_resnet34.onnx"),
        ("int8-qdq", "models/int8/resnet34_int8.onnx"),
        ("w8-qdq", "/tmp/resnet34_w8_qdq.onnx"),
        ("static-qdq", "/tmp/resnet34_static_qdq.onnx"),
    ];

    let mut ref_embs: Vec<Vec<f32>> = Vec::new();
    for (idx, (name, path)) in models.iter().enumerate() {
        let ext = FbankOnnxExtractor::new(Path::new(path), 256, 1, ExecutionProvider::Cpu)?;
        // latency
        let window = &samples[..24000];
        for _ in 0..3 {
            let _ = ext.embed(window)?;
        }
        let n = 50;
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = ext.embed(window)?;
        }
        let dt = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("{name:>9}: {dt:7.2} ms/call");

        // embeddings over the file (1.5s window, 1.5s hop for a cheap sweep)
        let mut embs = Vec::new();
        let mut off = 0usize;
        while off + 24000 <= samples.len() {
            embs.push(ext.embed(&samples[off..off + 24000])?);
            off += 24000;
        }
        if idx == 0 {
            ref_embs = embs;
        } else {
            let cos: Vec<f32> = ref_embs
                .iter()
                .zip(&embs)
                .map(|(a, b)| cosine_similarity(a, b))
                .collect();
            println!("drift cos(fp32, {name}) over {} windows:", cos.len());
            stats(&format!("fp32-vs-{name}"), cos);
        }
    }
    Ok(())
}
