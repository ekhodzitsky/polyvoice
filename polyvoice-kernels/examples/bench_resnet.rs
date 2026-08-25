//! Release microbench: GEMM shape of a ResNet layer-1 conv, then a full embed.
//!
//!   cargo run -p polyvoice-kernels --release --example bench_resnet

use polyvoice_kernels::{ResNet34, gemm_bias_row};
use std::time::Instant;

fn time_gemm(label: &str, m: usize, n: usize, k: usize, iters: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.0005 - 0.2).collect();
    let bias: Vec<f32> = (0..m).map(|i| i as f32 * 0.01).collect();
    let mut c = vec![0.0f32; m * n];
    gemm_bias_row(&a, &b, &bias, &mut c, m, n, k);
    let checksum = c.iter().sum::<f32>();
    let start = Instant::now();
    for _ in 0..iters {
        gemm_bias_row(&a, &b, &bias, &mut c, m, n, k);
    }
    let secs = start.elapsed().as_secs_f64() / iters as f64;
    let gflops = (2.0 * m as f64 * n as f64 * k as f64) / secs / 1e9;
    println!("{label}: {m}x{n}x{k}  {secs:.4}s  {gflops:.1} GFLOP/s  checksum={checksum:.3}");
}

fn main() {
    // Layer-1 identity 3×3: oc=32, k_col=32*9=288, spatial≈80×200 for ~2 s.
    time_gemm("layer1-ish (naive local)", 32, 16_000, 288, 4);
    time_gemm("layer3-ish (naive local)", 128, 2_000, 128 * 9, 4);

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/int8/resnet34_int8.onnx");
    if !path.is_file() {
        eprintln!("skip resnet: {path:?} missing");
        return;
    }
    let net = ResNet34::from_onnx_path(&path).expect("load");
    let n_frames = 200;
    let mut frames = vec![0.0f32; n_frames * 80];
    for (i, v) in frames.iter_mut().enumerate() {
        *v = ((i % 17) as f32) * 0.01 - 0.08;
    }
    // warmup
    let _ = net.embed_fbank(&frames, n_frames).expect("warmup");
    let iters = 8;
    let start = Instant::now();
    let mut last = Vec::new();
    for _ in 0..iters {
        last = net.embed_fbank(&frames, n_frames).expect("embed");
    }
    let secs = start.elapsed().as_secs_f64() / iters as f64;
    let max = last.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!("resnet 200 frames: {secs:.4}s/embed  |emb|_inf={max:.4}");
}
