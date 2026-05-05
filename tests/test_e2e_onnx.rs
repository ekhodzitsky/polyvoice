//! End-to-end tests with real ONNX models.
//! Run: POLYVOICE_MODEL_DIR=models cargo test --test test_e2e_onnx --features onnx

#[cfg(feature = "onnx")]
mod e2e {
    use polyvoice::{
        DiarizationConfig, EcapaTdnnExtractor, Pipeline, VadConfig,
        silero_vad::SileroVad,
        vad::VoiceActivityDetector,
    };
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        std::env::var("POLYVOICE_MODEL_DIR").ok().map(PathBuf::from)
    }

    fn sine_wave(freq: f32, duration_secs: f32, amplitude: f32) -> Vec<f32> {
        let sample_rate = 16000;
        let n = (duration_secs * sample_rate as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * amplitude
            })
            .collect()
    }

    #[test]
    fn test_silero_vad_detects_speech() {
        let dir = match model_dir() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: POLYVOICE_MODEL_DIR not set");
                return;
            }
        };

        let vad_path = dir.join("silero_vad.onnx");
        let mut vad = SileroVad::new(&vad_path, 512).expect("load Silero VAD");

        let loud = sine_wave(300.0, 1.0, 0.8);
        let probs = vad.process(&loud[..512 * (loud.len() / 512)]).unwrap();
        assert!(probs.iter().all(|&p| (0.0..=1.0).contains(&p)));
        eprintln!("Silero VAD max prob on sine: {:.3}",
            probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }

    #[test]
    fn test_wespeaker_embedding_shape() {
        let dir = match model_dir() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: POLYVOICE_MODEL_DIR not set");
                return;
            }
        };

        let model_path = dir.join("wespeaker_resnet34.onnx");
        let extractor = EcapaTdnnExtractor::new(&model_path, 256, 2)
            .expect("load WeSpeaker");

        let config = DiarizationConfig::default();
        let samples = sine_wave(440.0, 2.0, 0.5);

        let embedding = polyvoice::EmbeddingExtractor::extract(
            &extractor,
            &samples[..config.window_samples()],
            &config,
        )
        .expect("extract embedding");

        assert_eq!(embedding.len(), 256);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_full_pipeline_onnx() {
        let dir = match model_dir() {
            Some(d) => d,
            None => {
                eprintln!("Skipping: POLYVOICE_MODEL_DIR not set");
                return;
            }
        };

        let model_path = dir.join("wespeaker_resnet34.onnx");
        let vad_path = dir.join("silero_vad.onnx");

        let extractor = EcapaTdnnExtractor::new(&model_path, 256, 4)
            .expect("load WeSpeaker");
        let mut vad = SileroVad::new(&vad_path, 512)
            .expect("load Silero VAD");

        let config = DiarizationConfig {
            window_secs: 1.5,
            hop_secs: 0.75,
            threshold: 0.5,
            ..Default::default()
        };
        let vad_config = VadConfig {
            frame_size: 512,
            threshold: 0.3,
            min_silence_ms: 300.0,
        };

        let pipeline = Pipeline::new(config, vad_config);

        let mut samples = sine_wave(200.0, 5.0, 0.7);
        samples.extend_from_slice(&sine_wave(800.0, 5.0, 0.7));

        let result = pipeline.run(&samples, &extractor, &mut vad).unwrap();
        eprintln!("Pipeline result: {} speakers, {} turns",
            result.num_speakers, result.turns.len());
        for turn in &result.turns {
            eprintln!("  {}: {:.2}s - {:.2}s",
                turn.speaker, turn.time.start, turn.time.end);
        }
    }
}
