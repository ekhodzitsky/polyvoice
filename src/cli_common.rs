//! Shared wiring helpers for the CLI-family binaries (`polyvoice`,
//! `polyvoice-bench`, `polyvoice-measure`, `polyvoice-mcp`).
//!
//! Everything here is flag-to-config translation, pipeline construction, or
//! bench-dataset walking — the pieces every binary used to carry its own copy
//! of. Keeping them here holds the binaries to thin wrappers and stops the
//! copies from drifting apart (defaults, error wording, AMI id fallback).

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::models::ModelRegistry;
use crate::onnx::ExecutionProvider;
use crate::pipeline_v2::{ClustererKind, Pipeline, PipelineConfig};
use crate::rttm::{RttmSegment, group_by_file, parse_rttm_file, to_speaker_turns};
use crate::types::{ClusterConfig, DiarizationConfig, SpeakerTurn};
use crate::{FbankOnnxExtractor, SileroVad};

/// Parse a `--clusterer`-style selector into the v2 clusterer kind. `threshold`
/// is used only for `ahc`.
pub fn parse_clusterer_kind(name: &str, threshold: f32) -> Result<ClustererKind> {
    match name {
        "ahc" => Ok(ClustererKind::Ahc { threshold }),
        "vbx" => Ok(ClustererKind::Vbx),
        other => anyhow::bail!("unknown --clusterer '{other}' (expected 'ahc' or 'vbx')"),
    }
}

/// Effective AHC threshold from CLI-style flags. An explicit `--threshold`
/// always wins. Without one, the default depends on the active scorer: the
/// raw-cosine default is meaningless on the AS-norm z-scale (and vice versa),
/// so AS-norm falls back to the calibrated VoxConverse z-threshold. When a
/// domain profile applies (and no explicit threshold was given), the profile's
/// calibrated value replaces this fallback at pipeline build time.
pub fn resolve_ahc_threshold(explicit: Option<f32>, as_norm: bool) -> f32 {
    explicit.unwrap_or(if as_norm {
        // The VoxConverse z-threshold is calibrated and always Some.
        crate::clusterer::domain::VOXCONVERSE
            .as_norm_threshold
            .unwrap_or(4.0)
    } else {
        crate::types::DEFAULT_AHC_THRESHOLD
    })
}

/// Parse a `--domain-profile`-style selector into the calibrated domain
/// profile. Profiles are data: the profile swaps threshold / cohort-size
/// values, never code paths.
pub fn parse_domain_profile(name: &str) -> Result<crate::clusterer::DomainProfile> {
    crate::clusterer::domain_profile(name).ok_or_else(|| {
        anyhow::anyhow!("unknown --domain-profile '{name}' (expected voxconverse|ami|callhome)")
    })
}

/// Assemble the v2 AS-norm config from CLI-style flags. Without an explicit
/// cohort file the registry cohort model id is used (resolved at pipeline
/// build time, with the `POLYVOICE_ASNORM_COHORT` env override). `top_n`
/// falls back to the selected domain profile's cohort size, then the built-in
/// default. An explicit cohort path must exist — fail fast here, before any
/// model download.
pub fn resolve_as_norm_config(
    enabled: bool,
    cohort: Option<PathBuf>,
    top_n: Option<usize>,
) -> Result<Option<crate::clusterer::AsNormConfig>> {
    if !enabled {
        return Ok(None);
    }
    use crate::clusterer::CohortSource;
    let source = match cohort {
        Some(p) => {
            if !p.is_file() {
                anyhow::bail!("--cohort file not found: {}", p.display());
            }
            CohortSource::Path(p)
        }
        None => CohortSource::ModelId(crate::clusterer::DEFAULT_ASNORM_COHORT_MODEL_ID.to_owned()),
    };
    Ok(Some(crate::clusterer::AsNormConfig {
        top_n: top_n.unwrap_or(crate::clusterer::DEFAULT_AS_NORM_TOP_N),
        cohort: source,
    }))
}

/// Resolve the CLI clustering flags into the v2 config triple
/// `(clusterer, as_norm, domain)`, applying the CLI-level precedence and
/// validation rules that the library config cannot express:
///
/// - `--as-norm` and `--domain-profile` require the AHC clusterer (they are
///   silently inert otherwise — a hard error beats a surprise).
/// - An explicit `--threshold` beats the domain profile's calibrated
///   threshold: the profile is then dropped from the returned config so the
///   builder cannot override the explicit value (the profile's cohort size
///   was already folded into the AS-norm config). Library users get the
///   opposite precedence — `PipelineConfig.domain` always overrides the
///   configured threshold at build time.
///
/// Suspicious-but-legal combinations warn on stderr: a raw-cosine-looking
/// explicit threshold on the AS-norm z-scale, an AS-norm domain profile
/// without a calibrated z-threshold (the VoxConverse fallback is used), and a
/// `--cohort` path without `--as-norm` (ignored).
pub fn resolve_clusterer_flags(
    clusterer: &str,
    threshold: Option<f32>,
    as_norm: bool,
    cohort: Option<PathBuf>,
    domain_profile: Option<&str>,
) -> Result<(
    ClustererKind,
    Option<crate::clusterer::AsNormConfig>,
    Option<crate::clusterer::DomainProfile>,
)> {
    let kind = parse_clusterer_kind(clusterer, resolve_ahc_threshold(threshold, as_norm))?;
    let is_ahc = matches!(kind, ClustererKind::Ahc { .. });
    if as_norm && !is_ahc {
        anyhow::bail!("--as-norm applies to the AHC clusterer only (pass --clusterer ahc)");
    }
    let domain = domain_profile.map(parse_domain_profile).transpose()?;
    if domain.is_some() && !is_ahc {
        anyhow::bail!("--domain-profile applies to the AHC clusterer only (pass --clusterer ahc)");
    }
    if !as_norm && cohort.is_some() {
        eprintln!("warning: --cohort is ignored without --as-norm");
    }
    if as_norm {
        match (threshold, domain) {
            (Some(t), _) if t < 1.5 => {
                eprintln!(
                    "warning: --threshold {t} is on the raw-cosine scale; with --as-norm the \
                     merge threshold is a z-score (calibrated domains use z = 4-5)"
                );
            }
            (None, Some(d)) if d.as_norm_threshold.is_none() => {
                let fallback = resolve_ahc_threshold(None, true);
                eprintln!(
                    "warning: --domain-profile {} has no calibrated AS-norm z-threshold; \
                     using the VoxConverse-calibrated z = {fallback}",
                    d.name
                );
            }
            _ => {}
        }
    }
    let as_norm_config = resolve_as_norm_config(as_norm, cohort, domain.map(|d| d.as_norm_top_n))?;
    // CLI precedence: an explicit threshold already won via
    // `resolve_ahc_threshold`; drop the profile so the builder's library
    // contract (profile overrides the configured threshold) cannot undo it.
    let domain = if threshold.is_some() { None } else { domain };
    Ok((kind, as_norm_config, domain))
}

/// Parse an `--execution-provider`-style selector. `auto` resolves to the best
/// provider for the current target; providers not compiled into the build warn
/// and fall back to CPU at session-build time.
pub fn parse_execution_provider(s: &str) -> Result<ExecutionProvider> {
    Ok(match s {
        "auto" => ExecutionProvider::auto(),
        "cpu" => ExecutionProvider::Cpu,
        "coreml" => ExecutionProvider::CoreMl,
        "nnapi" => ExecutionProvider::Nnapi,
        "cuda" => ExecutionProvider::Cuda,
        "xnnpack" => ExecutionProvider::XnnPack,
        other => anyhow::bail!(
            "unknown --execution-provider '{other}' (expected auto|cpu|coreml|nnapi|cuda|xnnpack)"
        ),
    })
}

/// Convert a `--speakers`/`--max-speakers`-style value into the v2 config's
/// `u8` ceiling. Out-of-range input is rejected outright rather than silently
/// clamped, so the caller learns the value was invalid.
pub fn max_speakers_u8(n: usize) -> Result<u8> {
    u8::try_from(n)
        .ok()
        .filter(|&v| v > 0)
        .ok_or_else(|| anyhow::anyhow!("max_speakers must be in 1..=255, got {n}"))
}

/// Legacy (v1) diarization config: crate defaults with the caller's AHC
/// cosine-similarity threshold.
pub fn legacy_diarization_config(threshold: f32) -> DiarizationConfig {
    DiarizationConfig {
        cluster: ClusterConfig {
            threshold,
            ..Default::default()
        },
        ..DiarizationConfig::default()
    }
}

/// ONNX sessions for the legacy (v1) pipeline: the profile embedder plus a
/// Silero VAD. Sessions are file-independent, so callers processing many files
/// build one stack and reuse it — `LegacyPipeline::run` resets the VAD state
/// at the start of every run, keeping reused sessions numerically identical
/// to per-file construction.
pub struct LegacyStack {
    pub extractor: FbankOnnxExtractor,
    pub vad: SileroVad,
}

/// Load the legacy-pipeline ONNX sessions: embedder (on `embedder_ep`) and
/// Silero VAD (always CPU, its validated configuration).
pub fn load_legacy_stack(
    embedder_path: &Path,
    embedding_dim: usize,
    embedder_ep: ExecutionProvider,
    vad_path: &Path,
    vad_frame_size: usize,
) -> Result<LegacyStack> {
    let extractor = FbankOnnxExtractor::new(embedder_path, embedding_dim, 1, embedder_ep)
        .context("load embedder")?;
    let vad = SileroVad::new(vad_path, vad_frame_size).context("load vad")?;
    Ok(LegacyStack { extractor, vad })
}

/// Build the v2 pipeline from a model registry. When the VBx clusterer is
/// selected and its PLDA params cannot be resolved, the error names the
/// remedies (explicit dir, env var, registry download, AHC fallback) instead
/// of surfacing a bare build failure.
pub fn build_v2_pipeline(config: PipelineConfig, registry: ModelRegistry) -> Result<Pipeline> {
    let vbx = matches!(config.clusterer, ClustererKind::Vbx);
    Pipeline::builder()
        .config(config)
        .with_models_from(registry)
        .build()
        .with_context(|| {
            if vbx {
                "build pipeline v2 (clusterer=vbx): set vbx_plda_dir / POLYVOICE_VBX_PLDA_DIR \
                 (CLI flag: --vbx-plda-dir), allow registry PLDA download, or select the ahc \
                 clusterer (CLI: --clusterer ahc)"
                    .to_string()
            } else {
                "build pipeline v2".to_string()
            }
        })
}

/// Sorted `audio/*.wav` listing of a bench dataset directory (`{audio,rttm}`
/// layout), truncated to `max_files` when set.
pub fn list_wavs(dataset: &Path, max_files: Option<usize>) -> Result<Vec<PathBuf>> {
    let audio_dir = dataset.join("audio");
    let mut wavs: Vec<PathBuf> = std::fs::read_dir(&audio_dir)
        .with_context(|| format!("read_dir {}", audio_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "wav"))
        .map(|e| e.path())
        .collect();
    wavs.sort();
    if let Some(n) = max_files {
        wavs.truncate(n);
    }
    Ok(wavs)
}

/// Reference RTTM segments for `stem` from `rttm_dir`, with the AMI-style id
/// fallback (`EN2002a.Mix-Headset.wav` → `EN2002a`). Empty when neither the
/// full stem nor the fallback prefix names a file in the RTTM.
pub fn load_rttm_segments(rttm_dir: &Path, stem: &str) -> Result<Vec<RttmSegment>> {
    let rttm = rttm_dir.join(format!("{stem}.rttm"));
    let raw = parse_rttm_file(&rttm).with_context(|| format!("parse {}", rttm.display()))?;
    let grouped = group_by_file(&raw);
    let segs: Vec<RttmSegment> = grouped
        .get(stem)
        .or_else(|| stem.split('.').next().and_then(|s| grouped.get(s)))
        .map(|v| v.iter().map(|s| (*s).clone()).collect())
        .unwrap_or_default();
    Ok(segs)
}

/// Reference speaker turns for `stem` — [`load_rttm_segments`] projected onto
/// the canonical turn type.
pub fn load_ref_turns(rttm_dir: &Path, stem: &str) -> Result<Vec<SpeakerTurn>> {
    let (turns, _) = to_speaker_turns(&load_rttm_segments(rttm_dir, stem)?);
    Ok(turns)
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clusterer_kind_parses_known_names() {
        assert!(matches!(
            parse_clusterer_kind("vbx", 0.7).unwrap(),
            ClustererKind::Vbx
        ));
        match parse_clusterer_kind("ahc", 0.7).unwrap() {
            ClustererKind::Ahc { threshold } => assert_eq!(threshold, 0.7),
            other => panic!("expected Ahc, got {other:?}"),
        }
        assert!(parse_clusterer_kind("nope", 0.7).is_err());
    }

    #[test]
    fn domain_profile_parses_known_names() {
        for name in ["voxconverse", "ami", "callhome"] {
            let p = parse_domain_profile(name).unwrap();
            assert_eq!(p.name, name);
        }
        let err = parse_domain_profile("switchboard").err().unwrap();
        assert!(format!("{err:#}").contains("voxconverse|ami|callhome"));
    }

    #[test]
    fn ahc_threshold_resolution_picks_the_active_scorers_scale() {
        // Explicit value always wins, on either scale.
        assert_eq!(resolve_ahc_threshold(Some(0.7), false), 0.7);
        assert_eq!(resolve_ahc_threshold(Some(6.0), true), 6.0);
        // Defaults track the scorer: raw cosine vs AS-norm z-score.
        assert_eq!(
            resolve_ahc_threshold(None, false),
            crate::types::DEFAULT_AHC_THRESHOLD
        );
        let z = resolve_ahc_threshold(None, true);
        assert!(
            z > 1.0,
            "z-scale default must sit above the raw-cosine range, got {z}"
        );
        assert_eq!(
            z,
            crate::clusterer::domain::VOXCONVERSE
                .as_norm_threshold
                .unwrap()
        );
    }

    #[test]
    fn as_norm_config_disabled_is_none() {
        assert!(resolve_as_norm_config(false, None, None).unwrap().is_none());
        // Even a bogus cohort path is ignored when AS-norm is off.
        assert!(
            resolve_as_norm_config(false, Some(PathBuf::from("/no/such.npy")), None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn as_norm_config_requires_existing_cohort_file() {
        let err = resolve_as_norm_config(true, Some(PathBuf::from("/no/such/cohort.npy")), None)
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("/no/such/cohort.npy"));
    }

    #[test]
    fn as_norm_config_defaults_to_registry_cohort_and_default_top_n() {
        let cfg = resolve_as_norm_config(true, None, None).unwrap().unwrap();
        assert_eq!(cfg.top_n, crate::clusterer::DEFAULT_AS_NORM_TOP_N);
        match cfg.cohort {
            crate::clusterer::CohortSource::ModelId(id) => {
                assert_eq!(id, crate::clusterer::DEFAULT_ASNORM_COHORT_MODEL_ID);
            }
            other => panic!("expected ModelId, got {other:?}"),
        }
    }

    #[test]
    fn as_norm_config_honors_explicit_path_and_top_n() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cohort.npy");
        std::fs::write(&path, b"placeholder").unwrap();
        let cfg = resolve_as_norm_config(true, Some(path.clone()), Some(42))
            .unwrap()
            .unwrap();
        assert_eq!(cfg.top_n, 42);
        match cfg.cohort {
            crate::clusterer::CohortSource::Path(p) => assert_eq!(p, path),
            other => panic!("expected Path, got {other:?}"),
        }
    }

    #[test]
    fn clusterer_flags_require_ahc_for_as_norm_and_domain_profile() {
        let err = resolve_clusterer_flags("vbx", None, true, None, None)
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("--clusterer ahc"));
        // A domain profile with the default VBx clusterer must not be
        // silently inert either.
        let err = resolve_clusterer_flags("vbx", None, false, None, Some("ami"))
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("--domain-profile"));
        assert!(format!("{err:#}").contains("--clusterer ahc"));
    }

    #[test]
    fn clusterer_flags_explicit_threshold_beats_domain_profile() {
        let (kind, as_norm, domain) =
            resolve_clusterer_flags("ahc", Some(0.7), false, None, Some("ami")).unwrap();
        assert_eq!(kind, ClustererKind::Ahc { threshold: 0.7 });
        // The profile is dropped from the config so the builder cannot
        // override the explicit threshold; without AS-norm there is no
        // cohort size to preserve.
        assert!(domain.is_none());
        assert!(as_norm.is_none());
    }

    #[test]
    fn clusterer_flags_profile_supplies_threshold_and_top_n_without_explicit() {
        let z_fallback = crate::clusterer::domain::VOXCONVERSE
            .as_norm_threshold
            .unwrap();
        let (kind, as_norm, domain) =
            resolve_clusterer_flags("ahc", None, true, None, Some("ami")).unwrap();
        // No explicit threshold: the fallback z-threshold goes into the kind
        // and the profile stays in the config to override it at build time.
        assert_eq!(
            kind,
            ClustererKind::Ahc {
                threshold: z_fallback
            }
        );
        assert_eq!(domain.unwrap().name, "ami");
        // The profile's cohort size feeds the AS-norm config.
        assert_eq!(
            as_norm.unwrap().top_n,
            crate::clusterer::domain::AMI.as_norm_top_n
        );
    }

    #[test]
    fn clusterer_flags_warn_paths_still_resolve() {
        let z_fallback = crate::clusterer::domain::VOXCONVERSE
            .as_norm_threshold
            .unwrap();
        // Raw-scale-looking explicit threshold with AS-norm: warns, resolves.
        let (kind, _, domain) =
            resolve_clusterer_flags("ahc", Some(0.5), true, None, Some("ami")).unwrap();
        assert_eq!(kind, ClustererKind::Ahc { threshold: 0.5 });
        assert!(domain.is_none());
        // Uncalibrated-z domain with AS-norm: warns, keeps the fallback.
        let (kind, _, domain) =
            resolve_clusterer_flags("ahc", None, true, None, Some("callhome")).unwrap();
        assert_eq!(
            kind,
            ClustererKind::Ahc {
                threshold: z_fallback
            }
        );
        assert_eq!(domain.unwrap().name, "callhome");
        // Cohort without AS-norm: warns, ignored.
        let (_, as_norm, _) =
            resolve_clusterer_flags("ahc", None, false, Some(PathBuf::from("/x.npy")), None)
                .unwrap();
        assert!(as_norm.is_none());
    }

    #[test]
    fn execution_provider_parses_known_names() {
        for name in ["auto", "cpu", "coreml", "nnapi", "cuda", "xnnpack"] {
            assert!(parse_execution_provider(name).is_ok(), "{name}");
        }
        assert!(parse_execution_provider("tpu").is_err());
    }

    #[test]
    fn max_speakers_u8_accepts_valid_range() {
        assert_eq!(max_speakers_u8(1).unwrap(), 1);
        assert_eq!(max_speakers_u8(255).unwrap(), 255);
    }

    #[test]
    fn max_speakers_u8_rejects_out_of_range() {
        assert!(max_speakers_u8(0).is_err());
        assert!(max_speakers_u8(256).is_err());
    }

    #[test]
    fn rttm_segments_fall_back_to_ami_style_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let rttm_dir = dir.path();
        // Exact stem match: the RTTM file id column is used directly.
        std::fs::write(
            rttm_dir.join("plain.rttm"),
            "SPEAKER plain 1 0.0 1.0 <NA> <NA> B <NA> <NA>\n",
        )
        .unwrap();
        let exact = load_rttm_segments(rttm_dir, "plain").unwrap();
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].speaker, "B");
        // Full filename stem falls back to the prefix before the first dot.
        std::fs::write(
            rttm_dir.join("EN2002a.Mix-Headset.rttm"),
            "SPEAKER EN2002a 1 0.0 1.0 <NA> <NA> A <NA> <NA>\n",
        )
        .unwrap();
        let fallback = load_rttm_segments(rttm_dir, "EN2002a.Mix-Headset").unwrap();
        assert_eq!(fallback.len(), 1);
        assert_eq!(fallback[0].speaker, "A");
    }

    #[test]
    fn list_wavs_sorts_and_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let audio = dir.path().join("audio");
        std::fs::create_dir(&audio).unwrap();
        for name in ["b.wav", "a.wav", "c.txt"] {
            std::fs::write(audio.join(name), []).unwrap();
        }
        let all = list_wavs(dir.path(), None).unwrap();
        assert_eq!(all.len(), 2);
        assert!(all[0].ends_with("a.wav"));
        let one = list_wavs(dir.path(), Some(1)).unwrap();
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn list_wavs_missing_audio_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = list_wavs(dir.path(), None).err().unwrap();
        assert!(format!("{err:#}").contains("read_dir"));
    }

    #[test]
    fn load_rttm_segments_missing_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_rttm_segments(dir.path(), "nope").err().unwrap();
        assert!(format!("{err:#}").contains("parse"));
    }

    #[test]
    fn load_rttm_segments_unknown_stem_yields_empty() {
        let dir = tempfile::tempdir().unwrap();
        // The RTTM parses, but its file id matches neither the stem nor the
        // AMI-style prefix fallback, so the lookup comes back empty.
        std::fs::write(
            dir.path().join("absent.rttm"),
            "SPEAKER other 1 0.0 1.0 <NA> <NA> A <NA> <NA>\n",
        )
        .unwrap();
        let segs = load_rttm_segments(dir.path(), "absent").unwrap();
        assert!(segs.is_empty());
    }

    #[test]
    fn load_ref_turns_projects_segments_onto_turns() {
        let dir = tempfile::tempdir().unwrap();
        // RTTM columns are start + duration: 2.0 + 3.0 → turn ends at 5.0.
        std::fs::write(
            dir.path().join("plain.rttm"),
            "SPEAKER plain 1 0.0 1.5 <NA> <NA> B <NA> <NA>\n\
             SPEAKER plain 1 2.0 3.0 <NA> <NA> B <NA> <NA>\n",
        )
        .unwrap();
        let turns = load_ref_turns(dir.path(), "plain").unwrap();
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].speaker.0, 0);
        assert!((turns[0].time.start - 0.0).abs() < 1e-9);
        assert!((turns[1].time.end - 5.0).abs() < 1e-9);
    }

    #[test]
    fn legacy_config_applies_threshold() {
        let cfg = legacy_diarization_config(0.42);
        assert_eq!(cfg.cluster.threshold, 0.42);
    }

    #[test]
    fn load_legacy_stack_rejects_missing_embedder() {
        let err = load_legacy_stack(
            Path::new("/nonexistent/embedder.onnx"),
            192,
            ExecutionProvider::Cpu,
            Path::new("/nonexistent/vad.onnx"),
            512,
        )
        .err()
        .unwrap();
        assert!(format!("{err:#}").contains("load embedder"));
    }

    /// Registry rooted at the checked-in model files (SHA-256-verified cache
    /// hits, no network). `None` when the local models are absent.
    fn local_models_registry() -> Option<ModelRegistry> {
        // Prefer models/int8 (checked-in quant outputs); fall back to models/.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let int8 = root.join("int8");
        let dir = if int8.join("powerset_int8.onnx").is_file() {
            int8
        } else {
            root.clone()
        };
        for f in ["powerset_int8.onnx", "resnet34_int8.onnx"] {
            if !dir.join(f).is_file() {
                eprintln!(
                    "{} not found — skipping model-backed test",
                    dir.join(f).display()
                );
                return None;
            }
        }
        ModelRegistry::with_cache_dir(&dir).ok()
    }

    #[test]
    fn load_legacy_stack_loads_sessions_from_local_models() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let embedder = [
            models.join("int8/resnet34_int8.onnx"),
            models.join("resnet34_int8.onnx"),
            models.join("wespeaker_resnet34.onnx"),
        ]
        .into_iter()
        .find(|p| p.is_file());
        let vad = models.join("silero_vad.onnx");
        let Some(embedder) = embedder else {
            eprintln!("local embedder ONNX not found — skipping");
            return;
        };
        if !vad.is_file() {
            eprintln!("local ONNX models not found — skipping");
            return;
        }
        load_legacy_stack(&embedder, 256, ExecutionProvider::Cpu, &vad, 512).unwrap();
    }

    #[test]
    fn load_legacy_stack_rejects_missing_vad() {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
        let embedder = [
            models.join("int8/resnet34_int8.onnx"),
            models.join("resnet34_int8.onnx"),
            models.join("wespeaker_resnet34.onnx"),
        ]
        .into_iter()
        .find(|p| p.is_file());
        let Some(embedder) = embedder else {
            eprintln!("local embedder ONNX not found — skipping");
            return;
        };
        let err = load_legacy_stack(
            embedder.as_path(),
            256,
            ExecutionProvider::Cpu,
            Path::new("/nonexistent/vad.onnx"),
            512,
        )
        .err()
        .unwrap();
        assert!(format!("{err:#}").contains("load vad"));
    }

    #[test]
    fn build_v2_pipeline_ahc_builds_from_local_models() {
        let Some(registry) = local_models_registry() else {
            return;
        };
        let config = PipelineConfig {
            clusterer: ClustererKind::Ahc { threshold: 0.7 },
            execution_provider: ExecutionProvider::Cpu,
            ..PipelineConfig::default()
        };
        build_v2_pipeline(config, registry).unwrap();
    }

    #[cfg(feature = "vbx")]
    #[test]
    fn build_v2_pipeline_vbx_error_names_remedies() {
        let Some(registry) = local_models_registry() else {
            return;
        };
        let empty = tempfile::tempdir().unwrap();
        let config = PipelineConfig {
            clusterer: ClustererKind::Vbx,
            vbx_plda_dir: Some(empty.path().to_path_buf()),
            execution_provider: ExecutionProvider::Cpu,
            ..PipelineConfig::default()
        };
        let err = build_v2_pipeline(config, registry).err().unwrap();
        let msg = format!("{err:#}");
        assert!(msg.contains("vbx_plda_dir"), "{msg}");
        assert!(msg.contains("--clusterer ahc"), "{msg}");
    }
}
