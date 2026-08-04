//! Self-describing model configuration.
//!
//! Preference order when loading geometry / adapter metadata:
//! 1. ONNX `metadata_props` (via ort session metadata) — authoritative
//! 2. Manifest entry fields (schema v2) — transition fallback, `tracing::warn`
//! 3. Caller-supplied defaults — last resort, `tracing::warn`
//!
//! Hard-coded stage defaults (e.g. powerset 10 s / 1 s windows) remain valid
//! fallbacks until every shipped ONNX carries the props; the injection script
//! under `scripts/inject-model-metadata.py` writes them.

use super::manifest::ModelEntry;
use std::collections::HashMap;
use std::path::Path;

/// Where each field in [`ModelConfigMeta`] was sourced from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaSource {
    /// Read from ONNX `metadata_props`.
    OnnxProps,
    /// Filled from the TOML manifest entry (schema v2 fields).
    Manifest,
    /// Caller / stage hard-coded defaults.
    Defaults,
    /// Composite: some fields from model, some from fallback.
    Mixed,
}

/// Runtime configuration carried by (or about) a model file.
///
/// All fields are optional so partial metadata is representable. Callers merge
/// this onto their stage-specific defaults (e.g. `PowersetConfig`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelConfigMeta {
    pub model_type: Option<String>,
    pub adapter_type: Option<String>,
    pub version: Option<String>,
    pub sample_rate: Option<u32>,
    pub window_secs: Option<f32>,
    pub hop_secs: Option<f32>,
    pub embedding_dim: Option<usize>,
    pub num_speakers: Option<usize>,
    pub license: Option<String>,
    pub license_url: Option<String>,
    pub provenance: Option<String>,
    pub source: Option<MetaSource>,
}

impl ModelConfigMeta {
    /// True when no geometry / identity fields are set.
    pub fn is_empty(&self) -> bool {
        self.model_type.is_none()
            && self.adapter_type.is_none()
            && self.version.is_none()
            && self.sample_rate.is_none()
            && self.window_secs.is_none()
            && self.hop_secs.is_none()
            && self.embedding_dim.is_none()
            && self.num_speakers.is_none()
            && self.license.is_none()
            && self.license_url.is_none()
            && self.provenance.is_none()
    }

    /// Build from a manifest entry's schema-v2 fields (no ONNX I/O).
    pub fn from_manifest_entry(entry: &ModelEntry) -> Self {
        Self {
            model_type: entry.adapter_type.clone(),
            adapter_type: entry.adapter_type.clone(),
            version: entry.version.clone(),
            sample_rate: entry.sample_rate,
            window_secs: entry.window_secs,
            hop_secs: entry.hop_secs,
            embedding_dim: entry.embedding_dim,
            num_speakers: entry.num_speakers,
            license: entry.license.clone(),
            license_url: entry.license_url.clone(),
            provenance: entry.provenance.clone(),
            source: Some(MetaSource::Manifest),
        }
    }

    /// Overlay `other` onto `self`: only fill fields that are currently `None`.
    pub fn fill_from(&mut self, other: &Self) {
        if self.model_type.is_none() {
            self.model_type = other.model_type.clone();
        }
        if self.adapter_type.is_none() {
            self.adapter_type = other.adapter_type.clone();
        }
        if self.version.is_none() {
            self.version = other.version.clone();
        }
        if self.sample_rate.is_none() {
            self.sample_rate = other.sample_rate;
        }
        if self.window_secs.is_none() {
            self.window_secs = other.window_secs;
        }
        if self.hop_secs.is_none() {
            self.hop_secs = other.hop_secs;
        }
        if self.embedding_dim.is_none() {
            self.embedding_dim = other.embedding_dim;
        }
        if self.num_speakers.is_none() {
            self.num_speakers = other.num_speakers;
        }
        if self.license.is_none() {
            self.license = other.license.clone();
        }
        if self.license_url.is_none() {
            self.license_url = other.license_url.clone();
        }
        if self.provenance.is_none() {
            self.provenance = other.provenance.clone();
        }
    }

    /// Parse a flat key→value map (ONNX custom metadata or test fixture).
    pub fn from_props(props: &HashMap<String, String>) -> Self {
        fn get_str(props: &HashMap<String, String>, keys: &[&str]) -> Option<String> {
            keys.iter()
                .find_map(|k| props.get(*k).map(|s| s.trim().to_owned()))
                .filter(|s| !s.is_empty())
        }
        fn get_u32(props: &HashMap<String, String>, keys: &[&str]) -> Option<u32> {
            get_str(props, keys).and_then(|s| s.parse().ok())
        }
        fn get_f32(props: &HashMap<String, String>, keys: &[&str]) -> Option<f32> {
            get_str(props, keys).and_then(|s| s.parse().ok())
        }
        fn get_usize(props: &HashMap<String, String>, keys: &[&str]) -> Option<usize> {
            get_str(props, keys).and_then(|s| s.parse().ok())
        }

        Self {
            model_type: get_str(props, &["model_type", "model-type"]),
            adapter_type: get_str(props, &["adapter_type", "adapter-type"]),
            version: get_str(props, &["version", "model_version"]),
            sample_rate: get_u32(props, &["sample_rate", "sample-rate", "sr"]),
            window_secs: get_f32(props, &["window_secs", "window_size", "window-size"]),
            hop_secs: get_f32(props, &["hop_secs", "window_shift", "window-shift", "hop"]),
            embedding_dim: get_usize(props, &["embedding_dim", "embedding-dim", "output_dim"]),
            num_speakers: get_usize(props, &["num_speakers", "num-speakers", "max_speakers"]),
            license: get_str(props, &["license"]),
            license_url: get_str(props, &["license_url", "license-url"]),
            provenance: get_str(props, &["provenance", "author"]),
            source: Some(MetaSource::OnnxProps),
        }
    }
}

/// Load model configuration with the documented preference order.
///
/// - `onnx_path`: optional path to an ONNX file. When `None` or unreadable,
///   ONNX props are skipped.
/// - `manifest_entry`: optional schema-v2 fields used as fallback.
/// - `defaults`: last-resort values (stage hard-codes).
///
/// Emits `tracing::warn` whenever a fallback tier is used for any field that
/// the higher tier did not provide.
pub fn load_model_config(
    onnx_path: Option<&Path>,
    manifest_entry: Option<&ModelEntry>,
    defaults: &ModelConfigMeta,
) -> ModelConfigMeta {
    let mut meta = ModelConfigMeta::default();
    let mut used_onnx = false;
    let mut used_manifest = false;
    let mut used_defaults = false;

    // 1. ONNX metadata_props (feature-gated; no-op without onnx / missing file).
    if let Some(path) = onnx_path {
        match read_onnx_metadata_props(path) {
            Ok(props) if !props.is_empty() => {
                let from_onnx = ModelConfigMeta::from_props(&props);
                if !from_onnx.is_empty() {
                    meta = from_onnx;
                    used_onnx = true;
                }
            }
            Ok(_) => {
                tracing::warn!(
                    path = %path.display(),
                    "ONNX model has no polyvoice metadata_props; falling back to manifest/defaults"
                );
            }
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "failed to read ONNX metadata_props; falling back to manifest/defaults"
                );
            }
        }
    }

    // 2. Manifest entry fields.
    if let Some(entry) = manifest_entry {
        let from_manifest = ModelConfigMeta::from_manifest_entry(entry);
        let before = meta.clone();
        meta.fill_from(&from_manifest);
        if meta != before {
            used_manifest = true;
            if used_onnx {
                tracing::warn!(
                    "partial ONNX metadata_props; filled missing fields from manifest entry"
                );
            } else {
                tracing::warn!(
                    "using manifest entry fields for model config (no ONNX metadata_props)"
                );
            }
        }
    }

    // 3. Caller defaults.
    let before = meta.clone();
    meta.fill_from(defaults);
    if meta != before {
        used_defaults = true;
        tracing::warn!(
            "using hard-coded defaults for model config fields not present in ONNX/manifest"
        );
    }

    meta.source = Some(match (used_onnx, used_manifest, used_defaults) {
        (true, false, false) => MetaSource::OnnxProps,
        (false, true, false) => MetaSource::Manifest,
        (false, false, _) => MetaSource::Defaults,
        _ => MetaSource::Mixed,
    });
    meta
}

/// Read custom metadata key/value pairs from an ONNX file.
///
/// Opens a short-lived ort session and queries `ModelMetadata`. The typed
/// [`crate::onnx::OnnxError`] lets callers distinguish an unloadable model
/// from a metadata read failure.
#[cfg(feature = "onnx")]
pub fn read_onnx_metadata_props(
    path: &Path,
) -> Result<HashMap<String, String>, crate::onnx::OnnxError> {
    crate::onnx::read_model_metadata_props(path)
}

/// Read custom metadata key/value pairs from an ONNX file.
///
/// Without the `onnx` feature there is no runtime to query, so this is
/// infallible and always returns an empty map — callers fall through to the
/// manifest/defaults path.
#[cfg(not(feature = "onnx"))]
pub fn read_onnx_metadata_props(
    path: &Path,
) -> Result<HashMap<String, String>, std::convert::Infallible> {
    let _ = path;
    Ok(HashMap::new())
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::manifest::Manifest;

    const ENTRY_TOML: &str = r#"
        schema = "polyvoice-models-v2"
        [profiles.mobile]
        segmenter = "powerset_fp32"
        embedder  = "powerset_fp32"
        [models.powerset_fp32]
        url = "https://example.com/p.onnx"
        sha256 = "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234"
        filename = "powerset_fp32.onnx"
        adapter_type = "powerset-v1"
        version = "3.0"
        sample_rate = 16000
        window_secs = 10.0
        hop_secs = 1.0
        num_speakers = 3
        license = "MIT"
        provenance = "sherpa-onnx"
    "#;

    #[test]
    fn from_props_parses_known_keys() {
        let mut props = HashMap::new();
        props.insert("sample_rate".into(), "16000".into());
        props.insert("window_secs".into(), "10.0".into());
        props.insert("embedding_dim".into(), "256".into());
        props.insert("adapter_type".into(), "wespeaker-resnet34".into());
        let meta = ModelConfigMeta::from_props(&props);
        assert_eq!(meta.sample_rate, Some(16000));
        assert_eq!(meta.window_secs, Some(10.0));
        assert_eq!(meta.embedding_dim, Some(256));
        assert_eq!(meta.adapter_type.as_deref(), Some("wespeaker-resnet34"));
        assert_eq!(meta.source, Some(MetaSource::OnnxProps));
    }

    #[test]
    fn load_without_onnx_falls_back_to_manifest() {
        // Acceptance (a): model without metadata → fallback (+ warn via tracing).
        let m = Manifest::from_toml_str(ENTRY_TOML).unwrap();
        let entry = m.model("powerset_fp32").unwrap();
        let defaults = ModelConfigMeta {
            hop_secs: Some(0.5), // should NOT override manifest's 1.0
            ..ModelConfigMeta::default()
        };
        let meta = load_model_config(None, Some(entry), &defaults);
        assert_eq!(meta.sample_rate, Some(16000));
        assert_eq!(meta.window_secs, Some(10.0));
        assert_eq!(meta.hop_secs, Some(1.0));
        assert_eq!(meta.adapter_type.as_deref(), Some("powerset-v1"));
        assert_eq!(meta.source, Some(MetaSource::Manifest));
    }

    #[test]
    fn load_with_empty_everything_uses_defaults() {
        let defaults = ModelConfigMeta {
            sample_rate: Some(16000),
            window_secs: Some(10.0),
            source: Some(MetaSource::Defaults),
            ..ModelConfigMeta::default()
        };
        let meta = load_model_config(None, None, &defaults);
        assert_eq!(meta.sample_rate, Some(16000));
        assert_eq!(meta.window_secs, Some(10.0));
        assert_eq!(meta.source, Some(MetaSource::Defaults));
    }

    #[test]
    fn onnx_props_take_priority_over_manifest() {
        // Acceptance (b): metadata from model wins; hard-code/manifest ignored
        // for fields the model carries.
        let m = Manifest::from_toml_str(ENTRY_TOML).unwrap();
        let entry = m.model("powerset_fp32").unwrap();

        // Simulate ONNX props by feeding them through from_props + fill order.
        let mut props = HashMap::new();
        props.insert("sample_rate".into(), "8000".into()); // differs from manifest 16000
        props.insert("window_secs".into(), "5.0".into());
        let mut meta = ModelConfigMeta::from_props(&props);
        meta.fill_from(&ModelConfigMeta::from_manifest_entry(entry));
        assert_eq!(meta.sample_rate, Some(8000), "onnx wins");
        assert_eq!(meta.window_secs, Some(5.0), "onnx wins");
        // Fields only in manifest still fill in.
        assert_eq!(meta.hop_secs, Some(1.0));
        assert_eq!(meta.adapter_type.as_deref(), Some("powerset-v1"));
    }

    #[test]
    fn from_manifest_entry_maps_v2_fields() {
        let m = Manifest::from_toml_str(ENTRY_TOML).unwrap();
        let entry = m.model("powerset_fp32").unwrap();
        let meta = ModelConfigMeta::from_manifest_entry(entry);
        assert_eq!(meta.license.as_deref(), Some("MIT"));
        assert_eq!(meta.provenance.as_deref(), Some("sherpa-onnx"));
        assert_eq!(meta.num_speakers, Some(3));
        assert_eq!(meta.source, Some(MetaSource::Manifest));
    }

    #[test]
    fn is_empty_reflects_field_presence() {
        assert!(ModelConfigMeta::default().is_empty());
        // `source` alone does not count as content.
        let source_only = ModelConfigMeta {
            source: Some(MetaSource::Defaults),
            ..ModelConfigMeta::default()
        };
        assert!(source_only.is_empty());

        let cases: [ModelConfigMeta; 11] = [
            ModelConfigMeta {
                model_type: Some("x".into()),
                ..Default::default()
            },
            ModelConfigMeta {
                adapter_type: Some("x".into()),
                ..Default::default()
            },
            ModelConfigMeta {
                version: Some("x".into()),
                ..Default::default()
            },
            ModelConfigMeta {
                sample_rate: Some(16000),
                ..Default::default()
            },
            ModelConfigMeta {
                window_secs: Some(1.0),
                ..Default::default()
            },
            ModelConfigMeta {
                hop_secs: Some(0.5),
                ..Default::default()
            },
            ModelConfigMeta {
                embedding_dim: Some(256),
                ..Default::default()
            },
            ModelConfigMeta {
                num_speakers: Some(3),
                ..Default::default()
            },
            ModelConfigMeta {
                license: Some("MIT".into()),
                ..Default::default()
            },
            ModelConfigMeta {
                license_url: Some("u".into()),
                ..Default::default()
            },
            ModelConfigMeta {
                provenance: Some("p".into()),
                ..Default::default()
            },
        ];
        for (i, meta) in cases.iter().enumerate() {
            assert!(!meta.is_empty(), "case {i} must be non-empty");
        }
    }

    #[test]
    fn fill_from_never_overwrites_present_fields() {
        let full = ModelConfigMeta {
            model_type: Some("a".into()),
            adapter_type: Some("b".into()),
            version: Some("c".into()),
            sample_rate: Some(8000),
            window_secs: Some(2.0),
            hop_secs: Some(0.25),
            embedding_dim: Some(128),
            num_speakers: Some(2),
            license: Some("MIT".into()),
            license_url: Some("u".into()),
            provenance: Some("p".into()),
            source: None,
        };
        let mut meta = full.clone();
        let other = ModelConfigMeta {
            model_type: Some("z".into()),
            adapter_type: Some("z".into()),
            version: Some("z".into()),
            sample_rate: Some(16000),
            window_secs: Some(9.0),
            hop_secs: Some(9.0),
            embedding_dim: Some(512),
            num_speakers: Some(9),
            license: Some("Apache-2.0".into()),
            license_url: Some("z".into()),
            provenance: Some("z".into()),
            source: None,
        };
        meta.fill_from(&other);
        assert_eq!(meta, full, "no field may be overwritten");

        // And each None field is filled individually.
        let mut sparse = ModelConfigMeta::default();
        sparse.fill_from(&other);
        assert_eq!(sparse.model_type.as_deref(), Some("z"));
        assert_eq!(sparse.sample_rate, Some(16000));
        assert_eq!(sparse.num_speakers, Some(9));
    }

    #[test]
    fn from_props_accepts_alternate_key_spellings() {
        let mut props = HashMap::new();
        props.insert("model-type".into(), "segmenter".into());
        props.insert("adapter-type".into(), "powerset-v1".into());
        props.insert("model_version".into(), "3.0".into());
        props.insert("sr".into(), "16000".into());
        props.insert("window-size".into(), "10.0".into());
        props.insert("hop".into(), "1.0".into());
        props.insert("output_dim".into(), "256".into());
        props.insert("max_speakers".into(), "3".into());
        props.insert("license-url".into(), "https://example.com/L".into());
        props.insert("author".into(), "someone".into());
        let meta = ModelConfigMeta::from_props(&props);
        assert_eq!(meta.model_type.as_deref(), Some("segmenter"));
        assert_eq!(meta.adapter_type.as_deref(), Some("powerset-v1"));
        assert_eq!(meta.version.as_deref(), Some("3.0"));
        assert_eq!(meta.sample_rate, Some(16000));
        assert_eq!(meta.window_secs, Some(10.0));
        assert_eq!(meta.hop_secs, Some(1.0));
        assert_eq!(meta.embedding_dim, Some(256));
        assert_eq!(meta.num_speakers, Some(3));
        assert_eq!(meta.license_url.as_deref(), Some("https://example.com/L"));
        assert_eq!(meta.provenance.as_deref(), Some("someone"));

        // Remaining alternates for the numeric keys.
        let mut props = HashMap::new();
        props.insert("sample-rate".into(), "8000".into());
        props.insert("window_size".into(), "5.0".into());
        props.insert("window_shift".into(), "0.5".into());
        props.insert("window-shift".into(), "0.5".into());
        props.insert("embedding-dim".into(), "192".into());
        props.insert("num-speakers".into(), "4".into());
        let meta = ModelConfigMeta::from_props(&props);
        assert_eq!(meta.sample_rate, Some(8000));
        assert_eq!(meta.window_secs, Some(5.0));
        assert_eq!(meta.hop_secs, Some(0.5));
        assert_eq!(meta.embedding_dim, Some(192));
        assert_eq!(meta.num_speakers, Some(4));
    }

    #[test]
    fn from_props_trims_and_drops_empty_values() {
        let mut props = HashMap::new();
        props.insert("version".into(), "  1.2  ".into());
        props.insert("license".into(), "   ".into());
        props.insert("sample_rate".into(), " 16000 ".into());
        let meta = ModelConfigMeta::from_props(&props);
        assert_eq!(meta.version.as_deref(), Some("1.2"));
        assert_eq!(meta.license, None, "whitespace-only value counts as absent");
        assert_eq!(meta.sample_rate, Some(16000));
    }

    #[test]
    fn from_props_ignores_unparseable_numbers() {
        let mut props = HashMap::new();
        props.insert("sample_rate".into(), "not-a-number".into());
        props.insert("window_secs".into(), "abc".into());
        props.insert("hop_secs".into(), "1.0.0".into());
        props.insert("embedding_dim".into(), "-3".into());
        props.insert("num_speakers".into(), "3.5".into());
        let meta = ModelConfigMeta::from_props(&props);
        assert_eq!(meta.sample_rate, None);
        assert_eq!(meta.window_secs, None);
        assert_eq!(meta.hop_secs, None);
        assert_eq!(meta.embedding_dim, None);
        assert_eq!(meta.num_speakers, None);
    }

    #[test]
    fn load_with_propsless_onnx_falls_back_to_manifest() {
        // silero_vad.onnx ships without polyvoice metadata_props, so the read
        // succeeds but yields an empty map (and without the onnx feature the
        // read is a no-op) — either way the manifest tier is exercised.
        let m = Manifest::from_toml_str(ENTRY_TOML).unwrap();
        let entry = m.model("powerset_fp32").unwrap();
        let silero = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx");
        let meta = load_model_config(Some(&silero), Some(entry), &ModelConfigMeta::default());
        assert_eq!(meta.window_secs, Some(10.0));
        assert_eq!(meta.adapter_type.as_deref(), Some("powerset-v1"));
        assert_eq!(meta.source, Some(MetaSource::Manifest));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn load_with_unreadable_onnx_falls_back_to_manifest() {
        let m = Manifest::from_toml_str(ENTRY_TOML).unwrap();
        let entry = m.model("powerset_fp32").unwrap();
        let tmp = tempfile::TempDir::new().unwrap();
        let junk = tmp.path().join("junk.onnx");
        std::fs::write(&junk, b"this is not an onnx model").unwrap();
        let meta = load_model_config(Some(&junk), Some(entry), &ModelConfigMeta::default());
        assert_eq!(meta.window_secs, Some(10.0));
        assert_eq!(meta.source, Some(MetaSource::Manifest));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn load_reads_geometry_from_onnx_props() {
        let powerset = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/powerset_fp32.onnx");
        if !powerset.exists() {
            eprintln!("skip: models/powerset_fp32.onnx missing");
            return;
        }
        let meta = load_model_config(Some(&powerset), None, &ModelConfigMeta::default());
        assert_eq!(meta.source, Some(MetaSource::OnnxProps));
        assert_eq!(meta.sample_rate, Some(16000));
        assert_eq!(
            meta.model_type.as_deref(),
            Some("pyannote-segmentation-3.0")
        );
        assert_eq!(meta.num_speakers, Some(3));
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn load_mixing_onnx_manifest_and_defaults_is_marked_mixed() {
        let m = Manifest::from_toml_str(ENTRY_TOML).unwrap();
        let entry = m.model("powerset_fp32").unwrap();
        let powerset = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/powerset_fp32.onnx");
        // The bundled powerset props carry sample_rate/model_type but no hop
        // or adapter_type, so the manifest fills those; a remaining gap is
        // closed by caller defaults.
        let defaults = ModelConfigMeta {
            embedding_dim: Some(256),
            ..ModelConfigMeta::default()
        };
        let meta = load_model_config(Some(&powerset), Some(entry), &defaults);
        assert_eq!(meta.source, Some(MetaSource::Mixed));
        assert_eq!(meta.sample_rate, Some(16000), "identity from onnx props");
        assert_eq!(meta.hop_secs, Some(1.0), "hop only present in manifest");
        assert_eq!(
            meta.adapter_type.as_deref(),
            Some("powerset-v1"),
            "adapter_type only present in manifest"
        );
        assert_eq!(meta.embedding_dim, Some(256), "gap closed by defaults");
    }
}
