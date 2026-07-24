//! TOML manifest describing where each ONNX model lives, its checksum, and
//! which model each `Profile` resolves to.
//!
//! Schema versions:
//! - `polyvoice-models-v1` — url/sha256/size/filename/signature/calibration
//! - `polyvoice-models-v2` — adds license/provenance/adapter_type/version,
//!   optional geometry fallbacks, and stage-scoped version aliases

use serde::Deserialize;
use std::collections::HashMap;

/// Schema version v1 (legacy). Still accepted by the parser.
pub const SCHEMA_V1: &str = "polyvoice-models-v1";
/// Schema version v2: license/provenance/adapter_type/version + aliases.
pub const SCHEMA_V2: &str = "polyvoice-models-v2";

/// Returns true if `schema` is a known, supported manifest schema id.
pub fn is_supported_schema(schema: &str) -> bool {
    schema == SCHEMA_V1 || schema == SCHEMA_V2
}

/// The full registry manifest: list of model entries plus a profile → model_id map.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema: String,
    #[serde(default)]
    pub profiles: HashMap<String, ProfileEntry>,
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,
    /// Stage-scoped version aliases (`aliases.embedder.latest = "wespeaker_resnet34"`).
    /// Absent in v1 manifests; defaults to empty.
    #[serde(default)]
    pub aliases: HashMap<String, HashMap<String, String>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProfileEntry {
    pub segmenter: String,
    pub embedder: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub url: String,
    pub sha256: String,
    /// Optional declared size in bytes (informational).
    #[serde(default)]
    pub size: Option<u64>,
    /// Filename used when caching to disk. Required so the cache is deterministic
    /// across renames upstream.
    pub filename: String,
    /// Optional calibration descriptor (filled in M5 for INT8 entries).
    #[serde(default)]
    pub calibration: Option<String>,
    /// Minisign signature (raw .minisig text) — optional during transition.
    #[serde(default)]
    pub signature: Option<String>,

    // --- schema v2 fields (all optional so v1 records keep deserializing) ---
    /// SPDX license identifier (e.g. `"MIT"`, `"Apache-2.0"`).
    #[serde(default)]
    pub license: Option<String>,
    /// URL of the full license text.
    #[serde(default)]
    pub license_url: Option<String>,
    /// Free-form provenance string (upstream weights, author, conversion date).
    #[serde(default)]
    pub provenance: Option<String>,
    /// Adapter type string matching an [`super::adapter::AdapterStage`] registration
    /// (e.g. `"powerset-v1"`, `"silero"`, `"cam++"`, `"wespeaker-resnet34"`).
    #[serde(default)]
    pub adapter_type: Option<String>,
    /// Model version pin (e.g. `"1.0"`, `"3.0"`). Distinct from alias keys.
    #[serde(default)]
    pub version: Option<String>,

    // Geometry / runtime fallbacks used when ONNX `metadata_props` are absent.
    // These never override values successfully read from the binary.
    /// Expected input sample rate (Hz).
    #[serde(default)]
    pub sample_rate: Option<u32>,
    /// Sliding-window length in seconds (segmenters).
    #[serde(default)]
    pub window_secs: Option<f32>,
    /// Sliding-window hop in seconds (segmenters).
    #[serde(default)]
    pub hop_secs: Option<f32>,
    /// Embedding output dimension (embedders).
    #[serde(default)]
    pub embedding_dim: Option<usize>,
    /// Max local speakers / classes the model emits (segmenters).
    #[serde(default)]
    pub num_speakers: Option<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("toml parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("unsupported schema version: expected '{SCHEMA_V1}' or '{SCHEMA_V2}', got '{0}'")]
    UnsupportedSchema(String),
    #[error("profile '{profile}' references unknown model '{model}'")]
    DanglingModelRef { profile: String, model: String },
    #[error("alias '{stage}.{alias}' references unknown model '{model}'")]
    DanglingAliasRef {
        stage: String,
        alias: String,
        model: String,
    },
    #[error("model '{model}' has invalid sha256 '{sha}': expected 64 lowercase hex chars")]
    InvalidSha256 { model: String, sha: String },
}

impl Manifest {
    /// { true }
    /// `pub fn from_toml_str(s: &str) -> Result<Self, ManifestError>`
    /// { ret.as_ref().map_or(true, |m| is_supported_schema(&m.schema)) }
    /// Parse a manifest from a TOML source string.
    ///
    /// Validates: schema version (v1 or v2), that every profile's `segmenter`
    /// and `embedder` reference an existing entry in `models`, that every
    /// alias target exists, and that every `sha256` is exactly 64 lowercase
    /// hex characters.
    pub fn from_toml_str(s: &str) -> Result<Self, ManifestError> {
        let m: Manifest = toml::from_str(s)?;
        if !is_supported_schema(&m.schema) {
            return Err(ManifestError::UnsupportedSchema(m.schema));
        }
        // Check dangling profile references before sha256 so that a missing model
        // is reported as DanglingModelRef even when other models have invalid sha256.
        let mut sorted_profile_ids: Vec<&String> = m.profiles.keys().collect();
        sorted_profile_ids.sort();
        for name in sorted_profile_ids {
            let prof = &m.profiles[name];
            if !m.models.contains_key(&prof.segmenter) {
                return Err(ManifestError::DanglingModelRef {
                    profile: name.clone(),
                    model: prof.segmenter.clone(),
                });
            }
            if !m.models.contains_key(&prof.embedder) {
                return Err(ManifestError::DanglingModelRef {
                    profile: name.clone(),
                    model: prof.embedder.clone(),
                });
            }
        }
        // Alias targets must resolve to known models.
        let mut sorted_stages: Vec<&String> = m.aliases.keys().collect();
        sorted_stages.sort();
        for stage in sorted_stages {
            let stage_aliases = &m.aliases[stage];
            let mut sorted_alias_keys: Vec<&String> = stage_aliases.keys().collect();
            sorted_alias_keys.sort();
            for alias in sorted_alias_keys {
                let target = &stage_aliases[alias];
                if !m.models.contains_key(target) {
                    return Err(ManifestError::DanglingAliasRef {
                        stage: stage.clone(),
                        alias: alias.clone(),
                        model: target.clone(),
                    });
                }
            }
        }
        let mut sorted_model_ids: Vec<&String> = m.models.keys().collect();
        sorted_model_ids.sort();
        for model_id in sorted_model_ids {
            let entry = &m.models[model_id];
            if !is_valid_sha256_hex(&entry.sha256) {
                return Err(ManifestError::InvalidSha256 {
                    model: model_id.clone(),
                    sha: truncate_for_display(&entry.sha256),
                });
            }
        }
        Ok(m)
    }

    /// { true }
    /// `pub fn profile(&self, id: &str) -> Option<&ProfileEntry>`
    /// { ret == self.profiles.get(id) }
    pub fn profile(&self, id: &str) -> Option<&ProfileEntry> {
        self.profiles.get(id)
    }

    /// { true }
    /// `pub fn model(&self, id: &str) -> Option<&ModelEntry>`
    /// { ret == self.models.get(id) }
    pub fn model(&self, id: &str) -> Option<&ModelEntry> {
        self.models.get(id)
    }

    /// Resolve a stage-scoped version alias to a concrete model id.
    ///
    /// Returns `None` if the stage or alias key is unknown. Does **not**
    /// fall through to direct model-id lookup — use [`Self::resolve_model_ref`]
    /// for that.
    pub fn alias(&self, stage: &str, alias: &str) -> Option<&str> {
        self.aliases
            .get(stage)
            .and_then(|m| m.get(alias))
            .map(String::as_str)
    }

    /// Resolve `id_or_alias` for `stage` to a concrete model id.
    ///
    /// Order:
    /// 1. If `id_or_alias` is a known model id → return it unchanged.
    /// 2. Else if it is a known alias for `stage` → return the pinned target
    ///    and log the resolution (reproducible DER reports need the pin).
    /// 3. Else → `None`.
    ///
    /// The returned `&str` is borrowed from `self` (model map key or alias target).
    pub fn resolve_model_ref<'a>(&'a self, stage: &str, id_or_alias: &str) -> Option<&'a str> {
        if let Some((key, _)) = self.models.get_key_value(id_or_alias) {
            return Some(key.as_str());
        }
        if let Some(target) = self.alias(stage, id_or_alias) {
            tracing::info!(
                stage = stage,
                alias = id_or_alias,
                resolved = target,
                "resolved model version alias to pinned model id"
            );
            return Some(target);
        }
        None
    }
}

fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

/// Truncate a string for inclusion in error messages. SHA-256 is 64 chars;
/// anything past 80 is almost certainly garbage and bloating the message hurts.
fn truncate_for_display(s: &str) -> String {
    if s.len() <= 80 {
        s.to_owned()
    } else {
        format!("{}…[{} more chars]", &s[..72], s.len() - 72)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_V1: &str = r#"
        schema = "polyvoice-models-v1"

        [profiles.mobile]
        segmenter = "silero_vad"
        embedder  = "wespeaker_resnet34"

        [profiles.balanced]
        segmenter = "silero_vad"
        embedder  = "wespeaker_resnet34"

        [models.silero_vad]
        url      = "https://example.com/silero_vad.onnx"
        sha256   = "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234"
        size     = 2300000
        filename = "silero_vad.onnx"

        [models.wespeaker_resnet34]
        url      = "https://example.com/wespeaker.onnx"
        sha256   = "11112222333344445555666677778888aaaabbbbccccddddeeeeffff00001111"
        size     = 25000000
        filename = "wespeaker_resnet34.onnx"
    "#;

    const SAMPLE_V2: &str = r#"
        schema = "polyvoice-models-v2"

        [aliases.segmenter]
        latest = "powerset_fp32"
        v1 = "powerset_fp32"

        [aliases.embedder]
        latest = "wespeaker_resnet34"

        [profiles.mobile]
        segmenter = "powerset_fp32"
        embedder  = "wespeaker_resnet34"

        [models.powerset_fp32]
        url      = "https://example.com/powerset.onnx"
        sha256   = "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234"
        size     = 6000000
        filename = "powerset_fp32.onnx"
        license  = "MIT"
        license_url = "https://example.com/LICENSE"
        provenance = "sherpa-onnx-pyannote-segmentation-3-0"
        adapter_type = "powerset-v1"
        version  = "3.0"
        sample_rate = 16000
        window_secs = 10.0
        hop_secs = 1.0
        num_speakers = 3

        [models.wespeaker_resnet34]
        url      = "https://example.com/wespeaker.onnx"
        sha256   = "11112222333344445555666677778888aaaabbbbccccddddeeeeffff00001111"
        size     = 25000000
        filename = "wespeaker_resnet34.onnx"
        license  = "Apache-2.0"
        adapter_type = "wespeaker-resnet34"
        version  = "1.0"
        sample_rate = 16000
        embedding_dim = 256
    "#;

    #[test]
    fn parse_known_good_manifest_v1() {
        let m = Manifest::from_toml_str(SAMPLE_V1).expect("must parse");
        assert_eq!(m.schema, "polyvoice-models-v1");
        assert_eq!(m.profiles.len(), 2);
        assert_eq!(m.models.len(), 2);
        assert_eq!(m.profiles["mobile"].segmenter, "silero_vad");
        assert_eq!(m.models["silero_vad"].size, Some(2300000));
        assert_eq!(m.models["silero_vad"].filename, "silero_vad.onnx");
        // v1 records leave v2 fields empty
        assert!(m.models["silero_vad"].license.is_none());
        assert!(m.models["silero_vad"].adapter_type.is_none());
        assert!(m.aliases.is_empty());
    }

    #[test]
    fn parse_known_good_manifest_v2() {
        let m = Manifest::from_toml_str(SAMPLE_V2).expect("must parse v2");
        assert_eq!(m.schema, SCHEMA_V2);
        let p = &m.models["powerset_fp32"];
        assert_eq!(p.license.as_deref(), Some("MIT"));
        assert_eq!(p.adapter_type.as_deref(), Some("powerset-v1"));
        assert_eq!(p.version.as_deref(), Some("3.0"));
        assert_eq!(p.sample_rate, Some(16000));
        assert_eq!(p.embedding_dim, None);
        assert_eq!(p.window_secs, Some(10.0));
        assert_eq!(m.alias("segmenter", "latest"), Some("powerset_fp32"));
    }

    #[test]
    fn v2_parser_still_reads_v1_records() {
        // Acceptance (f): manifest v2 path keeps reading pure v1 TOML.
        let m = Manifest::from_toml_str(SAMPLE_V1).expect("v1 still readable");
        assert!(is_supported_schema(&m.schema));
        assert!(m.model("silero_vad").is_some());
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let bad = SAMPLE_V1.replace("polyvoice-models-v1", "polyvoice-models-v999");
        let err = Manifest::from_toml_str(&bad).expect_err("must fail");
        assert!(format!("{err}").contains("v999") || format!("{err}").contains("schema"));
    }

    #[test]
    fn profile_lookup_resolves_to_models() {
        let m = Manifest::from_toml_str(SAMPLE_V1).unwrap();
        let prof = m.profile("mobile").expect("mobile profile present");
        let seg = m.model(&prof.segmenter).expect("segmenter resolved");
        assert_eq!(seg.filename, "silero_vad.onnx");
    }

    #[test]
    fn missing_profile_returns_none() {
        let m = Manifest::from_toml_str(SAMPLE_V1).unwrap();
        assert!(m.profile("nope").is_none());
    }

    #[test]
    fn rejects_profile_with_missing_model_reference() {
        let bad = r#"
            schema = "polyvoice-models-v1"
            [profiles.mobile]
            segmenter = "ghost_model"
            embedder  = "ghost_model"
            [models.silero_vad]
            url = "https://example.com/x"
            sha256 = "abc"
            filename = "silero_vad.onnx"
        "#;
        let err = Manifest::from_toml_str(bad).expect_err("must fail");
        assert!(format!("{err}").to_lowercase().contains("ghost_model"));
    }

    #[test]
    fn rejects_dangling_alias_target() {
        let bad = r#"
            schema = "polyvoice-models-v2"
            [aliases.embedder]
            latest = "ghost_model"
            [profiles.mobile]
            segmenter = "silero_vad"
            embedder  = "silero_vad"
            [models.silero_vad]
            url = "https://example.com/x"
            sha256 = "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234"
            filename = "silero_vad.onnx"
        "#;
        let err = Manifest::from_toml_str(bad).expect_err("must fail");
        assert!(format!("{err}").contains("ghost_model"));
    }

    #[test]
    fn rejects_invalid_sha256_length() {
        let bad = SAMPLE_V1.replace(
            "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
            "tooshort",
        );
        let err = Manifest::from_toml_str(&bad).expect_err("must fail");
        assert!(format!("{err}").to_lowercase().contains("sha256"));
    }

    #[test]
    fn resolve_model_ref_returns_direct_id() {
        let m = Manifest::from_toml_str(SAMPLE_V2).unwrap();
        assert_eq!(
            m.resolve_model_ref("embedder", "wespeaker_resnet34"),
            Some("wespeaker_resnet34")
        );
    }

    #[test]
    fn resolve_model_ref_resolves_latest_alias() {
        let m = Manifest::from_toml_str(SAMPLE_V2).unwrap();
        assert_eq!(
            m.resolve_model_ref("segmenter", "latest"),
            Some("powerset_fp32")
        );
        assert_eq!(
            m.resolve_model_ref("embedder", "latest"),
            Some("wespeaker_resnet34")
        );
    }

    #[test]
    fn resolve_model_ref_unknown_returns_none() {
        let m = Manifest::from_toml_str(SAMPLE_V2).unwrap();
        assert!(m.resolve_model_ref("embedder", "nope").is_none());
        assert!(m.resolve_model_ref("vad", "latest").is_none());
    }
}
