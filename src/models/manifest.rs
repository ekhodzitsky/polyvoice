//! TOML manifest describing where each ONNX model lives, its checksum, and
//! which model each `Profile` resolves to.

use serde::Deserialize;
use std::collections::HashMap;

/// Schema version identifier the parser accepts. Bump when manifest format changes.
pub const SCHEMA_V1: &str = "polyvoice-models-v1";

/// The full registry manifest: list of model entries plus a profile → model_id map.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema: String,
    #[serde(default)]
    pub profiles: HashMap<String, ProfileEntry>,
    #[serde(default)]
    pub models: HashMap<String, ModelEntry>,
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
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("toml parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("unsupported schema version: expected '{}', got '{0}'", SCHEMA_V1)]
    UnsupportedSchema(String),
    #[error("profile '{profile}' references unknown model '{model}'")]
    DanglingModelRef { profile: String, model: String },
    #[error("model '{model}' has invalid sha256 '{sha}': expected 64 lowercase hex chars")]
    InvalidSha256 { model: String, sha: String },
}

impl Manifest {
    /// { true }
    /// `pub fn from_toml_str(s: &str) -> Result<Self, ManifestError>`
    /// { ret.as_ref().map_or(true, |m| m.schema == SCHEMA_V1) }
    /// Parse a manifest from a TOML source string.
    ///
    /// Validates: schema version, that every profile's `segmenter` and `embedder`
    /// reference an existing entry in `models`, and that every `sha256` is exactly
    /// 64 lowercase hex characters.
    pub fn from_toml_str(s: &str) -> Result<Self, ManifestError> {
        let m: Manifest = toml::from_str(s)?;
        if m.schema != SCHEMA_V1 {
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

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
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

    #[test]
    fn parse_known_good_manifest() {
        let m = Manifest::from_toml_str(SAMPLE).expect("must parse");
        assert_eq!(m.schema, "polyvoice-models-v1");
        assert_eq!(m.profiles.len(), 2);
        assert_eq!(m.models.len(), 2);
        assert_eq!(m.profiles["mobile"].segmenter, "silero_vad");
        assert_eq!(m.models["silero_vad"].size, Some(2300000));
        assert_eq!(m.models["silero_vad"].filename, "silero_vad.onnx");
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let bad = SAMPLE.replace("polyvoice-models-v1", "polyvoice-models-v999");
        let err = Manifest::from_toml_str(&bad).expect_err("must fail");
        assert!(format!("{err}").contains("v999") || format!("{err}").contains("schema"));
    }

    #[test]
    fn profile_lookup_resolves_to_models() {
        let m = Manifest::from_toml_str(SAMPLE).unwrap();
        let prof = m.profile("mobile").expect("mobile profile present");
        let seg = m.model(&prof.segmenter).expect("segmenter resolved");
        assert_eq!(seg.filename, "silero_vad.onnx");
    }

    #[test]
    fn missing_profile_returns_none() {
        let m = Manifest::from_toml_str(SAMPLE).unwrap();
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
    fn rejects_invalid_sha256_length() {
        let bad = SAMPLE.replace(
            "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
            "tooshort",
        );
        let err = Manifest::from_toml_str(&bad).expect_err("must fail");
        assert!(format!("{err}").to_lowercase().contains("sha256"));
    }
}
