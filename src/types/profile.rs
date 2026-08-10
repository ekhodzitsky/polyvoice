//! Model profile bundles (Mobile / Balanced / Fast / Custom).
use serde::{Deserialize, Serialize};

/// Pre-configured model bundles.
///
/// As of 0.17 every shipping profile (`Mobile`, `Balanced`, `Fast`) resolves
/// the same **INT8** pair (`powerset_int8` + `resnet34_int8`, ~8.4 MB total).
/// `Fast` remains a CLI/manifest alias of that pair. `Custom` defers model
/// selection to the caller (`PipelineBuilder` with injected stages).
///
/// Added in v0.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Profile {
    Mobile,
    Balanced,
    Fast,
    Custom,
}

impl Profile {
    /// Embedding dimension produced by the embedder for this profile.
    /// Returns 0 for `Custom` (caller must resolve dimension explicitly).
    pub const fn embedding_dim(self) -> usize {
        match self {
            // INT8 WeSpeaker ResNet34 (same for mobile/balanced/fast).
            Profile::Mobile | Profile::Balanced | Profile::Fast => 256,
            Profile::Custom => 0,
        }
    }

    /// Default cosine similarity threshold tuned to the embedding space of this profile.
    pub const fn default_threshold(self) -> f32 {
        match self {
            Profile::Mobile => 0.55,
            Profile::Balanced => super::config::DEFAULT_AHC_THRESHOLD,
            Profile::Fast => super::config::DEFAULT_AHC_THRESHOLD,
            Profile::Custom => 0.5,
        }
    }

    /// Stable identifier used in the manifest TOML and CLI flags.
    pub const fn manifest_id(self) -> &'static str {
        match self {
            Profile::Mobile => "mobile",
            Profile::Balanced => "balanced",
            Profile::Fast => "fast",
            Profile::Custom => "custom",
        }
    }
}

impl std::str::FromStr for Profile {
    type Err = ProfileParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "mobile" => Ok(Profile::Mobile),
            "balanced" => Ok(Profile::Balanced),
            "fast" => Ok(Profile::Fast),
            "custom" => Ok(Profile::Custom),
            other => Err(ProfileParseError(other.to_owned())),
        }
    }
}

/// Returned by `Profile::from_str` when the input doesn't match a known variant.
#[derive(Debug, Clone)]
pub struct ProfileParseError(pub String);

impl std::fmt::Display for ProfileParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unknown profile '{}': expected mobile|balanced|fast|custom",
            self.0
        )
    }
}

impl std::error::Error for ProfileParseError {}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_is_case_insensitive() {
        assert_eq!("mobile".parse::<Profile>().unwrap(), Profile::Mobile);
        assert_eq!("BALANCED".parse::<Profile>().unwrap(), Profile::Balanced);
        assert_eq!("Fast".parse::<Profile>().unwrap(), Profile::Fast);
        assert_eq!("CUSTOM".parse::<Profile>().unwrap(), Profile::Custom);
    }

    #[test]
    fn from_str_unknown_profile_reports_expected_set() {
        let err = "weird".parse::<Profile>().unwrap_err();
        assert_eq!(
            err.to_string(),
            "unknown profile 'weird': expected mobile|balanced|fast|custom"
        );
        // Usable as a trait object error.
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn manifest_ids_parse_back_to_their_profile() {
        for p in [
            Profile::Mobile,
            Profile::Balanced,
            Profile::Fast,
            Profile::Custom,
        ] {
            assert_eq!(p.manifest_id().parse::<Profile>().unwrap(), p);
        }
    }

    #[test]
    fn embedding_dims_and_thresholds_match_model_bundles() {
        assert_eq!(Profile::Mobile.embedding_dim(), 256);
        assert_eq!(Profile::Balanced.embedding_dim(), 256);
        assert_eq!(Profile::Fast.embedding_dim(), 256);
        assert_eq!(Profile::Custom.embedding_dim(), 0);

        assert_eq!(Profile::Mobile.default_threshold(), 0.55);
        assert_eq!(
            Profile::Balanced.default_threshold(),
            crate::types::config::DEFAULT_AHC_THRESHOLD
        );
        assert_eq!(
            Profile::Fast.default_threshold(),
            crate::types::config::DEFAULT_AHC_THRESHOLD
        );
        assert_eq!(Profile::Custom.default_threshold(), 0.5);
    }

    #[test]
    fn profile_serde_roundtrip() {
        for p in [
            Profile::Mobile,
            Profile::Balanced,
            Profile::Fast,
            Profile::Custom,
        ] {
            let json = serde_json::to_string(&p).unwrap();
            assert_eq!(serde_json::from_str::<Profile>(&json).unwrap(), p);
        }
    }
}
