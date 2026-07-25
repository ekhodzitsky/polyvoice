//! Model profile bundles (Mobile / Balanced / Custom).
use serde::{Deserialize, Serialize};

/// Pre-configured model bundles trading off accuracy and footprint.
///
/// `Mobile` targets weak/embedded ARM CPUs (≤10 MB total models, ≤200 MB peak RAM).
/// `Balanced` targets modern phone/laptop ARM CPUs (≤35 MB total models, ≤400 MB peak RAM).
/// `Custom` defers all model selection to the caller and is used by `PipelineBuilder`
/// when individual `Segmenter`/`Embedder`/`Clusterer` instances are supplied directly.
///
/// Added in v0.6 (M0).
/// §5.1 for the full motivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Profile {
    Mobile,
    Balanced,
    Custom,
}

impl Profile {
    /// Embedding dimension produced by the embedder for this profile.
    /// Returns 0 for `Custom` (caller must resolve dimension explicitly).
    pub const fn embedding_dim(self) -> usize {
        match self {
            Profile::Mobile => 512,   // CAM++ output dim (voxceleb_CAM++.onnx)
            Profile::Balanced => 256, // WeSpeaker ResNet34 output dim
            Profile::Custom => 0,
        }
    }

    /// Default cosine similarity threshold tuned to the embedding space of this profile.
    pub const fn default_threshold(self) -> f32 {
        match self {
            Profile::Mobile => 0.55,
            Profile::Balanced => 0.45,
            Profile::Custom => 0.5,
        }
    }

    /// Stable identifier used in the manifest TOML and CLI flags.
    pub const fn manifest_id(self) -> &'static str {
        match self {
            Profile::Mobile => "mobile",
            Profile::Balanced => "balanced",
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
            "unknown profile '{}': expected mobile|balanced|custom",
            self.0
        )
    }
}

impl std::error::Error for ProfileParseError {}
