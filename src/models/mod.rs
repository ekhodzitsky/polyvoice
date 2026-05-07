//! Model registry — manifest-driven downloads with SHA-256 verification.

pub mod manifest;
pub use manifest::{Manifest, ManifestError, ModelEntry, ProfileEntry, SCHEMA_V1};

/// The default manifest shipped with the crate. Embedded at compile time.
pub const DEFAULT_MANIFEST_TOML: &str = include_str!("manifest.toml");

/// Parse the bundled default manifest. Panics in debug if the embedded TOML is
/// malformed — that's a static asset bug caught by `cargo test`.
pub fn default_manifest() -> Manifest {
    Manifest::from_toml_str(DEFAULT_MANIFEST_TOML)
        .expect("embedded manifest.toml must parse — this is a static-asset bug")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_parses() {
        // This will panic if the bundled manifest.toml is malformed.
        let m = default_manifest();
        assert_eq!(m.schema, SCHEMA_V1);
        assert!(m.profiles.contains_key("mobile"));
        assert!(m.profiles.contains_key("balanced"));
    }

    #[test]
    fn embedded_manifest_lists_legacy_models() {
        let m = default_manifest();
        assert!(m.models.contains_key("silero_vad"));
        assert!(m.models.contains_key("wespeaker_resnet34"));
    }

    #[test]
    fn both_profiles_resolve_to_legacy_models_in_m0() {
        // Documented assumption: until M2/M5, both profiles map to the same
        // physical models. Remove this test when CAM++ lands in M2.
        let m = default_manifest();
        let mob = m.profile("mobile").unwrap();
        let bal = m.profile("balanced").unwrap();
        assert_eq!(mob.segmenter, bal.segmenter);
        assert_eq!(mob.embedder, bal.embedder);
    }
}
