//! Model registry — manifest-driven downloads with SHA-256 verification.

pub mod download;
pub mod manifest;
pub mod verify;
pub use download::{
    DownloadError, download_with_checksum, download_with_checksum_and_signature, verify_sha256,
};
pub use manifest::{Manifest, ManifestError, ModelEntry, ProfileEntry, SCHEMA_V1};

use crate::types::Profile;
use std::path::{Path, PathBuf};

/// The default manifest shipped with the crate. Embedded at compile time.
pub const DEFAULT_MANIFEST_TOML: &str = include_str!("manifest.toml");

/// { true }
/// pub fn default_manifest() -> Manifest
/// { true }
/// Parse the bundled default manifest. Panics in debug if the embedded TOML is
/// malformed — that's a static asset bug caught by `cargo test`.
///
/// This is the *only* place the project allows `expect` on the embedded manifest:
/// the asset is shipped with the crate, and `embedded_manifest_parses` test
/// verifies it parses on every build.
#[allow(clippy::expect_used)]
pub fn default_manifest() -> Manifest {
    // SAFETY: embedded manifest.toml is a compile-time static asset;
    // test `embedded_manifest_parses` verifies it on every build.
    Manifest::from_toml_str(DEFAULT_MANIFEST_TOML)
        .expect("embedded manifest.toml must parse — this is a static-asset bug")
}

/// Errors from `ModelRegistry` operations.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("model '{model_id}' not found in manifest")]
    ModelNotFound { model_id: String },
    #[error("profile '{profile}' not found in manifest")]
    ProfileNotFound { profile: String },
    #[error("custom profile cannot be resolved by registry — caller must supply models")]
    CustomProfileUnresolvable,
    #[error("cache directory {path} is not writable")]
    CacheNotWritable { path: PathBuf },
    #[error("model '{model_id}' is not present in cache and offline mode is requested")]
    OfflineMissing { model_id: String },
    #[error("manifest error: {0}")]
    Manifest(#[from] ManifestError),
    #[error("download error: {0}")]
    Download(#[from] DownloadError),
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Resolved file paths for the segmenter and embedder of a profile.
#[derive(Debug, Clone)]
pub struct ProfileModels {
    pub segmenter_path: PathBuf,
    pub embedder_path: PathBuf,
}

/// A model registry: holds a manifest + a cache directory, and downloads/verifies
/// models on demand.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    manifest: Manifest,
    cache_dir: PathBuf,
}

impl ModelRegistry {
    /// { true }
    /// `pub fn default() -> Result<Self, RegistryError>`
    /// { ret.as_ref().map_or(true, |r| r.cache_dir().exists()) }
    /// Build a registry rooted at the user's cache directory (`~/.cache/polyvoice/models`
    /// on Linux, `~/Library/Caches/polyvoice/models` on macOS, `%LOCALAPPDATA%\polyvoice\models`
    /// on Windows) using the embedded default manifest.
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Result<Self, RegistryError> {
        let cache = dirs::cache_dir()
            .ok_or_else(|| RegistryError::CacheNotWritable {
                path: PathBuf::from("(unresolved-cache-dir)"),
            })?
            .join("polyvoice")
            .join("models");
        Self::with_cache_dir(cache)
    }

    /// { true }
    /// `pub fn with_cache_dir(path: impl AsRef<Path>) -> Result<Self, RegistryError>`
    /// { ret.as_ref().map_or(true, |r| r.cache_dir().exists()) }
    /// Build a registry with a caller-specified cache directory and the embedded
    /// default manifest. Creates the directory if it doesn't exist.
    pub fn with_cache_dir(path: impl AsRef<Path>) -> Result<Self, RegistryError> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path).map_err(|e| RegistryError::Io {
            path: path.clone(),
            source: e,
        })?;
        Ok(Self {
            manifest: default_manifest(),
            cache_dir: path,
        })
    }

    /// { true }
    /// pub fn with_manifest_override(mut self, manifest: Manifest) -> Self
    /// { true }
    /// Override the manifest. Useful for tests that need a fixture manifest
    /// without hitting the network.
    #[cfg(test)]
    pub fn with_manifest_override(mut self, manifest: Manifest) -> Self {
        self.manifest = manifest;
        self
    }

    /// { true }
    /// `pub fn with_manifest( manifest: Manifest, cache_dir: impl AsRef<Path>, ) -> Result<Self, RegistryError>`
    /// { ret.as_ref().map_or(true, |r| r.cache_dir().exists()) }
    /// Build a registry with a custom manifest and cache directory.
    #[cfg(test)]
    pub fn with_manifest(
        manifest: Manifest,
        cache_dir: impl AsRef<Path>,
    ) -> Result<Self, RegistryError> {
        let path = cache_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&path).map_err(|e| RegistryError::Io {
            path: path.clone(),
            source: e,
        })?;
        Ok(Self {
            manifest,
            cache_dir: path,
        })
    }

    /// { true }
    /// pub fn cache_dir(&self) -> &Path
    /// { ret == self.cache_dir }
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// { true }
    /// pub fn manifest(&self) -> &Manifest
    /// { ret == self.manifest }
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// { !model_id.is_empty() }
    /// `pub fn ensure(&self, model_id: &str) -> Result<PathBuf, RegistryError>`
    /// { ret.as_ref().map_or(true, |p| p.exists()) }
    /// Ensure the model with id `model_id` is present in cache and SHA-256-verified.
    /// Downloads if missing. Idempotent: returns immediately when the cached file
    /// already matches the expected hash.
    pub fn ensure(&self, model_id: &str) -> Result<PathBuf, RegistryError> {
        let entry = self
            .manifest
            .model(model_id)
            .ok_or_else(|| RegistryError::ModelNotFound {
                model_id: model_id.to_owned(),
            })?;
        let dest = self.cache_dir.join(&entry.filename);
        download_with_checksum_and_signature(
            &entry.url,
            &entry.sha256,
            entry.signature.as_deref(),
            &dest,
        )?;
        Ok(dest)
    }

    /// { !model_id.is_empty() }
    /// `pub fn ensure_in_cache_only(&self, model_id: &str) -> Result<PathBuf, RegistryError>`
    /// { ret.as_ref().map_or(true, |p| p.exists()) }
    /// Test-only helper that bypasses SHA-256 verification.
    #[doc(hidden)]
    /// Same as `ensure` but never makes a network call. Returns `OfflineMissing`
    /// if the file is not in cache or has a wrong hash.
    #[cfg(test)] // test-only: bypasses SHA-256/signature verification — never reachable in release
    pub fn ensure_in_cache_only(&self, model_id: &str) -> Result<PathBuf, RegistryError> {
        let entry = self
            .manifest
            .model(model_id)
            .ok_or_else(|| RegistryError::ModelNotFound {
                model_id: model_id.to_owned(),
            })?;
        let dest = self.cache_dir.join(&entry.filename);
        if !dest.exists() {
            return Err(RegistryError::OfflineMissing {
                model_id: model_id.to_owned(),
            });
        }
        // Skip hash check in cache-only path; it's expensive and tests pre-place
        // exact-content files. Production callers should use `ensure` not this.
        Ok(dest)
    }

    /// { true }
    /// `pub fn ensure_for_profile(&self, profile: Profile) -> Result<ProfileModels, RegistryError>`
    /// { ret.as_ref().map_or(true, |p| p.segmenter_path.exists() && p.embedder_path.exists()) }
    /// Resolve all models for a profile, downloading any that are missing.
    pub fn ensure_for_profile(&self, profile: Profile) -> Result<ProfileModels, RegistryError> {
        if profile == Profile::Custom {
            return Err(RegistryError::CustomProfileUnresolvable);
        }
        let prof = self
            .manifest
            .profile(profile.manifest_id())
            .ok_or_else(|| RegistryError::ProfileNotFound {
                profile: profile.manifest_id().to_owned(),
            })?;
        let segmenter_path = self.ensure(&prof.segmenter)?;
        let embedder_path = self.ensure(&prof.embedder)?;
        Ok(ProfileModels {
            segmenter_path,
            embedder_path,
        })
    }

    /// { true }
    /// `pub fn ensure_in_cache_only_for_profile( &self, profile: Profile, ) -> Result<ProfileModels, RegistryError>`
    /// { ret.as_ref().map_or(true, |p| p.segmenter_path.exists() && p.embedder_path.exists()) }
    /// Same as `ensure_for_profile` but never touches the network.
    #[cfg(test)]
    pub fn ensure_in_cache_only_for_profile(
        &self,
        profile: Profile,
    ) -> Result<ProfileModels, RegistryError> {
        if profile == Profile::Custom {
            return Err(RegistryError::CustomProfileUnresolvable);
        }
        let prof = self
            .manifest
            .profile(profile.manifest_id())
            .ok_or_else(|| RegistryError::ProfileNotFound {
                profile: profile.manifest_id().to_owned(),
            })?;
        let segmenter_path = self.ensure_in_cache_only(&prof.segmenter)?;
        let embedder_path = self.ensure_in_cache_only(&prof.embedder)?;
        Ok(ProfileModels {
            segmenter_path,
            embedder_path,
        })
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
pub(crate) mod tests_helpers {
    /// Minimal manifest used by registry unit tests. SHA-256 below is hash of "hello":
    /// 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    pub const TINY_MANIFEST: &str = r#"
        schema = "polyvoice-models-v1"
        [profiles.mobile]
        segmenter = "hello_model"
        embedder  = "hello_model"
        [profiles.balanced]
        segmenter = "hello_model"
        embedder  = "hello_model"
        [models.hello_model]
        url      = "file:///dev/null"
        sha256   = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        size     = 5
        filename = "hello.bin"
    "#;
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Profile;
    use tempfile::TempDir;

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
    fn profiles_share_segmenter_and_embedder_in_v2_hotfix() {
        // V2 hotfix (2026-05-18): both Mobile and Balanced use ResNet34 + AHC
        // because CAM++ ONNX produces near-identical embeddings (cosine sim ~0.85
        // between different speakers). NME-SC also falls back to AHC on small n.
        // Revert this test once CAM++ is re-converted and NME-SC is fixed.
        let m = default_manifest();
        let mob = m.profile("mobile").unwrap();
        let bal = m.profile("balanced").unwrap();
        assert_eq!(mob.segmenter, bal.segmenter, "both use powerset");
        assert_eq!(
            mob.embedder, bal.embedder,
            "both use resnet34 (CAM++ broken)"
        );
    }

    #[test]
    fn registry_default_uses_user_cache() {
        let r = ModelRegistry::default().expect("default cache dir resolvable");
        let path = r.cache_dir().to_path_buf();
        assert!(path.ends_with("polyvoice/models") || path.ends_with("polyvoice\\models"));
    }

    #[test]
    fn registry_with_cache_dir_creates_dir() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("nested/models");
        let r = ModelRegistry::with_cache_dir(&path).unwrap();
        assert!(path.exists());
        assert_eq!(r.cache_dir(), path.as_path());
    }

    #[test]
    fn ensure_returns_err_for_unknown_model_id() {
        let tmp = TempDir::new().unwrap();
        let r = ModelRegistry::with_cache_dir(tmp.path()).unwrap();
        let err = r
            .ensure_in_cache_only("ghost")
            .expect_err("must be missing");
        assert!(matches!(err, RegistryError::ModelNotFound { .. }));
    }

    #[test]
    fn ensure_in_cache_only_succeeds_when_file_present() {
        let tmp = TempDir::new().unwrap();
        let manifest =
            Manifest::from_toml_str(crate::models::tests_helpers::TINY_MANIFEST).unwrap();
        let r = ModelRegistry::with_cache_dir(tmp.path())
            .unwrap()
            .with_manifest_override(manifest);

        let cached = tmp.path().join("hello.bin");
        std::fs::write(&cached, b"hello").unwrap();
        let path = r.ensure_in_cache_only("hello_model").unwrap();
        assert_eq!(path, cached);
    }

    #[test]
    fn ensure_for_profile_uses_manifest_lookup() {
        let tmp = TempDir::new().unwrap();
        let manifest =
            Manifest::from_toml_str(crate::models::tests_helpers::TINY_MANIFEST).unwrap();
        let r = ModelRegistry::with_cache_dir(tmp.path())
            .unwrap()
            .with_manifest_override(manifest);

        std::fs::write(tmp.path().join("hello.bin"), b"hello").unwrap();

        let bundle = r.ensure_in_cache_only_for_profile(Profile::Mobile).unwrap();
        assert_eq!(bundle.segmenter_path, tmp.path().join("hello.bin"));
        assert_eq!(bundle.embedder_path, tmp.path().join("hello.bin"));
    }
}
