//! Model registry — manifest-driven downloads with SHA-256 verification,
//! adapter selection by config string, and self-describing model metadata.

pub mod adapter;
pub mod download;
pub mod manifest;
pub mod metadata;
pub mod verify;
pub use adapter::{AdapterError, AdapterFactory, AdapterRegistry, AdapterStage, BuiltinAdapter};
pub use download::{
    DownloadError, download_with_checksum, download_with_checksum_and_signature, verify_sha256,
};
pub use manifest::{
    Manifest, ManifestError, ModelEntry, ProfileEntry, SCHEMA_V1, SCHEMA_V2, is_supported_schema,
};
pub use metadata::{MetaSource, ModelConfigMeta, load_model_config, read_onnx_metadata_props};

use crate::types::Profile;
use std::path::{Path, PathBuf};

/// The default manifest shipped with the crate. Embedded at compile time.
pub const DEFAULT_MANIFEST_TOML: &str = include_str!("manifest.toml");

/// One VBx PLDA artifact: manifest id, on-disk filename, sha256, byte size.
///
/// Source of truth for ids + integrity checks (must match `manifest.toml` and
/// `fixtures/vbx-plda/`). Not profile-resolved — only pulled when the `vbx`
/// clusterer is selected without a local PLDA dir. SHA-256 only until minisign
/// signatures land (same optional-model pattern as `sortformer_v2`).
#[derive(Clone, Copy, Debug)]
pub struct VbxPldaArtifact {
    pub id: &'static str,
    pub filename: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// The six precomputed VBx PLDA `.npy` files, in `PldaModel::from_dir` order.
pub const VBX_PLDA_ARTIFACTS: &[VbxPldaArtifact] = &[
    VbxPldaArtifact {
        id: "vbx_plda_transform",
        filename: "plda_transform.npy",
        sha256: "90261469714415743f4b8a86ee6b89466db858bde3c5944367cccfb7abd34f14",
        size: 131_200,
    },
    VbxPldaArtifact {
        id: "vbx_plda_phi_computed",
        filename: "plda_phi_computed.npy",
        sha256: "6ef7cf2f5a23a45b66f440f9a996a4cf5c047b369829af695d50ef18aa0a35e3",
        size: 1_152,
    },
    VbxPldaArtifact {
        id: "vbx_plda_mean1",
        filename: "plda_mean1.npy",
        sha256: "e424c0c352182aa8e0f555dec1f3b30e29a20b9ed6b25d339f112af92e51e36f",
        size: 2_176,
    },
    VbxPldaArtifact {
        id: "vbx_plda_mean2",
        filename: "plda_mean2.npy",
        sha256: "6f6fb708a2037197b5b84ffeaa8f140cb878088fbecd6ab042ad26a7691bd2cf",
        size: 640,
    },
    VbxPldaArtifact {
        id: "vbx_plda_lda",
        filename: "plda_lda.npy",
        sha256: "e20c9b012bebd1aabda5a38a127e63a43cf35debdc502715fc143e2fb6bc3c4b",
        size: 131_200,
    },
    VbxPldaArtifact {
        id: "vbx_plda_mu",
        filename: "plda_mu.npy",
        sha256: "d286d48acf99bbc1ed1502fed0a3e361ae5626ce1870c8be9f7397c5e47886c6",
        size: 1_152,
    },
];

/// Manifest model ids only (same order as [`VBX_PLDA_ARTIFACTS`]).
pub const VBX_PLDA_MODEL_IDS: &[&str] = &[
    VBX_PLDA_ARTIFACTS[0].id,
    VBX_PLDA_ARTIFACTS[1].id,
    VBX_PLDA_ARTIFACTS[2].id,
    VBX_PLDA_ARTIFACTS[3].id,
    VBX_PLDA_ARTIFACTS[4].id,
    VBX_PLDA_ARTIFACTS[5].id,
];

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
    #[error(
        "model '{model_id}' has no signature in the manifest — release builds require a \
         minisign signature for every profile-resolved model (a manifest that drops the \
         signature would otherwise silently downgrade authenticity to a self-consistent hash)"
    )]
    UnsignedModel { model_id: String },
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
    /// When true (the default in release builds), profile resolution refuses
    /// manifest entries without a minisign signature (`UnsignedModel`). Debug
    /// builds stay lenient so local fixtures don't need signatures.
    require_signatures: bool,
}

/// Signature presence is enforced for profile-resolved models in release
/// builds; debug builds keep the lenient transition behavior.
const REQUIRE_SIGNATURES_DEFAULT: bool = cfg!(not(debug_assertions));

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
            require_signatures: REQUIRE_SIGNATURES_DEFAULT,
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

    /// Test-only: force the signature-presence strictness regardless of build
    /// profile, so both the strict and lenient paths are testable in debug.
    #[cfg(test)]
    pub fn with_require_signatures(mut self, require: bool) -> Self {
        self.require_signatures = require;
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
            require_signatures: REQUIRE_SIGNATURES_DEFAULT,
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

    /// Ensure all six VBx PLDA `.npy` files are present in the cache (SHA-256
    /// verified, optionally minisign-verified when signatures land). Returns the
    /// directory that holds them — pass it to
    /// [`crate::clusterer::vbx::VbxClusterer::from_dir`].
    ///
    /// Files land next to other cached models under [`Self::cache_dir`]; their
    /// filenames match the `PldaModel::from_dir` set (`plda_transform.npy`, …).
    pub fn ensure_vbx_plda_dir(&self) -> Result<PathBuf, RegistryError> {
        for id in VBX_PLDA_MODEL_IDS {
            self.ensure(id)?;
        }
        Ok(self.cache_dir.clone())
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

    /// Enforce signature presence for a profile-resolved model when strict mode
    /// is on. Runs BEFORE any network access, so a tampered manifest that drops
    /// a signature fails fast instead of downloading. Ad-hoc single-model
    /// `ensure` stays lenient by design (dev/test convenience); only profile
    /// resolution is strict.
    fn require_signature_for(&self, model_id: &str) -> Result<(), RegistryError> {
        if !self.require_signatures {
            return Ok(());
        }
        let entry = self
            .manifest
            .model(model_id)
            .ok_or_else(|| RegistryError::ModelNotFound {
                model_id: model_id.to_owned(),
            })?;
        if entry.signature.is_none() {
            return Err(RegistryError::UnsignedModel {
                model_id: model_id.to_owned(),
            });
        }
        Ok(())
    }

    /// { true }
    /// `pub fn ensure_for_profile(&self, profile: Profile) -> Result<ProfileModels, RegistryError>`
    /// { ret.as_ref().map_or(true, |p| p.segmenter_path.exists() && p.embedder_path.exists()) }
    /// Resolve all models for a profile, downloading any that are missing.
    /// In release builds every profile-resolved model must carry a manifest
    /// signature (`UnsignedModel` otherwise); all bundled models are signed.
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
        self.require_signature_for(&prof.segmenter)?;
        self.require_signature_for(&prof.embedder)?;
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
        // Mirror ensure_for_profile's strictness so the offline test path can
        // exercise both modes without network access.
        self.require_signature_for(&prof.segmenter)?;
        self.require_signature_for(&prof.embedder)?;
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
    use std::path::Path;
    use tempfile::TempDir;

    #[test]
    fn embedded_manifest_parses() {
        // This will panic if the bundled manifest.toml is malformed.
        let m = default_manifest();
        assert!(
            is_supported_schema(&m.schema),
            "embedded schema must be v1 or v2, got {}",
            m.schema
        );
        assert!(m.profiles.contains_key("mobile"));
        assert!(m.profiles.contains_key("balanced"));
    }

    #[test]
    fn embedded_manifest_is_v2_with_adapter_metadata() {
        let m = default_manifest();
        assert_eq!(m.schema, SCHEMA_V2);
        // Every shipped model carries adapter_type + license (schema v2).
        for (id, entry) in &m.models {
            assert!(
                entry.adapter_type.is_some(),
                "model '{id}' missing adapter_type"
            );
            assert!(entry.license.is_some(), "model '{id}' missing license");
            assert!(entry.version.is_some(), "model '{id}' missing version");
        }
        // latest aliases resolve to pinned model ids.
        assert_eq!(
            m.resolve_model_ref("segmenter", "latest"),
            Some("powerset_fp32")
        );
        assert_eq!(
            m.resolve_model_ref("embedder", "latest"),
            Some("wespeaker_resnet34")
        );
        assert_eq!(m.resolve_model_ref("vad", "latest"), Some("silero_vad"));
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
            .with_manifest_override(manifest)
            // TINY_MANIFEST is unsigned; pin the lenient mode so this lookup
            // test also passes under `cargo test --release`.
            .with_require_signatures(false);

        std::fs::write(tmp.path().join("hello.bin"), b"hello").unwrap();

        let bundle = r.ensure_in_cache_only_for_profile(Profile::Mobile).unwrap();
        assert_eq!(bundle.segmenter_path, tmp.path().join("hello.bin"));
        assert_eq!(bundle.embedder_path, tmp.path().join("hello.bin"));
    }

    /// Signed variant of TINY_MANIFEST — the signature value only needs to be
    /// present for the strictness check (cryptographic verification happens on
    /// the download path, not here).
    const TINY_MANIFEST_SIGNED: &str = r#"
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
        signature = "untrusted comment: fixture\nRWQfixturesignature"
    "#;

    #[test]
    fn strict_profile_resolution_rejects_unsigned_model() {
        let tmp = TempDir::new().unwrap();
        let manifest =
            Manifest::from_toml_str(crate::models::tests_helpers::TINY_MANIFEST).unwrap();
        let r = ModelRegistry::with_cache_dir(tmp.path())
            .unwrap()
            .with_manifest_override(manifest)
            .with_require_signatures(true);

        // Fails before any network/cache access — both profile paths agree.
        let err = r.ensure_for_profile(Profile::Mobile).expect_err("unsigned");
        assert!(
            matches!(err, RegistryError::UnsignedModel { ref model_id } if model_id == "hello_model")
        );
        let err = r
            .ensure_in_cache_only_for_profile(Profile::Mobile)
            .expect_err("unsigned");
        assert!(matches!(err, RegistryError::UnsignedModel { .. }));
    }

    #[test]
    fn strict_profile_resolution_accepts_signed_model() {
        let tmp = TempDir::new().unwrap();
        let manifest = Manifest::from_toml_str(TINY_MANIFEST_SIGNED).unwrap();
        let r = ModelRegistry::with_cache_dir(tmp.path())
            .unwrap()
            .with_manifest_override(manifest)
            .with_require_signatures(true);

        std::fs::write(tmp.path().join("hello.bin"), b"hello").unwrap();
        let bundle = r.ensure_in_cache_only_for_profile(Profile::Mobile).unwrap();
        assert_eq!(bundle.segmenter_path, tmp.path().join("hello.bin"));
    }

    #[test]
    fn every_profile_model_is_signed() {
        // Profile resolution in release builds requires signatures. Optional
        // models (e.g. sortformer_v2) are not profile-resolved and may ship
        // with SHA-256-only integrity until a signed release artifact exists.
        let m = default_manifest();
        for (profile_id, prof) in &m.profiles {
            for model_id in [&prof.segmenter, &prof.embedder] {
                let entry = m.models.get(model_id).unwrap_or_else(|| {
                    panic!("profile '{profile_id}' references missing model '{model_id}'")
                });
                assert!(
                    entry.signature.is_some(),
                    "profile model '{model_id}' (via '{profile_id}') has no signature — \
                     release profile resolution would fail"
                );
            }
        }
    }

    #[test]
    fn optional_sortformer_entry_present_but_not_in_profiles() {
        let m = default_manifest();
        let entry = m.model("sortformer_v2").expect("sortformer_v2 in manifest");
        assert_eq!(entry.adapter_type.as_deref(), Some("sortformer-v2"));
        assert_eq!(entry.license.as_deref(), Some("CC-BY-4.0"));
        assert_eq!(entry.num_speakers, Some(4));
        // Must never be a default profile target (opt-in download only).
        for (pid, prof) in &m.profiles {
            assert_ne!(
                prof.segmenter, "sortformer_v2",
                "profile {pid} must not pull sortformer as segmenter"
            );
            assert_ne!(
                prof.embedder, "sortformer_v2",
                "profile {pid} must not pull sortformer as embedder"
            );
        }
    }

    #[test]
    fn optional_vbx_plda_entries_present_but_not_in_profiles() {
        let m = default_manifest();
        assert_eq!(VBX_PLDA_MODEL_IDS.len(), VBX_PLDA_ARTIFACTS.len());
        for (art, listed_id) in VBX_PLDA_ARTIFACTS.iter().zip(VBX_PLDA_MODEL_IDS.iter()) {
            assert_eq!(art.id, *listed_id);
            let entry = m
                .model(art.id)
                .unwrap_or_else(|| panic!("missing manifest entry {}", art.id));
            assert_eq!(entry.sha256, art.sha256, "{} sha256 mismatch", art.id);
            assert_eq!(entry.size, Some(art.size), "{} size mismatch", art.id);
            assert_eq!(entry.filename, art.filename, "{} filename mismatch", art.id);
            assert_eq!(entry.adapter_type.as_deref(), Some("vbx-plda"));
            assert_eq!(entry.license.as_deref(), Some("CC-BY-4.0"));
            // Optional path: no minisign until a human signs the release assets.
            assert!(
                entry.signature.is_none(),
                "{} must stay unsigned until a release engineer signs it \
                 (profile resolution never pulls these)",
                art.id
            );
            assert!(
                entry.url.starts_with("https://"),
                "{} url must be https",
                art.id
            );
            for (pid, prof) in &m.profiles {
                assert_ne!(
                    prof.segmenter.as_str(),
                    art.id,
                    "profile {pid} must not pull PLDA as segmenter"
                );
                assert_ne!(
                    prof.embedder.as_str(),
                    art.id,
                    "profile {pid} must not pull PLDA as embedder"
                );
            }
        }
    }

    #[test]
    fn ensure_vbx_plda_dir_uses_local_cache_without_network() {
        // Pre-place the six fixture files under a temp cache; ensure must
        // treat them as cache hits (hash match) and never touch the network.
        let tmp = TempDir::new().unwrap();
        let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/vbx-plda");
        for art in VBX_PLDA_ARTIFACTS {
            let src = fixture_dir.join(art.filename);
            assert!(
                src.is_file(),
                "fixture missing: {} (run scripts/build-vbx-plda.py)",
                src.display()
            );
            std::fs::copy(&src, tmp.path().join(art.filename)).unwrap();
        }
        let r = ModelRegistry::with_cache_dir(tmp.path()).unwrap();
        let dir = r
            .ensure_vbx_plda_dir()
            .expect("cache-hit ensure must succeed offline");
        assert_eq!(dir, tmp.path());
        for art in VBX_PLDA_ARTIFACTS {
            assert!(dir.join(art.filename).is_file());
        }
    }
}
