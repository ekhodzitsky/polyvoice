//! Adapter registry — select segmentation / embedder / clusterer / scoring /
//! VAD implementations by config string, with a public registration API.
//!
//! This is intentionally type-erased (`Box<dyn Any>`): each stage has a
//! different trait surface (`Segmenter`, `Embedder`, `Clusterer`, …). Callers
//! downcast after `create`, and external adapters register factories the same
//! way built-ins do. Unknown `adapter_type` returns a descriptive error —
//! never panics.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Stage of the diarization pipeline an adapter serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdapterStage {
    Segmentation,
    Embedder,
    Clusterer,
    Scoring,
    Vad,
    /// End-to-end diarizer (e.g. Sortformer) that replaces segmenter +
    /// embedder + clusterer with a single model.
    Diarizer,
}

impl AdapterStage {
    /// Stable lowercase id used in logs and config strings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Segmentation => "segmentation",
            Self::Embedder => "embedder",
            Self::Clusterer => "clusterer",
            Self::Scoring => "scoring",
            Self::Vad => "vad",
            Self::Diarizer => "diarizer",
        }
    }

    /// Parse a stage name (case-insensitive). Accepts both short and long forms
    /// (`"vad"` / `"segmentation"`).
    ///
    /// Thin compatibility wrapper over [`FromStr`](std::str::FromStr); new code
    /// should prefer `s.parse()` or `AdapterStage::from_str` for a typed error.
    pub fn parse(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl std::str::FromStr for AdapterStage {
    type Err = AdapterError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "segmentation" | "segmenter" => Ok(Self::Segmentation),
            "embedder" | "embedding" => Ok(Self::Embedder),
            "clusterer" | "clustering" => Ok(Self::Clusterer),
            "scoring" | "score" => Ok(Self::Scoring),
            "vad" => Ok(Self::Vad),
            "diarizer" | "e2e" | "e2e-diarizer" => Ok(Self::Diarizer),
            _ => Err(AdapterError::InvalidStage(s.to_owned())),
        }
    }
}

impl fmt::Display for AdapterStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors from [`AdapterRegistry`] operations.
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("unknown adapter type '{adapter_type}' for stage {stage}")]
    UnknownAdapter {
        stage: AdapterStage,
        adapter_type: String,
    },
    #[error("adapter type '{adapter_type}' is already registered for stage {stage}")]
    AlreadyRegistered {
        stage: AdapterStage,
        adapter_type: String,
    },
    #[error("unknown version alias '{alias}' for stage {stage}")]
    UnknownAlias { stage: AdapterStage, alias: String },
    #[error("alias '{alias}' for stage {stage} targets unregistered adapter '{target}'")]
    AliasTargetMissing {
        stage: AdapterStage,
        alias: String,
        target: String,
    },
    #[error("invalid adapter stage name '{0}'")]
    InvalidStage(String),
}

/// Factory that builds a type-erased adapter instance.
///
/// External (test / third-party) adapters register one of these via
/// [`AdapterRegistry::register`]. The returned `Box<dyn Any + Send + Sync>` is
/// downcast by the caller to the stage trait object they need.
pub type AdapterFactory = Arc<dyn Fn() -> Box<dyn Any + Send + Sync> + Send + Sync>;

/// Registry of named adapter factories, one map per [`AdapterStage`].
///
/// Built-in adapter type ids match the `adapter_type` field of schema-v2
/// model entries (e.g. `"powerset-v1"`, `"silero"`, `"wespeaker-resnet34"`,
/// `"ahc"`, `"vbx"`, `"plda"`, `"cosine"`). Version aliases such as `"latest"`
/// resolve to a pinned id and are logged for DER reproducibility.
#[derive(Clone, Default)]
pub struct AdapterRegistry {
    factories: HashMap<(AdapterStage, String), AdapterFactory>,
    /// `(stage, alias)` → pinned adapter type id.
    aliases: HashMap<(AdapterStage, String), String>,
}

impl AdapterRegistry {
    /// Empty registry — no built-ins. Prefer [`Self::with_builtins`] for the
    /// production set of known type names.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry pre-loaded with the shipped adapter type ids.
    ///
    /// Factories are **name markers** only: they return a [`BuiltinAdapter`]
    /// handle recording the type id. Concrete ONNX construction stays in the
    /// existing stage modules / pipeline builder so this registry never pulls
    /// model files or changes DER behaviour.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        // Segmentation
        reg.register_builtin(AdapterStage::Segmentation, "powerset-v1");
        // Embedders
        reg.register_builtin(AdapterStage::Embedder, "wespeaker-resnet34");
        reg.register_builtin(AdapterStage::Embedder, "cam++");
        // Optional domain / short-segment embedders (weights not profile-default).
        reg.register_builtin(AdapterStage::Embedder, "eres2netv2");
        reg.register_builtin(AdapterStage::Embedder, "cam++-zh");
        let _ = reg.register_alias(AdapterStage::Embedder, "eres2net-v2", "eres2netv2");
        let _ = reg.register_alias(AdapterStage::Embedder, "campplus-zh", "cam++-zh");
        // Clusterers
        reg.register_builtin(AdapterStage::Clusterer, "ahc");
        reg.register_builtin(AdapterStage::Clusterer, "nme-sc");
        reg.register_builtin(AdapterStage::Clusterer, "vbx");
        // Scoring
        reg.register_builtin(AdapterStage::Scoring, "cosine");
        reg.register_builtin(AdapterStage::Scoring, "plda");
        // VAD
        reg.register_builtin(AdapterStage::Vad, "silero");
        reg.register_builtin(AdapterStage::Vad, "energy");
        // Optional pure-Rust earshot VAD (feature-gated). Name marker only —
        // concrete construction lives in `crate::earshot_vad`. Never the
        // default: aliases still resolve to silero.
        #[cfg(feature = "vad-earshot")]
        reg.register_builtin(AdapterStage::Vad, "earshot");

        // Version aliases → pinned implementations (Deepgram-style latest|v1).
        let _ = reg.register_alias(AdapterStage::Segmentation, "latest", "powerset-v1");
        let _ = reg.register_alias(AdapterStage::Segmentation, "v1", "powerset-v1");
        let _ = reg.register_alias(AdapterStage::Embedder, "latest", "wespeaker-resnet34");
        let _ = reg.register_alias(AdapterStage::Embedder, "v1", "wespeaker-resnet34");
        let _ = reg.register_alias(AdapterStage::Clusterer, "latest", "ahc");
        let _ = reg.register_alias(AdapterStage::Scoring, "latest", "cosine");
        let _ = reg.register_alias(AdapterStage::Vad, "latest", "silero");
        let _ = reg.register_alias(AdapterStage::Vad, "v1", "silero");

        // Optional E2E Sortformer diarizer (feature-gated). Name marker only —
        // concrete construction lives in `crate::sortformer`.
        #[cfg(feature = "sortformer")]
        {
            reg.register_builtin(AdapterStage::Diarizer, "sortformer-v2");
            let _ = reg.register_alias(AdapterStage::Diarizer, "latest", "sortformer-v2");
            let _ = reg.register_alias(AdapterStage::Diarizer, "v2", "sortformer-v2");
        }
        reg
    }

    fn register_builtin(&mut self, stage: AdapterStage, id: &str) {
        let id_owned = id.to_owned();
        let factory: AdapterFactory = Arc::new(move || {
            Box::new(BuiltinAdapter {
                stage,
                id: id_owned.clone(),
            })
        });
        // Built-ins never collide with themselves; ignore AlreadyRegistered.
        let _ = self.register(stage, id, factory);
    }

    /// Register an external adapter factory under `adapter_type` for `stage`.
    ///
    /// Returns [`AdapterError::AlreadyRegistered`] if the id is taken — use a
    /// distinct name or call [`Self::register_or_replace`].
    pub fn register(
        &mut self,
        stage: AdapterStage,
        adapter_type: impl Into<String>,
        factory: AdapterFactory,
    ) -> Result<(), AdapterError> {
        let adapter_type = adapter_type.into();
        let key = (stage, adapter_type.clone());
        if self.factories.contains_key(&key) {
            return Err(AdapterError::AlreadyRegistered {
                stage,
                adapter_type,
            });
        }
        self.factories.insert(key, factory);
        Ok(())
    }

    /// Like [`Self::register`] but overwrites an existing entry.
    pub fn register_or_replace(
        &mut self,
        stage: AdapterStage,
        adapter_type: impl Into<String>,
        factory: AdapterFactory,
    ) {
        let adapter_type = adapter_type.into();
        self.factories.insert((stage, adapter_type), factory);
    }

    /// Pin a version alias (`"latest"`, `"v1"`, …) to a registered adapter type.
    pub fn register_alias(
        &mut self,
        stage: AdapterStage,
        alias: impl Into<String>,
        target: impl Into<String>,
    ) -> Result<(), AdapterError> {
        let alias = alias.into();
        let target = target.into();
        if !self.factories.contains_key(&(stage, target.clone())) {
            return Err(AdapterError::AliasTargetMissing {
                stage,
                alias,
                target,
            });
        }
        self.aliases.insert((stage, alias), target);
        Ok(())
    }

    /// Resolve `id_or_alias` to a concrete registered adapter type id.
    ///
    /// Alias resolution is logged at `info` so DER reports can record which
    /// pin was actually used. Returns an owned `String` so both direct ids and
    /// alias targets share a single return type without lifetime coupling.
    pub fn resolve(&self, stage: AdapterStage, id_or_alias: &str) -> Result<String, AdapterError> {
        if self
            .factories
            .contains_key(&(stage, id_or_alias.to_owned()))
        {
            return Ok(id_or_alias.to_owned());
        }
        if let Some(target) = self.aliases.get(&(stage, id_or_alias.to_owned())) {
            tracing::info!(
                stage = %stage,
                alias = id_or_alias,
                resolved = target.as_str(),
                "resolved adapter version alias to pinned adapter type"
            );
            return Ok(target.clone());
        }
        Err(AdapterError::UnknownAdapter {
            stage,
            adapter_type: id_or_alias.to_owned(),
        })
    }

    /// Create an adapter instance for `id_or_alias` (alias-resolved).
    pub fn create(
        &self,
        stage: AdapterStage,
        id_or_alias: &str,
    ) -> Result<Box<dyn Any + Send + Sync>, AdapterError> {
        let id = self.resolve(stage, id_or_alias)?;
        let factory =
            self.factories
                .get(&(stage, id.clone()))
                .ok_or(AdapterError::UnknownAdapter {
                    stage,
                    adapter_type: id,
                })?;
        Ok(factory())
    }

    /// True if `adapter_type` is registered (aliases do not count).
    pub fn contains(&self, stage: AdapterStage, adapter_type: &str) -> bool {
        self.factories
            .contains_key(&(stage, adapter_type.to_owned()))
    }

    /// Number of registered adapter types (aliases excluded).
    pub fn len(&self) -> usize {
        self.factories.len()
    }

    /// True when no adapter types are registered.
    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Iterate registered `(stage, adapter_type)` pairs.
    pub fn registered(&self) -> impl Iterator<Item = (AdapterStage, &str)> + '_ {
        self.factories
            .keys()
            .map(|(stage, id)| (*stage, id.as_str()))
    }
}

/// Handle returned by built-in factories. Downstream code matches on `id` to
/// construct the real stage implementation; the registry itself stays free of
/// ONNX / feature-gated stage crates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltinAdapter {
    pub stage: AdapterStage,
    pub id: String,
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn builtins_register_all_five_stages() {
        let reg = AdapterRegistry::with_builtins();
        assert!(reg.contains(AdapterStage::Segmentation, "powerset-v1"));
        assert!(reg.contains(AdapterStage::Embedder, "wespeaker-resnet34"));
        assert!(reg.contains(AdapterStage::Clusterer, "ahc"));
        assert!(reg.contains(AdapterStage::Scoring, "cosine"));
        assert!(reg.contains(AdapterStage::Vad, "silero"));
        assert!(reg.len() >= 5);
    }

    #[test]
    fn unknown_adapter_type_returns_error() {
        let reg = AdapterRegistry::with_builtins();
        let err = reg
            .create(AdapterStage::Embedder, "sortformer-v99")
            .expect_err("must reject unknown type");
        let msg = format!("{err}");
        match err {
            AdapterError::UnknownAdapter {
                stage,
                adapter_type,
            } => {
                assert_eq!(stage, AdapterStage::Embedder);
                assert_eq!(adapter_type, "sortformer-v99");
            }
            other => panic!("unexpected error: {other}"),
        }
        // Display is human-readable (acceptance: descriptive error, not panic).
        assert!(msg.contains("sortformer-v99"));
        assert!(msg.contains("embedder"));
    }

    #[test]
    fn latest_alias_resolves_to_pinned() {
        let reg = AdapterRegistry::with_builtins();
        assert_eq!(
            reg.resolve(AdapterStage::Embedder, "latest").unwrap(),
            "wespeaker-resnet34".to_owned()
        );
        assert_eq!(
            reg.resolve(AdapterStage::Vad, "latest").unwrap(),
            "silero".to_owned()
        );
        assert_eq!(
            reg.resolve(AdapterStage::Segmentation, "v1").unwrap(),
            "powerset-v1".to_owned()
        );
    }

    #[test]
    fn external_mock_adapter_registers_and_is_selected() {
        // Acceptance (e): public register API + config-string selection.
        let mut reg = AdapterRegistry::new();
        let calls = Arc::new(Mutex::new(0_u32));
        let calls_factory = Arc::clone(&calls);
        let factory: AdapterFactory = Arc::new(move || {
            *calls_factory.lock().unwrap() += 1;
            Box::new(String::from("mock-embedder-instance"))
        });
        reg.register(AdapterStage::Embedder, "mock-embedder", factory)
            .unwrap();

        assert!(reg.contains(AdapterStage::Embedder, "mock-embedder"));
        let instance = reg.create(AdapterStage::Embedder, "mock-embedder").unwrap();
        let s = instance.downcast_ref::<String>().expect("downcast");
        assert_eq!(s, "mock-embedder-instance");
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[test]
    fn register_rejects_duplicate() {
        let mut reg = AdapterRegistry::new();
        let factory: AdapterFactory = Arc::new(|| Box::new(0_u8));
        reg.register(AdapterStage::Vad, "energy", Arc::clone(&factory))
            .unwrap();
        let err = reg
            .register(AdapterStage::Vad, "energy", factory)
            .expect_err("duplicate");
        assert!(matches!(err, AdapterError::AlreadyRegistered { .. }));
    }

    #[test]
    fn alias_to_missing_target_fails() {
        let mut reg = AdapterRegistry::new();
        let err = reg
            .register_alias(AdapterStage::Scoring, "latest", "ghost")
            .expect_err("missing target");
        assert!(matches!(err, AdapterError::AliasTargetMissing { .. }));
    }

    #[test]
    fn stage_parse_roundtrip() {
        for stage in [
            AdapterStage::Segmentation,
            AdapterStage::Embedder,
            AdapterStage::Clusterer,
            AdapterStage::Scoring,
            AdapterStage::Vad,
            AdapterStage::Diarizer,
        ] {
            assert_eq!(AdapterStage::parse(stage.as_str()), Some(stage));
            assert_eq!(
                stage.as_str().parse::<AdapterStage>().expect("roundtrip"),
                stage
            );
        }
        assert_eq!(
            AdapterStage::parse("segmenter"),
            Some(AdapterStage::Segmentation)
        );
        assert_eq!(AdapterStage::parse("e2e"), Some(AdapterStage::Diarizer));
        assert!(AdapterStage::parse("nope").is_none());
        assert!(matches!(
            "nope".parse::<AdapterStage>(),
            Err(AdapterError::InvalidStage(_))
        ));
    }

    #[cfg(feature = "sortformer")]
    #[test]
    fn sortformer_builtin_registered_when_feature_on() {
        let reg = AdapterRegistry::with_builtins();
        assert!(reg.contains(AdapterStage::Diarizer, "sortformer-v2"));
        assert_eq!(
            reg.resolve(AdapterStage::Diarizer, "latest").unwrap(),
            "sortformer-v2"
        );
        let handle = reg.create(AdapterStage::Diarizer, "sortformer-v2").unwrap();
        let builtin = handle.downcast_ref::<BuiltinAdapter>().expect("marker");
        assert_eq!(builtin.id, "sortformer-v2");
        assert_eq!(builtin.stage, AdapterStage::Diarizer);
    }

    #[cfg(feature = "vad-earshot")]
    #[test]
    fn earshot_builtin_registered_when_feature_on_default_stays_silero() {
        let reg = AdapterRegistry::with_builtins();
        assert!(reg.contains(AdapterStage::Vad, "earshot"));
        // Aliases must still pin silero — earshot is opt-in only.
        assert_eq!(reg.resolve(AdapterStage::Vad, "latest").unwrap(), "silero");
        assert_eq!(reg.resolve(AdapterStage::Vad, "v1").unwrap(), "silero");
        let handle = reg.create(AdapterStage::Vad, "earshot").unwrap();
        let builtin = handle.downcast_ref::<BuiltinAdapter>().expect("marker");
        assert_eq!(builtin.id, "earshot");
        assert_eq!(builtin.stage, AdapterStage::Vad);
    }
}
