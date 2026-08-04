//! Optional NVIDIA Streaming Sortformer v2 end-to-end diarizer.
//!
//! # Feature gate
//!
//! Enabled only with `--features sortformer` (implies `onnx`). Core builds
//! without this feature never reference Sortformer.
//!
//! # Model weights
//!
//! Weights are **never bundled**. Download on demand via the model registry:
//!
//! ```ignore
//! // requires features: sortformer, download
//! let path = ModelRegistry::default()?.ensure("sortformer_v2")?;
//! let mut diarizer = SortformerDiarizer::from_path(path)?;
//! let turns = diarizer.diarize(&audio_16k_mono)?;
//! ```
//!
//! Community ONNX (CC-BY-4.0): `cgus/diar_streaming_sortformer_4spk-v2-onnx`
//! on Hugging Face, converted for the MIT [parakeet-rs](https://github.com/altunenes/parakeet-rs)
//! project. Upstream training checkpoint: `nvidia/diar_streaming_sortformer_4spk-v2`.
//!
//! # Hard speaker cap
//!
//! The model has four sigmoid heads — `max_speakers > 4` is a config error.
//! Prefer the VBx clusterer path for meetings with more than four speakers.
//! See `docs/sortformer.md` for licensing and when to choose which backend.
//!
//! # Streaming state
//!
//! Between chunk calls the adapter keeps FIFO and speaker-cache tensors and
//! feeds them back as named ONNX inputs (parakeet-rs / NeMo pattern). Call
//! [`SortformerDiarizer::reset`](crate::sortformer::SortformerDiarizer::reset)
//! between independent recordings.

mod config;
mod diarizer;
mod features;

pub use config::{
    ADAPTER_TYPE, FRAME_DURATION_SECS, MAX_SPEAKERS, MODEL_ID, PostProcessConfig, SAMPLE_RATE,
    SortformerConfig, SortformerError,
};
pub use diarizer::SortformerDiarizer;

#[cfg(feature = "download")]
use crate::models::{AdapterError, AdapterFactory, AdapterRegistry, AdapterStage, BuiltinAdapter};
#[cfg(feature = "download")]
use std::sync::Arc;

/// Register the Sortformer v2 adapter type with an [`AdapterRegistry`].
///
/// Safe to call once; returns [`AdapterError::AlreadyRegistered`] if the id
/// is already present (e.g. after [`AdapterRegistry::with_builtins`] when the
/// `sortformer` feature is on).
#[cfg(feature = "download")]
pub fn register_with(registry: &mut AdapterRegistry) -> Result<(), AdapterError> {
    let factory: AdapterFactory = Arc::new(|| {
        Box::new(BuiltinAdapter {
            stage: AdapterStage::Diarizer,
            id: ADAPTER_TYPE.to_owned(),
        })
    });
    registry.register(AdapterStage::Diarizer, ADAPTER_TYPE, factory)?;
    let _ = registry.register_alias(AdapterStage::Diarizer, "latest", ADAPTER_TYPE);
    let _ = registry.register_alias(AdapterStage::Diarizer, "v2", ADAPTER_TYPE);
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn adapter_type_constant_matches_manifest_convention() {
        assert_eq!(ADAPTER_TYPE, "sortformer-v2");
        assert_eq!(MODEL_ID, "sortformer_v2");
        assert_eq!(MAX_SPEAKERS, 4);
    }

    #[cfg(feature = "download")]
    #[test]
    fn register_with_empty_registry() {
        let mut reg = AdapterRegistry::new();
        register_with(&mut reg).unwrap();
        assert!(reg.contains(AdapterStage::Diarizer, ADAPTER_TYPE));
        let err = register_with(&mut reg).expect_err("duplicate");
        assert!(matches!(err, AdapterError::AlreadyRegistered { .. }));
    }

    #[cfg(feature = "download")]
    #[test]
    fn register_aliases_resolve_to_adapter_type() {
        let mut reg = AdapterRegistry::new();
        register_with(&mut reg).unwrap();
        for alias in ["latest", "v2"] {
            let resolved = reg.resolve(AdapterStage::Diarizer, alias).unwrap();
            assert_eq!(resolved, ADAPTER_TYPE, "alias {alias}");
        }
        // The direct id also resolves, and the factory produces a handle.
        let resolved = reg.resolve(AdapterStage::Diarizer, ADAPTER_TYPE).unwrap();
        assert_eq!(resolved, ADAPTER_TYPE);
        let handle = reg.create(AdapterStage::Diarizer, "v2").unwrap();
        let adapter = handle.downcast::<BuiltinAdapter>().unwrap();
        assert_eq!(adapter.id, ADAPTER_TYPE);
    }
}
