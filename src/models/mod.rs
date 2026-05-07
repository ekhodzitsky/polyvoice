//! Model registry — manifest-driven downloads with SHA-256 verification.
//!
//! Added in v0.6 (M0). See `docs/superpowers/specs/2026-05-07-perfect-diarization-roadmap-v1-design.md` §6.

pub mod manifest;
pub use manifest::{Manifest, ManifestError, ModelEntry, ProfileEntry, SCHEMA_V1};
