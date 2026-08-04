//! Speaker identity and remapping tables.
#![allow(deprecated)] // doc-links to soft-deprecated SpeakerCluster

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

/// Opaque identifier for a speaker cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpeakerId(pub u32);

/// A remapping table produced by [`SpeakerCluster::merge`](crate::cluster::SpeakerCluster::merge).
///
/// When two speaker centroids are merged, all indices after the removed one shift
/// left by one. This struct captures the old → new mapping so that callers can
/// update any stored [`SpeakerId`]s (e.g. in segments or speaker turns).
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerIdRemap {
    /// Mapping from old SpeakerId to new SpeakerId.
    mapping: Vec<(SpeakerId, SpeakerId)>,
}

impl SpeakerIdRemap {
    /// Create a remap from a raw vector of (old, new) pairs.
    ///
    /// { true }
    /// `fn from_mapping(mapping: Vec<(SpeakerId, SpeakerId)>) -> Option<Self>`
    /// { ret.is_some() == (mapping.iter().map(|(old, _)| old).collect::<HashSet<_>>().len() == mapping.len()) }
    pub fn from_mapping(mapping: Vec<(SpeakerId, SpeakerId)>) -> Option<Self> {
        let mut seen = HashSet::with_capacity(mapping.len());
        for (old, _) in &mapping {
            if !seen.insert(old) {
                return None;
            }
        }
        Some(Self { mapping })
    }

    /// { true }
    /// pub fn remap(&self, id: SpeakerId) -> SpeakerId
    /// { ret == self.mapping.iter().find(|(old, _)| *old == id).map(|(_, new)| *new).unwrap_or(id) }
    /// Apply the remap to a single [`SpeakerId`].
    ///
    /// Returns the new ID if the old ID was remapped, otherwise returns `id` unchanged.
    pub fn remap(&self, id: SpeakerId) -> SpeakerId {
        self.mapping
            .iter()
            .find(|(old, _)| *old == id)
            .map(|(_, new)| *new)
            .unwrap_or(id)
    }

    /// { true }
    /// pub fn is_empty(&self) -> bool
    /// { ret == (self.mapping.len() == 0) }
    /// Returns true if no IDs were changed.
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// { true }
    /// pub fn len(&self) -> usize
    /// { ret == self.mapping.len() }
    /// Returns the number of remapped IDs.
    pub fn len(&self) -> usize {
        self.mapping.len()
    }
}

impl fmt::Display for SpeakerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SPEAKER_{:02}", self.0)
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_pads_to_two_digits() {
        assert_eq!(SpeakerId(0).to_string(), "SPEAKER_00");
        assert_eq!(SpeakerId(9).to_string(), "SPEAKER_09");
        assert_eq!(SpeakerId(10).to_string(), "SPEAKER_10");
        assert_eq!(SpeakerId(123).to_string(), "SPEAKER_123");
    }

    #[test]
    fn from_mapping_rejects_duplicate_old_ids() {
        let dup = SpeakerIdRemap::from_mapping(vec![
            (SpeakerId(0), SpeakerId(1)),
            (SpeakerId(0), SpeakerId(2)),
        ]);
        assert!(dup.is_none());
    }

    #[test]
    fn remap_applies_known_ids_and_passes_through_unknown() {
        let remap = SpeakerIdRemap::from_mapping(vec![(SpeakerId(2), SpeakerId(0))]).unwrap();
        assert_eq!(remap.len(), 1);
        assert!(!remap.is_empty());
        assert_eq!(remap.remap(SpeakerId(2)), SpeakerId(0));
        assert_eq!(remap.remap(SpeakerId(7)), SpeakerId(7));
    }

    #[test]
    fn empty_remap_is_identity() {
        let remap = SpeakerIdRemap::from_mapping(vec![]).unwrap();
        assert!(remap.is_empty());
        assert_eq!(remap.len(), 0);
        assert_eq!(remap.remap(SpeakerId(3)), SpeakerId(3));
    }

    #[test]
    fn speaker_id_serde_roundtrip() {
        let id = SpeakerId(42);
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(serde_json::from_str::<SpeakerId>(&json).unwrap(), id);
    }
}
