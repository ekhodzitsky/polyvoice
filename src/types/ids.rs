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
