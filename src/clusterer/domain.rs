//! Per-domain scoring profiles: calibrated AHC merge thresholds and AS-norm
//! cohort sizes per evaluation domain.
//!
//! Profiles are **data, not code branching**: selecting a domain only swaps
//! the threshold/top-N values the AHC clusterer is built with. The default is
//! VoxConverse — the domain the shipped pipeline threshold was tuned on.

/// A calibrated scoring profile for one evaluation domain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DomainProfile {
    /// Stable lowercase name (CLI value): `voxconverse`, `ami`, `callhome`.
    pub name: &'static str,
    /// Merge threshold for fixed-threshold AHC on raw cosine scores in
    /// `[-1, 1]`.
    pub ahc_threshold: f32,
    /// Merge threshold on the AS-norm z-score scale, used only when AS-norm
    /// is enabled. `None` = not calibrated for this domain.
    pub as_norm_threshold: Option<f32>,
    /// Cohort size (top-N) for AS-norm normalization stats in this domain.
    pub as_norm_top_n: usize,
}

/// VoxConverse: the shipped default. The raw threshold is the shipped CLI
/// default and the dev-sweep optimum (10.85% DER @ 0.45 vs 11.11% @ 0.50 on
/// the 30-file VoxConverse-dev sweep, no-collar micro). The z-threshold comes
/// from the same sweep (z=3 10.31%, z=4 10.40%), confirmed on
/// VoxConverse-test (z=4: 22.98% vs raw 24.14% no-collar micro DER).
pub const VOXCONVERSE: DomainProfile = DomainProfile {
    name: "voxconverse",
    ahc_threshold: 0.45,
    as_norm_threshold: Some(4.0),
    as_norm_top_n: 100,
};

/// AMI (meetings, distant microphones).
// PLACEHOLDER raw threshold: no AMI dev split is available locally, so the
// raw value is not dev-calibrated. The z-threshold (z=5: 28.68% vs raw 30.77%
// no-collar micro DER) was measured on ami-test and must be re-derived from a
// proper dev split before being trusted — treat it as directional only.
pub const AMI: DomainProfile = DomainProfile {
    name: "ami",
    ahc_threshold: 0.55,
    as_norm_threshold: Some(5.0),
    as_norm_top_n: 100,
};

/// CALLHOME (telephone speech).
// PLACEHOLDER: no CALLHOME data is available locally at all — neither value
// is calibrated. Kept distinct from the other profiles so profile selection
// stays observable; do not treat these as tuned numbers.
pub const CALLHOME: DomainProfile = DomainProfile {
    name: "callhome",
    ahc_threshold: 0.5,
    as_norm_threshold: None,
    as_norm_top_n: 100,
};

/// All shipped domain profiles.
pub const DOMAIN_PROFILES: &[DomainProfile] = &[VOXCONVERSE, AMI, CALLHOME];

/// The default domain profile (VoxConverse).
pub const DEFAULT_DOMAIN_PROFILE: DomainProfile = VOXCONVERSE;

/// Look up a domain profile by name.
pub fn domain_profile(name: &str) -> Option<DomainProfile> {
    DOMAIN_PROFILES.iter().copied().find(|p| p.name == name)
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_resolves_every_shipped_profile() {
        for p in DOMAIN_PROFILES {
            assert_eq!(domain_profile(p.name), Some(*p), "{}", p.name);
        }
    }

    #[test]
    fn lookup_is_deterministic_and_rejects_unknown_names() {
        assert_eq!(domain_profile("ami"), domain_profile("ami"));
        assert_eq!(domain_profile("voxconverse"), Some(DEFAULT_DOMAIN_PROFILE));
        for bogus in ["", "AMI", "vox", "switchboard", "voxconverse-test"] {
            assert_eq!(domain_profile(bogus), None, "{bogus}");
        }
    }

    #[test]
    fn profiles_carry_distinct_thresholds() {
        // Profile selection must observably change the effective threshold;
        // identical placeholders would hide a wiring bug.
        let v = VOXCONVERSE.ahc_threshold;
        let a = AMI.ahc_threshold;
        let c = CALLHOME.ahc_threshold;
        assert_ne!(v, a);
        assert_ne!(v, c);
        assert_ne!(a, c);
    }
}
