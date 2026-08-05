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
    /// Merge threshold for fixed-threshold AHC, on the active scorer's scale
    /// (raw cosine in `[-1, 1]`, or AS-norm z-scores when AS-norm is enabled).
    pub ahc_threshold: f32,
    /// Cohort size (top-N) for AS-norm normalization stats in this domain.
    pub as_norm_top_n: usize,
}

/// VoxConverse: the shipped default. The threshold matches the pipeline's
/// long-standing AHC default; AS-norm z-scale thresholds are calibrated
/// separately before being frozen here.
pub const VOXCONVERSE: DomainProfile = DomainProfile {
    name: "voxconverse",
    ahc_threshold: 0.5,
    as_norm_top_n: 100,
};

/// AMI (meetings, distant microphones).
// PLACEHOLDER: not yet calibrated — must be tuned on the AMI dev split before
// being trusted. Distinct from the VoxConverse value so profile selection is
// observable; do not treat 0.55 as a tuned number.
pub const AMI: DomainProfile = DomainProfile {
    name: "ami",
    ahc_threshold: 0.55,
    as_norm_top_n: 100,
};

/// CALLHOME (telephone speech).
// PLACEHOLDER: not yet calibrated — must be tuned on the CALLHOME dev split
// before being trusted. Distinct from the other profiles so profile selection
// is observable; do not treat 0.45 as a tuned number.
pub const CALLHOME: DomainProfile = DomainProfile {
    name: "callhome",
    ahc_threshold: 0.45,
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
