//! Env overlay for [`super::vbx::VbxClustererConfig::from_env`].
//!
//! Kept beside the variational loop so tuning knobs are not mixed into the
//! inference math. Production construction does not call this.

use super::vbx::{VbxClustererConfig, VbxConfig};

pub(super) fn overlay(d: VbxClustererConfig) -> VbxClustererConfig {
    fn parse_or<T: std::str::FromStr>(name: &str, fallback: T) -> T {
        std::env::var(name)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(fallback)
    }
    fn parse_flag(name: &str, fallback: bool) -> bool {
        match std::env::var(name) {
            Ok(s)
                if s == "1" || s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("yes") =>
            {
                true
            }
            Ok(s)
                if s == "0" || s.eq_ignore_ascii_case("false") || s.eq_ignore_ascii_case("no") =>
            {
                false
            }
            _ => fallback,
        }
    }
    VbxClustererConfig {
        vbx: VbxConfig {
            fa: parse_or("POLYVOICE_VBX_FA", d.vbx.fa),
            fb: parse_or("POLYVOICE_VBX_FB", d.vbx.fb),
            loop_prob: parse_or("POLYVOICE_VBX_LOOP_PROB", d.vbx.loop_prob),
            ..d.vbx
        },
        ahc_threshold: parse_or("POLYVOICE_VBX_AHC_THRESHOLD", d.ahc_threshold),
        emb_scale: parse_or("POLYVOICE_VBX_EMB_SCALE", d.emb_scale),
        min_embedding_secs: parse_or("POLYVOICE_VBX_MIN_EMB_SECS", d.min_embedding_secs),
        ahc_established_min_members: parse_or(
            "POLYVOICE_VBX_AHC_ASC_MEMBERS",
            d.ahc_established_min_members,
        ),
        ahc_on_raw_l2: parse_flag("POLYVOICE_VBX_AHC_RAW_L2", d.ahc_on_raw_l2),
        soft_reassign: parse_flag("POLYVOICE_VBX_SOFT_REASSIGN", d.soft_reassign),
    }
}
