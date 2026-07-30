//! Scalar hysteresis primitives shared by VAD region detection
//! ([`VadStateMachine`](crate::vad::VadStateMachine)) and powerset
//! binarization (`segmentation::binarize`).
//!
//! Two composable pieces:
//!
//! - [`HysteresisGate`] — per-frame onset/offset threshold memory.
//! - [`RegionTracker`] — min-off hangover over a boolean track (gaps shorter
//!   than `min_off` frames are bridged) with two closing policies, plus the
//!   min-on region filter.

/// Per-frame hysteresis gate: turns ON when the probability reaches `onset`,
/// and once ON stays ON until the probability drops below `offset`
/// (`offset <= onset` gives hysteresis; equal values give a plain single
/// threshold). A NaN probability never satisfies `>=`, so it reads as
/// inactive.
// Only constructed by `segmentation::binarize` today; VAD region detection
// compares against its single threshold directly.
#[cfg_attr(not(feature = "segmentation"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct HysteresisGate {
    onset: f32,
    offset: f32,
    on: bool,
}

#[cfg_attr(not(feature = "segmentation"), allow(dead_code))]
impl HysteresisGate {
    /// { true }
    /// `pub(crate) fn new(onset: f32, offset: f32) -> Self`
    /// { true }
    /// Create a gate in the OFF state.
    pub(crate) fn new(onset: f32, offset: f32) -> Self {
        Self {
            onset,
            offset,
            on: false,
        }
    }

    /// { true }
    /// `pub(crate) fn update(&mut self, prob: f32) -> bool`
    /// { ret == self.on }
    /// Advance by one frame probability; returns the new gate state.
    pub(crate) fn update(&mut self, prob: f32) -> bool {
        self.on = if self.on {
            prob >= self.offset
        } else {
            prob >= self.onset
        };
        self.on
    }

    /// { true }
    /// `pub(crate) fn reset(&mut self)`
    /// { !self.on }
    /// Force the gate OFF (e.g. a frame without model coverage).
    pub(crate) fn reset(&mut self) {
        self.on = false;
    }
}

/// Event emitted by [`RegionTracker`] when a region opens or closes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegionEvent {
    /// A region started at the given frame index.
    Start { start_frame: usize },
    /// A region ended. `end_frame` is exclusive.
    End {
        start_frame: usize,
        end_frame: usize,
    },
}

/// How a region's end is placed when it closes.
// `Trim` is only constructed by `segmentation::binarize` today.
#[cfg_attr(not(feature = "segmentation"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TailPolicy {
    /// The closing run of inactive frames stays inside the region, and a
    /// flush extends the region to the flush point (VAD regions end *after*
    /// the silence that closed them).
    Keep,
    /// The region rewinds to the frame after the last active one
    /// (pyannote-style gap bridging never extends a run into a trailing gap).
    Trim,
}

/// Region tracker over a boolean active/inactive track: an open region
/// survives isolated inactive frames and closes only after `min_off`
/// consecutive ones, so inactive runs shorter than `min_off` frames are
/// bridged. Closed regions shorter than `min_on` frames fail
/// [`RegionTracker::keeps`].
///
/// Events always alternate `Start`/`End`; the min-on filter never suppresses
/// an event, callers apply [`RegionTracker::keeps`] to closed regions.
#[derive(Debug, Clone)]
pub(crate) struct RegionTracker {
    min_off: usize,
    min_on: usize,
    tail: TailPolicy,
    in_region: bool,
    start_frame: usize,
    last_active_frame: usize,
    off_count: usize,
}

impl RegionTracker {
    /// { true }
    /// `pub(crate) fn new(min_off: usize, min_on: usize, tail: TailPolicy) -> Self`
    /// { true }
    /// Create a tracker with no open region.
    pub(crate) fn new(min_off: usize, min_on: usize, tail: TailPolicy) -> Self {
        Self {
            min_off,
            min_on,
            tail,
            in_region: false,
            start_frame: 0,
            last_active_frame: 0,
            off_count: 0,
        }
    }

    /// { true }
    /// `pub(crate) fn advance(&mut self, active: bool, frame: usize) -> Option<RegionEvent>`
    /// { true }
    /// Advance by one frame decision.
    ///
    /// Returns [`RegionEvent::Start`] when a region opens and
    /// [`RegionEvent::End`] when the min-off hangover expires.
    pub(crate) fn advance(&mut self, active: bool, frame: usize) -> Option<RegionEvent> {
        if self.in_region {
            if active {
                self.last_active_frame = frame;
                self.off_count = 0;
                None
            } else {
                self.off_count += 1;
                if self.off_count >= self.min_off {
                    let end_frame = match self.tail {
                        TailPolicy::Keep => frame + 1,
                        TailPolicy::Trim => self.last_active_frame + 1,
                    };
                    Some(self.close(end_frame))
                } else {
                    None
                }
            }
        } else if active {
            self.in_region = true;
            self.start_frame = frame;
            self.last_active_frame = frame;
            self.off_count = 0;
            Some(RegionEvent::Start { start_frame: frame })
        } else {
            None
        }
    }

    /// { true }
    /// `pub(crate) fn flush(&mut self, frame: usize) -> Option<RegionEvent>`
    /// { !self.in_region }
    /// Finalize any open region. `Keep` extends it to `frame`; `Trim` ends it
    /// after the last active frame.
    pub(crate) fn flush(&mut self, frame: usize) -> Option<RegionEvent> {
        if self.in_region {
            let end_frame = match self.tail {
                TailPolicy::Keep => frame,
                TailPolicy::Trim => self.last_active_frame + 1,
            };
            Some(self.close(end_frame))
        } else {
            None
        }
    }

    /// { true }
    /// `pub(crate) fn reset(&mut self) -> Option<RegionEvent>`
    /// { !self.in_region }
    /// Hard-close any open region at the frame after the last active one,
    /// without consuming a frame. Coverage holes (frames with no data) break
    /// regions instead of being bridged.
    // Only used by `segmentation::binarize` today.
    #[cfg_attr(not(feature = "segmentation"), allow(dead_code))]
    pub(crate) fn reset(&mut self) -> Option<RegionEvent> {
        if self.in_region {
            Some(self.close(self.last_active_frame + 1))
        } else {
            None
        }
    }

    /// { true }
    /// `pub(crate) fn in_region(&self) -> bool`
    /// { ret == self.in_region }
    /// Whether a region is currently open.
    pub(crate) fn in_region(&self) -> bool {
        self.in_region
    }

    /// { true }
    /// `pub(crate) fn min_on(&self) -> usize`
    /// { ret == self.min_on }
    /// Minimum region length in frames; shorter regions fail [`Self::keeps`].
    pub(crate) fn min_on(&self) -> usize {
        self.min_on
    }

    /// { true }
    /// `pub(crate) fn keeps(&self, start_frame: usize, end_frame: usize) -> bool`
    /// { ret == (end_frame - start_frame >= self.min_on) }
    /// Whether a closed `[start_frame, end_frame)` region survives the
    /// minimum-duration filter.
    pub(crate) fn keeps(&self, start_frame: usize, end_frame: usize) -> bool {
        end_frame - start_frame >= self.min_on
    }

    fn close(&mut self, end_frame: usize) -> RegionEvent {
        let event = RegionEvent::End {
            start_frame: self.start_frame,
            end_frame,
        };
        self.in_region = false;
        self.off_count = 0;
        event
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_plain_threshold_flips_immediately() {
        let mut gate = HysteresisGate::new(0.5, 0.5);
        assert!(!gate.update(0.4));
        assert!(gate.update(0.5));
        assert!(!gate.update(0.4));
    }

    #[test]
    fn gate_hysteresis_holds_through_dip() {
        let mut gate = HysteresisGate::new(0.6, 0.3);
        assert!(!gate.update(0.55)); // below onset while OFF
        assert!(gate.update(0.7));
        assert!(gate.update(0.45)); // between offset and onset: stays ON
        assert!(!gate.update(0.2)); // below offset: OFF
        assert!(!gate.update(0.45)); // still below onset while OFF
    }

    #[test]
    fn tracker_keep_bridges_short_gap_and_keeps_closing_tail() {
        let mut t = RegionTracker::new(3, 0, TailPolicy::Keep);
        assert_eq!(
            t.advance(true, 0),
            Some(RegionEvent::Start { start_frame: 0 })
        );
        assert_eq!(t.advance(false, 1), None);
        assert_eq!(t.advance(false, 2), None);
        assert_eq!(t.advance(true, 3), None); // 2-frame gap bridged
        // 3 consecutive inactive frames close the region; the tail stays in.
        assert_eq!(t.advance(false, 4), None);
        assert_eq!(t.advance(false, 5), None);
        assert_eq!(
            t.advance(false, 6),
            Some(RegionEvent::End {
                start_frame: 0,
                end_frame: 7,
            })
        );
        assert!(!t.in_region());
    }

    #[test]
    fn tracker_trim_rewinds_to_last_active_frame() {
        let mut t = RegionTracker::new(3, 0, TailPolicy::Trim);
        assert_eq!(
            t.advance(true, 0),
            Some(RegionEvent::Start { start_frame: 0 })
        );
        assert_eq!(t.advance(true, 2), None);
        assert_eq!(t.advance(false, 3), None);
        assert_eq!(t.advance(false, 4), None);
        assert_eq!(
            t.advance(false, 5),
            Some(RegionEvent::End {
                start_frame: 0,
                end_frame: 3, // last active frame was 2
            })
        );
        // Flush also trims the trailing gap.
        let mut t = RegionTracker::new(3, 0, TailPolicy::Trim);
        let _ = t.advance(true, 1);
        let _ = t.advance(false, 2);
        assert_eq!(
            t.flush(10),
            Some(RegionEvent::End {
                start_frame: 1,
                end_frame: 2,
            })
        );
    }

    #[test]
    fn tracker_keep_flush_extends_to_flush_point() {
        let mut t = RegionTracker::new(3, 0, TailPolicy::Keep);
        let _ = t.advance(true, 0);
        let _ = t.advance(false, 1);
        assert_eq!(
            t.flush(5),
            Some(RegionEvent::End {
                start_frame: 0,
                end_frame: 5,
            })
        );
        assert_eq!(t.flush(5), None, "already closed");
    }

    #[test]
    fn tracker_reset_hard_closes_at_last_active_frame() {
        let mut t = RegionTracker::new(5, 0, TailPolicy::Trim);
        let _ = t.advance(true, 0);
        let _ = t.advance(false, 1); // short gap would still be bridged...
        assert_eq!(
            t.reset(),
            Some(RegionEvent::End {
                start_frame: 0,
                end_frame: 1,
            }),
            "...but a coverage hole closes the region immediately"
        );
        assert_eq!(t.reset(), None);
    }

    #[test]
    fn tracker_min_off_zero_or_one_disables_bridging() {
        for min_off in [0, 1] {
            let mut t = RegionTracker::new(min_off, 0, TailPolicy::Trim);
            let _ = t.advance(true, 0);
            assert_eq!(
                t.advance(false, 1),
                Some(RegionEvent::End {
                    start_frame: 0,
                    end_frame: 1,
                }),
                "min_off {min_off} must close on the first inactive frame"
            );
        }
    }

    #[test]
    fn tracker_keeps_applies_min_on() {
        let t = RegionTracker::new(0, 3, TailPolicy::Trim);
        assert!(t.keeps(0, 3));
        assert!(!t.keeps(0, 2));
        assert_eq!(t.min_on(), 3);
        // min_on 0/1 keeps every region (regions are at least one frame).
        let t = RegionTracker::new(0, 1, TailPolicy::Trim);
        assert!(t.keeps(4, 5));
    }
}
