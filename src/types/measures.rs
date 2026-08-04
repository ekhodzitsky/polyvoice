//! Validated scalars and time ranges.
use serde::{Deserialize, Serialize};

/// A validated sample rate (8000–192000 Hz).
///
/// Invariant: 8000 <= inner <= 192000.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SampleRate(u32);

impl SampleRate {
    /// { true }
    /// `pub fn new(rate: u32) -> Option<Self>`
    /// { ret.is_some() == (8000..=192000).contains(&rate) }
    /// Create a validated sample rate.
    ///
    /// Returns `None` if the rate is outside the supported range (8000–192000 Hz).
    ///
    /// ```rust
    /// use polyvoice::SampleRate;
    /// let sr = SampleRate::new(16000).expect("valid rate");
    /// assert_eq!(sr.get(), 16000);
    /// assert!(SampleRate::new(7000).is_none());
    /// ```
    pub fn new(rate: u32) -> Option<Self> {
        (8000..=192000).contains(&rate).then_some(Self(rate))
    }

    /// { true }
    /// pub fn get(&self) -> u32
    /// { ret == self.0 && 8000 <= ret && ret <= 192000 }
    /// Return the raw sample rate value in Hz.
    ///
    /// ```rust
    /// use polyvoice::SampleRate;
    /// let sr = SampleRate::new(44100).unwrap();
    /// assert_eq!(sr.get(), 44100);
    /// ```
    pub fn get(&self) -> u32 {
        self.0
    }
}

impl Default for SampleRate {
    fn default() -> Self {
        Self(16000)
    }
}

/// A validated confidence score in [0.0, 1.0].
///
/// Invariant: 0.0 <= inner <= 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Confidence(f32);

impl Confidence {
    /// { true }
    /// `pub fn new(v: f32) -> Option<Self>`
    /// { ret.is_some() == (0.0..=1.0).contains(&v) }
    /// Create a validated confidence score.
    ///
    /// Returns `None` if `v` is outside `[0.0, 1.0]`.
    ///
    /// ```rust
    /// use polyvoice::Confidence;
    /// assert!(Confidence::new(0.75).is_some());
    /// assert!(Confidence::new(1.5).is_none());
    /// ```
    pub fn new(v: f32) -> Option<Self> {
        (0.0..=1.0).contains(&v).then_some(Self(v))
    }

    /// { true }
    /// pub fn get(&self) -> f32
    /// { ret == self.0 && 0.0 <= ret && ret <= 1.0 }
    /// Return the raw confidence value.
    ///
    /// ```rust
    /// use polyvoice::Confidence;
    /// let c = Confidence::new(0.9).unwrap();
    /// assert_eq!(c.get(), 0.9);
    /// ```
    pub fn get(&self) -> f32 {
        self.0
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self(1.0)
    }
}

/// A time interval in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
}

impl TimeRange {
    /// { true }
    /// pub fn duration(&self) -> f64
    /// { ret >= 0.0 }
    /// Return the duration of this time range in seconds.
    ///
    /// ```rust
    /// use polyvoice::TimeRange;
    /// let tr = TimeRange { start: 1.0, end: 3.5 };
    /// assert_eq!(tr.duration(), 2.5);
    /// ```
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    /// Midpoint `(start + end) / 2` in seconds.
    #[inline]
    pub fn midpoint(&self) -> f64 {
        (self.start + self.end) / 2.0
    }

    /// Inclusive coverage: `start <= t && end >= t`.
    ///
    /// Used by midpoint word→speaker labeling; on overlapping turns the caller
    /// decides which covering turn wins (typically first in slice order).
    #[inline]
    pub fn contains_instant(&self, t: f64) -> bool {
        self.start <= t && self.end >= t
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rate_bounds_are_inclusive() {
        assert_eq!(SampleRate::new(8000).unwrap().get(), 8000);
        assert_eq!(SampleRate::new(192000).unwrap().get(), 192000);
        assert!(SampleRate::new(7999).is_none());
        assert!(SampleRate::new(192001).is_none());
        assert!(SampleRate::new(0).is_none());
        assert_eq!(SampleRate::default().get(), 16000);
    }

    #[test]
    fn confidence_bounds_are_inclusive() {
        assert_eq!(Confidence::new(0.0).unwrap().get(), 0.0);
        assert_eq!(Confidence::new(1.0).unwrap().get(), 1.0);
        assert!(Confidence::new(-0.01).is_none());
        assert!(Confidence::new(1.01).is_none());
        assert!(Confidence::new(f32::NAN).is_none());
        assert_eq!(Confidence::default().get(), 1.0);
    }

    #[test]
    fn time_range_geometry() {
        let tr = TimeRange {
            start: 1.0,
            end: 3.5,
        };
        assert_eq!(tr.duration(), 2.5);
        assert_eq!(tr.midpoint(), 2.25);
        // Inverted ranges clamp the duration to zero.
        let inv = TimeRange {
            start: 3.0,
            end: 1.0,
        };
        assert_eq!(inv.duration(), 0.0);
        // Bounds are inclusive.
        assert!(tr.contains_instant(1.0));
        assert!(tr.contains_instant(3.5));
        assert!(tr.contains_instant(2.0));
        assert!(!tr.contains_instant(0.999));
        assert!(!tr.contains_instant(3.501));
    }

    #[test]
    fn measures_serde_roundtrip() {
        let tr = TimeRange {
            start: 0.5,
            end: 1.5,
        };
        let json = serde_json::to_string(&tr).unwrap();
        assert_eq!(serde_json::from_str::<TimeRange>(&json).unwrap(), tr);

        let sr = SampleRate::new(44100).unwrap();
        let json = serde_json::to_string(&sr).unwrap();
        assert_eq!(serde_json::from_str::<SampleRate>(&json).unwrap(), sr);

        let c = Confidence::new(0.25).unwrap();
        let json = serde_json::to_string(&c).unwrap();
        assert_eq!(serde_json::from_str::<Confidence>(&json).unwrap(), c);
    }
}
