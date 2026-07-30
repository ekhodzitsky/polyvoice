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
