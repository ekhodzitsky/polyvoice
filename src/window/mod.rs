//! Sliding-window utilities for batch and streaming pipelines.
//!
//! [`WindowIter`] iterates over `(start, end)` sample ranges.
//! [`WindowBuffer`] buffers incoming samples and yields full windows.

/// Invalid window geometry rejected by [`WindowIter::try_new`] and
/// [`WindowBuffer::try_new`].
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowError {
    /// Window size must be positive.
    #[error("win must be > 0")]
    ZeroWindow,
    /// Hop size must be positive.
    #[error("hop must be > 0")]
    ZeroHop,
    /// Hop must not exceed the window size.
    #[error("hop ({hop}) must be <= win ({win})")]
    HopExceedsWindow {
        /// Requested hop size in samples.
        hop: usize,
        /// Requested window size in samples.
        win: usize,
    },
}

/// Iterator over fixed-size (or partial trailing) windows.
///
/// Yields `(start_sample, end_sample)` pairs.  `end_sample` is exclusive.
#[derive(Debug, Clone)]
pub struct WindowIter {
    start: usize,
    total: usize,
    win: usize,
    hop: usize,
    include_partial: bool,
}

impl WindowIter {
    /// { win > 0 && hop > 0 }
    /// `pub fn new(total: usize, win: usize, hop: usize) -> Self`
    /// { true }
    /// Create a new iterator.
    ///
    /// * `total` — total number of samples in the audio region.
    /// * `win`   — window size in samples.
    /// * `hop`   — hop size in samples.
    ///
    /// # Panics
    ///
    /// Panics if `win == 0`, `hop == 0`, or `hop > win`.
    /// Use [`try_new`](Self::try_new) for a fallible alternative.
    #[allow(clippy::panic)] // Documented convenience over `try_new`.
    pub fn new(total: usize, win: usize, hop: usize) -> Self {
        match Self::try_new(total, win, hop) {
            Ok(iter) => iter,
            Err(e) => panic!("WindowIter::new: {e}"),
        }
    }

    /// { true }
    /// `pub fn try_new(total: usize, win: usize, hop: usize) -> Result<Self, WindowError>`
    /// { ret.is_ok() == (win > 0 && hop > 0 && hop <= win) }
    /// Fallible constructor: validates the window geometry and returns
    /// [`WindowError`] instead of panicking.
    ///
    /// * `total` — total number of samples in the audio region.
    /// * `win`   — window size in samples (must be > 0).
    /// * `hop`   — hop size in samples (must be > 0 and <= `win`).
    pub fn try_new(total: usize, win: usize, hop: usize) -> Result<Self, WindowError> {
        if win == 0 {
            return Err(WindowError::ZeroWindow);
        }
        if hop == 0 {
            return Err(WindowError::ZeroHop);
        }
        if hop > win {
            return Err(WindowError::HopExceedsWindow { hop, win });
        }
        Ok(Self {
            start: 0,
            total,
            win,
            hop,
            include_partial: false,
        })
    }

    /// { true }
    /// `pub fn include_partial(mut self) -> Self`
    /// { ret.include_partial }
    /// Include a final partial window if the region does not divide evenly.
    pub fn include_partial(mut self) -> Self {
        self.include_partial = true;
        self
    }
}

impl Iterator for WindowIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.total {
            return None;
        }
        let end = if self.include_partial {
            (self.start + self.win).min(self.total)
        } else if self.start + self.win > self.total {
            return None;
        } else {
            self.start + self.win
        };
        let item = (self.start, end);
        self.start += self.hop;
        Some(item)
    }
}

/// Streaming buffer that accumulates samples and yields full windows.
///
/// Maintains an internal ring-like buffer.  Call [`extend`](Self::extend)
/// with incoming chunks, then repeatedly call [`try_pop`](Self::try_pop)
/// to consume every window that is ready.
#[derive(Debug, Clone)]
pub struct WindowBuffer {
    buf: Vec<f32>,
    win: usize,
    hop: usize,
    next_start: usize,
}

impl WindowBuffer {
    /// { win > 0 && hop > 0 }
    /// `pub fn new(win: usize, hop: usize) -> Self`
    /// { true }
    /// Create a new buffer.
    ///
    /// * `win` — window size in samples.
    /// * `hop` — hop size in samples.
    ///
    /// # Panics
    ///
    /// Panics if `win == 0`, `hop == 0`, or `hop > win`.
    /// Use [`try_new`](Self::try_new) for a fallible alternative.
    #[allow(clippy::panic)] // Documented convenience over `try_new`.
    pub fn new(win: usize, hop: usize) -> Self {
        match Self::try_new(win, hop) {
            Ok(buf) => buf,
            Err(e) => panic!("WindowBuffer::new: {e}"),
        }
    }

    /// { true }
    /// `pub fn try_new(win: usize, hop: usize) -> Result<Self, WindowError>`
    /// { ret.is_ok() == (win > 0 && hop > 0 && hop <= win) }
    /// Fallible constructor: validates the window geometry and returns
    /// [`WindowError`] instead of panicking.
    ///
    /// * `win` — window size in samples (must be > 0).
    /// * `hop` — hop size in samples (must be > 0 and <= `win`).
    pub fn try_new(win: usize, hop: usize) -> Result<Self, WindowError> {
        if win == 0 {
            return Err(WindowError::ZeroWindow);
        }
        if hop == 0 {
            return Err(WindowError::ZeroHop);
        }
        if hop > win {
            return Err(WindowError::HopExceedsWindow { hop, win });
        }
        Ok(Self {
            buf: Vec::new(),
            win,
            hop,
            next_start: 0,
        })
    }

    /// { true }
    /// `pub fn extend(&mut self, samples: &[f32])`
    /// { true }
    /// Append samples to the buffer.
    pub fn extend(&mut self, samples: &[f32]) {
        self.buf.extend_from_slice(samples);
    }

    /// { true }
    /// `pub fn try_pop(&mut self) -> Option<(usize, Vec<f32>)>`
    /// { ret.as_ref().map(|(_, w)| w.len() == self.win) }
    /// Return the next full window if one is available.
    ///
    /// Returns `Some((global_start, buf[..win].to_vec()))` where `global_start` is the
    /// sample offset of this window relative to the start of the stream.
    /// The window is cloned so that the internal buffer can advance immediately.
    pub fn try_pop(&mut self) -> Option<(usize, Vec<f32>)> {
        if self.buf.len() < self.win {
            return None;
        }
        let start = self.next_start;
        let window = self.buf[..self.win].to_vec();
        self.next_start += self.hop;
        self.buf.drain(..self.hop);
        Some((start, window))
    }

    /// { true }
    /// `pub fn flush(&mut self) -> Option<(usize, Vec<f32>)>`
    /// { true }
    /// Zero-pad the remaining buffer to `win` and return the final window.
    ///
    /// Returns `None` if the buffer is empty.
    pub fn flush(&mut self) -> Option<(usize, Vec<f32>)> {
        if self.buf.is_empty() {
            return None;
        }
        let start = self.next_start;
        let mut padded = self.buf.clone();
        if padded.len() < self.win {
            padded.resize(self.win, 0.0f32);
        }
        self.buf.clear();
        Some((start, padded))
    }

    /// { true }
    /// `pub fn is_empty(&self) -> bool`
    /// { ret == (self.buf.len() == 0) }
    /// Whether the buffer is currently empty.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// { true }
    /// `pub fn len(&self) -> usize`
    /// { ret == self.buf.len() }
    /// Current length of the buffered samples.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// { true }
    /// `pub fn clear(&mut self)`
    /// { self.buf.is_empty() }
    /// Clear all buffered samples and reset the next-start offset.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// { true }
    /// `pub fn set_next_start(&mut self, start: usize)`
    /// { self.next_start == start }
    /// Set the next-start offset to a specific value.
    pub fn set_next_start(&mut self, start: usize) {
        self.next_start = start;
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_iter_complete_only() {
        let ranges: Vec<_> = WindowIter::new(10, 3, 2).collect();
        assert_eq!(ranges, vec![(0, 3), (2, 5), (4, 7), (6, 9)]);
    }

    #[test]
    fn window_iter_include_partial() {
        let ranges: Vec<_> = WindowIter::new(10, 3, 2).include_partial().collect();
        assert_eq!(ranges, vec![(0, 3), (2, 5), (4, 7), (6, 9), (8, 10)]);
    }

    #[test]
    fn window_iter_empty() {
        let ranges: Vec<_> = WindowIter::new(0, 3, 2).collect();
        assert!(ranges.is_empty());
    }

    #[test]
    fn window_iter_shorter_than_win() {
        let ranges: Vec<_> = WindowIter::new(2, 3, 2).collect();
        assert!(ranges.is_empty());
        let ranges: Vec<_> = WindowIter::new(2, 3, 2).include_partial().collect();
        assert_eq!(ranges, vec![(0, 2)]);
    }

    #[test]
    fn window_buffer_pop_and_flush() {
        let mut buf = WindowBuffer::new(4, 2);
        buf.extend(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (s1, w1) = buf.try_pop().unwrap();
        assert_eq!(s1, 0);
        assert_eq!(w1, vec![1.0, 2.0, 3.0, 4.0]);
        let (s2, w2) = buf.try_pop().unwrap();
        assert_eq!(s2, 2);
        assert_eq!(w2, vec![3.0, 4.0, 5.0, 6.0]);
        assert!(buf.try_pop().is_none());

        let (s3, w3) = buf.flush().unwrap();
        assert_eq!(s3, 4);
        assert_eq!(w3, vec![5.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn window_buffer_flush_empty() {
        let mut buf = WindowBuffer::new(4, 2);
        assert!(buf.flush().is_none());
    }

    #[test]
    #[should_panic(expected = "WindowIter::new: win must be > 0")]
    fn window_iter_rejects_zero_win() {
        let _ = WindowIter::new(10, 0, 1);
    }

    #[test]
    #[should_panic(expected = "WindowIter::new: hop must be > 0")]
    fn window_iter_rejects_zero_hop() {
        let _ = WindowIter::new(10, 2, 0);
    }

    #[test]
    #[should_panic(expected = "WindowIter::new: hop (3) must be <= win (2)")]
    fn window_iter_rejects_hop_greater_than_win() {
        let _ = WindowIter::new(10, 2, 3);
    }

    #[test]
    #[should_panic(expected = "WindowBuffer::new: win must be > 0")]
    fn window_buffer_rejects_zero_win() {
        let _ = WindowBuffer::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "WindowBuffer::new: hop must be > 0")]
    fn window_buffer_rejects_zero_hop() {
        let _ = WindowBuffer::new(4, 0);
    }

    #[test]
    #[should_panic(expected = "WindowBuffer::new: hop (5) must be <= win (4)")]
    fn window_buffer_rejects_hop_greater_than_win() {
        let _ = WindowBuffer::new(4, 5);
    }

    #[test]
    fn window_iter_try_new_reports_geometry_errors() {
        assert_eq!(
            WindowIter::try_new(10, 0, 1).unwrap_err(),
            WindowError::ZeroWindow
        );
        assert_eq!(
            WindowIter::try_new(10, 2, 0).unwrap_err(),
            WindowError::ZeroHop
        );
        assert_eq!(
            WindowIter::try_new(10, 2, 3).unwrap_err(),
            WindowError::HopExceedsWindow { hop: 3, win: 2 }
        );
        assert!(WindowIter::try_new(10, 2, 1).is_ok());
    }

    #[test]
    fn window_buffer_try_new_reports_geometry_errors() {
        assert_eq!(
            WindowBuffer::try_new(0, 1).unwrap_err(),
            WindowError::ZeroWindow
        );
        assert_eq!(
            WindowBuffer::try_new(4, 0).unwrap_err(),
            WindowError::ZeroHop
        );
        assert_eq!(
            WindowBuffer::try_new(4, 5).unwrap_err(),
            WindowError::HopExceedsWindow { hop: 5, win: 4 }
        );
        assert!(WindowBuffer::try_new(4, 2).is_ok());
    }

    #[test]
    fn window_buffer_len_empty_clear_and_set_next_start() {
        let mut buf = WindowBuffer::new(4, 2);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        buf.extend(&[1.0, 2.0]);
        assert!(!buf.is_empty());
        assert_eq!(buf.len(), 2);
        buf.set_next_start(100);
        buf.extend(&[3.0, 4.0]);
        let (start, window) = buf.try_pop().unwrap();
        assert_eq!(start, 100);
        assert_eq!(window, vec![1.0, 2.0, 3.0, 4.0]);
        buf.clear();
        assert!(buf.is_empty());
        assert!(buf.try_pop().is_none());
    }

    #[test]
    fn window_buffer_flush_without_padding_when_buf_exceeds_win() {
        let mut buf = WindowBuffer::new(4, 4);
        buf.extend(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (start, window) = buf.flush().unwrap();
        assert_eq!(start, 0);
        assert_eq!(window, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(buf.is_empty());
    }
}

#[allow(clippy::unwrap_used)]
#[cfg(test)]
mod prop_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 1000,
            ..ProptestConfig::default()
        })]

        /// All yielded (start, end) pairs satisfy start < end.
        #[test]
        fn window_iter_start_less_than_end(
            total in 0usize..10000,
            (win, hop) in (1usize..5000)
                .prop_flat_map(|w| (Just(w), 1usize..=w)),
            include_partial in any::<bool>(),
        ) {
            let iter = if include_partial {
                WindowIter::new(total, win, hop).include_partial()
            } else {
                WindowIter::new(total, win, hop)
            };
            for (start, end) in iter {
                prop_assert!(start < end, "start {} must be < end {}", start, end);
            }
        }

        /// All yielded end values are <= total.
        #[test]
        fn window_iter_end_le_total(
            total in 0usize..10000,
            (win, hop) in (1usize..5000)
                .prop_flat_map(|w| (Just(w), 1usize..=w)),
            include_partial in any::<bool>(),
        ) {
            let iter = if include_partial {
                WindowIter::new(total, win, hop).include_partial()
            } else {
                WindowIter::new(total, win, hop)
            };
            for (_, end) in iter {
                prop_assert!(end <= total, "end {} must be <= total {}", end, total);
            }
        }

        /// When include_partial = false, every yielded window has end - start == win.
        #[test]
        fn window_iter_no_partial(
            total in 0usize..10000,
            (win, hop) in (1usize..5000)
                .prop_flat_map(|w| (Just(w), 1usize..=w)),
        ) {
            for (start, end) in WindowIter::new(total, win, hop) {
                prop_assert_eq!(
                    end - start,
                    win,
                    "window [{}..{}] must be full size {} when include_partial=false",
                    start, end, win
                );
            }
        }

        /// When include_partial = true, every yielded window satisfies
        /// end == (start + win).min(total). The final window reaches total
        /// when total > 0.
        #[test]
        fn window_iter_partial_last(
            total in 0usize..10000,
            (win, hop) in (1usize..5000)
                .prop_flat_map(|w| (Just(w), 1usize..=w)),
        ) {
            let ranges: Vec<_> = WindowIter::new(total, win, hop).include_partial().collect();
            for (start, end) in &ranges {
                prop_assert!(
                    *end == (*start + win).min(total),
                    "window [{}..{}] must equal ({} + {}).min({})",
                    start, end, start, win, total
                );
            }
        }
    }
}
