//! Stub — implemented in Task 4.

/// Stub — implemented in Task 4.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowersetClass {
    Silence,
    Speaker(u8),
    Pair(u8, u8),
}

/// Stub — implemented in Task 4.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameLabel {
    pub class: PowersetClass,
    pub max_softmax: f32,
}

/// Stub — implemented in Task 4.
#[allow(dead_code)]
pub struct PowersetDecoder;
