//! Hand-written inference kernels for polyvoice.
//!
//! This crate is **not** an ONNX runtime. It implements two shipping graphs
//! from their initializers only:
//! - WeSpeaker ResNet34 (`resnet34_int8.onnx`, QDQ weights)
//! - Pyannote powerset-3.0 (`powerset_int8.onnx`) — SincNet + 4× biLSTM

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]

#[cfg(target_vendor = "apple")]
mod accelerate;
#[cfg(linux_cblas)]
mod linux_cblas;
#[cfg(target_vendor = "apple")]
mod bnns;
#[cfg(target_vendor = "apple")]
mod bnns_graph;
mod conv;
mod conv_i8;
mod intra;
mod rten_matmul;
mod error;
mod gemm;
mod lstm;
mod onnx_init;
mod powerset;
mod qlinear;
mod resnet34;
mod seq1d;
mod tensor;

#[cfg(target_vendor = "apple")]
pub use bnns::prof as bnns_prof;
pub use conv_i8::set_intra_threads;
pub use error::KernelError;
pub use gemm::gemm_bias_row;
pub use powerset::{MIN_SAMPLES as POWERSET_MIN_SAMPLES, N_CLASSES, Powerset};
pub use resnet34::{EMBED_DIM, N_MELS, ResNet34};
