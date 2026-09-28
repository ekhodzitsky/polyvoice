//! Hand-written inference kernels for polyvoice.
//!
//! This crate is **not** an ONNX runtime. It implements two shipping graphs
//! from their initializers only:
//! - WeSpeaker ResNet34 (`resnet34_int8.onnx`, QDQ weights)
//! - Pyannote powerset-3.0 (`powerset_int8.onnx`) — SincNet + 4× biLSTM

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(not(test), deny(clippy::panic))]

#[cfg(apple_accelerate)]
mod accelerate;
#[cfg(apple_accelerate)]
mod bnns;
#[cfg(apple_accelerate)]
mod bnns_graph;
mod conv;
mod conv_i8;
mod error;
mod gemm;
mod intra;
#[cfg(linux_cblas)]
mod linux_cblas;
mod lstm;
mod onnx_init;
mod powerset;
mod qlinear;
mod resnet34;
mod rten_matmul;
mod scratch;
mod seq1d;
mod tensor;

#[cfg(apple_accelerate)]
pub use bnns::prof as bnns_prof;
pub use conv_i8::{file_parallelism, set_file_parallelism, set_intra_threads};
pub use error::KernelError;
pub use gemm::gemm_bias_row;
pub use powerset::{N_CLASSES, Powerset};
pub use resnet34::{EMBED_DIM, N_MELS, ResNet34};
pub use scratch::reclaim as reclaim_scratch;

/// BNNS counters are zero when the experimental Apple Rust backend is selected.
#[cfg(all(target_vendor = "apple", not(apple_accelerate)))]
pub fn bnns_prof() -> (u64, u64, u64) {
    (0, 0, 0)
}
