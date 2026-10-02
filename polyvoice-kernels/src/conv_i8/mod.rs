//! Implicit INT8 GEMM convolution — no full im2col.
//!
//! Matches ONNX QDQ: `y[oc] = act_scale * w_scale[oc] * Σ (w_i8 - w_zp)(x_i8 - x_zp) + bias`.
//! Activations are quantized from the incoming f32 map; 3×3 / 1×1 patches are
//! packed a few output pixels at a time so the working set stays in L1.
//! On Apple Silicon the inner product is `sdot` (not the unstable `vdotq_s32`).

include!("dispatch.rs");
include!("quant.rs");
include!("s1.rs");
include!("s2.rs");
include!("pointwise.rs");
include!("gemm.rs");
