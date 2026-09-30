# polyvoice-kernels

Inference kernels for the specific WeSpeaker ResNet34 and Pyannote powerset
graphs used by polyvoice. This is not a general ONNX runtime. Model files are
supplied by the caller; the crate has no model downloader.

Default features are empty. Linux and Windows use Rust kernels. Apple targets
use Accelerate/BNNS and compile two bundled C wrappers with the platform SDK.
`experimental-darwin-rust` replaces that Apple path with Rust kernels; it is
experimental and is not the qualified product default. `system-openblas`
explicitly selects system LP64 OpenBLAS on Linux and requires its development
files and pkg-config; other targets retain their native implementation.

This crate is not dependency-free: it uses rten-gemm, rten-tensor, thiserror and
memmap2, with cc as a build dependency and optional pkg-config. Model licenses
are separate from this crate's MIT license; no model weights are included.

The 0.1.3 release preserves the 0.1.2 public Rust surface, adds file parallelism
and scratch reclamation controls, maps model weights, and introduces explicit
backend features. Linux no longer automatically selects installed OpenBLAS;
enable `system-openblas` to request that backend.

See the [project repository](https://github.com/ekhodzitsky/polyvoice) for the
diarization pipeline, supported platforms and release qualification policy.
