//! Whole-ResNet BNNSGraph (compiled Core ML mlmodelc) on Apple.
//!
//! One compiled graph with dynamic T replaces 36 per-layer BNNS creates.
//! Missing artifact → caller keeps the layer path.

#![cfg(target_vendor = "apple")]

use crate::error::KernelError;
use crate::tensor::Tensor;
use std::ffi::CString;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::OnceLock;

unsafe extern "C" {
    fn pv_bnns_graph_compile(path: *const i8) -> *mut c_void;
    fn pv_bnns_graph_free(graph: *mut c_void);
    fn pv_bnns_graph_context(graph: *mut c_void) -> *mut c_void;
    fn pv_bnns_graph_context_free(ctx: *mut c_void);
    fn pv_bnns_graph_exec(
        ctx: *mut c_void,
        t: i32,
        input: *const f32,
        output: *mut f32,
        out_w: *mut i32,
    ) -> i32;
}

struct Graph {
    raw: NonNull<c_void>,
}

// SAFETY: the compiled graph is immutable after compile.
unsafe impl Send for Graph {}
unsafe impl Sync for Graph {}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { pv_bnns_graph_free(self.raw.as_ptr()) }
    }
}

struct Ctx {
    raw: NonNull<c_void>,
}

// SAFETY: execute is serialized behind a process-wide mutex.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

impl Drop for Ctx {
    fn drop(&mut self) {
        unsafe { pv_bnns_graph_context_free(self.raw.as_ptr()) }
    }
}

fn enabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    !*OFF.get_or_init(|| std::env::var_os("POLYVOICE_NO_BNNS_GRAPH").is_some())
}

fn is_mlmodelc(p: &Path) -> bool {
    p.join("model.mil").is_file() || p.join("coremldata.bin").is_file()
}

/// Resolve a compiled `mlmodelc` next to the ONNX, or from the env override.
pub fn resolve_path(onnx: &Path) -> Option<PathBuf> {
    if !enabled() {
        return None;
    }
    if let Some(p) = std::env::var_os("POLYVOICE_RESNET_MLMODELC") {
        let p = PathBuf::from(p);
        if is_mlmodelc(&p) {
            return Some(p);
        }
        let nested = p.join("resnet34_bnns.mlmodelc");
        if is_mlmodelc(&nested) {
            return Some(nested);
        }
    }
    let sib = onnx.with_file_name("resnet34_bnns.mlmodelc");
    if is_mlmodelc(&sib) {
        return Some(sib);
    }
    let nested = sib.join("resnet34_bnns.mlmodelc");
    is_mlmodelc(&nested).then_some(nested)
}

fn compile(path: &Path) -> Option<Graph> {
    let c = CString::new(path.to_string_lossy().as_bytes()).ok()?;
    let ptr = unsafe { pv_bnns_graph_compile(c.as_ptr()) };
    NonNull::new(ptr).map(|raw| Graph { raw })
}

static GRAPH: OnceLock<Option<Graph>> = OnceLock::new();

fn graph(path: &Path) -> Option<&'static Graph> {
    GRAPH.get_or_init(|| compile(path)).as_ref()
}

/// Compile the graph if a `mlmodelc` is present. True on success.
pub fn warmup(onnx: &Path) -> bool {
    resolve_path(onnx).and_then(|p| graph(&p)).is_some()
}

fn shared_ctx(g: &Graph) -> Option<std::sync::MutexGuard<'static, Option<Ctx>>> {
    static CTX: OnceLock<std::sync::Mutex<Option<Ctx>>> = OnceLock::new();
    let lock = CTX.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = lock.lock().ok()?;
    if guard.is_none() {
        let ptr = unsafe { pv_bnns_graph_context(g.raw.as_ptr()) };
        *guard = NonNull::new(ptr).map(|raw| Ctx { raw });
    }
    Some(guard)
}

/// Run the compiled conv stack. `x` is NCHW `[1,1,80,T]`.
pub fn try_forward(onnx: &Path, x: &Tensor) -> Result<Option<Tensor>, KernelError> {
    if x.n != 1 || x.c != 1 || x.h != 80 || x.w < 8 {
        return Ok(None);
    }
    let Some(path) = resolve_path(onnx) else {
        return Ok(None);
    };
    let Some(g) = graph(&path) else {
        return Ok(None);
    };
    let Some(guard) = shared_ctx(g) else {
        return Ok(None);
    };
    let Some(ctx) = guard.as_ref() else {
        return Ok(None);
    };
    let t = i32::try_from(x.w).map_err(|_| KernelError::Model {
        detail: "resnet T does not fit i32".into(),
    })?;
    let mut ow_guess = x.w;
    for _ in 0..3 {
        ow_guess = (ow_guess + 2 - 3) / 2 + 1;
    }
    let mut y = Tensor::uninit(1, 256, 10, ow_guess.max(1));
    let mut ow: i32 = 0;
    let rc = unsafe {
        pv_bnns_graph_exec(
            ctx.raw.as_ptr(),
            t,
            x.data.as_ptr(),
            y.data.as_mut_ptr(),
            &mut ow,
        )
    };
    if rc != 0 {
        return Ok(None);
    }
    let ow = usize::try_from(ow).unwrap_or(ow_guess).max(1);
    if ow != y.w {
        y.w = ow;
        let len = 256usize.saturating_mul(10).saturating_mul(ow);
        if y.data.len() > len {
            y.data.truncate(len);
        }
    }
    Ok(Some(y))
}
