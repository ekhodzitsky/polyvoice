#!/usr/bin/env python3
"""FFI memory safety tests for the polyvoice C ABI v3.

Requires the shared library built with --features ffi and a cached Balanced
ONNX model bundle (polyvoice_pipeline_create loads it eagerly):
    cargo build --features ffi
    cargo run --features cli --bin polyvoice -- download-models --profile balanced

Set POLYVOICE_MODELS_DIR to point at a non-default model cache directory.

Usage:
    python scripts/ffi_memory.py
"""

import ctypes
import os
import platform
import tempfile

PROFILE_BALANCED = 1
MODELS_DIR = os.environ.get("POLYVOICE_MODELS_DIR")


def find_library() -> str:
    """Locate the polyvoice shared library relative to the project root."""
    system = platform.system()
    if system == "Darwin":
        name = "libpolyvoice.dylib"
    elif system == "Windows":
        name = "polyvoice.dll"
    else:
        name = "libpolyvoice.so"

    # Try target/debug first, then target/release.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for profile in ("debug", "release"):
        path = os.path.join(root, "target", profile, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {name}. Build with: cargo build --features ffi")


def declare_signatures(lib) -> None:
    """Declare argtypes/restypes for the ABI v3 entry points used below."""
    lib.polyvoice_pipeline_create.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.polyvoice_pipeline_create.restype = ctypes.c_int

    lib.polyvoice_pipeline_run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.polyvoice_pipeline_run.restype = ctypes.c_int

    lib.polyvoice_pipeline_destroy.argtypes = [ctypes.c_void_p]
    lib.polyvoice_pipeline_destroy.restype = None

    lib.polyvoice_free_string.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.polyvoice_free_string.restype = None


def create_pipeline(lib) -> ctypes.c_void_p:
    """Create a Balanced pipeline, honoring POLYVOICE_MODELS_DIR."""
    cache_dir = MODELS_DIR.encode() if MODELS_DIR else None
    handle = ctypes.c_void_p()
    rc = lib.polyvoice_pipeline_create(PROFILE_BALANCED, cache_dir, ctypes.byref(handle))
    assert rc == 0, f"pipeline_create returned status {rc} (cached Balanced ONNX bundle required)"
    assert handle, "pipeline_create returned OK but a NULL handle"
    return handle


def test_basic_lifecycle():
    """Create, run, free — basic happy path."""
    lib = ctypes.CDLL(find_library())
    declare_signatures(lib)

    handle = create_pipeline(lib)

    # 2 seconds of silence at 16 kHz.
    samples = (ctypes.c_float * 32000)(*([0.0] * 32000))
    out = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    rc = lib.polyvoice_pipeline_run(
        handle, samples, len(samples), 16000, ctypes.byref(out), ctypes.byref(out_len)
    )
    assert rc == 0, f"pipeline_run returned status {rc}"
    assert out, "pipeline_run returned OK but a NULL result string"

    # Free the result string first, then the pipeline.
    lib.polyvoice_free_string(out, out_len)
    lib.polyvoice_pipeline_destroy(handle)
    print("test_basic_lifecycle: PASSED")


def test_null_handling():
    """NULL pointers should be rejected gracefully, not crash."""
    lib = ctypes.CDLL(find_library())
    declare_signatures(lib)

    samples = (ctypes.c_float * 100)(*([0.0] * 100))
    out = ctypes.c_void_p()
    out_len = ctypes.c_size_t()

    # run with NULL pipeline -> non-zero status, out stays NULL
    rc = lib.polyvoice_pipeline_run(
        None, samples, len(samples), 16000, ctypes.byref(out), ctypes.byref(out_len)
    )
    assert rc != 0, "run with NULL pipeline should return a non-zero status"
    assert not out, "run with NULL pipeline must not produce a result string"

    # destroy/free NULL -> no crash
    lib.polyvoice_pipeline_destroy(None)
    lib.polyvoice_free_string(None, 0)
    print("test_null_handling: PASSED")


def test_large_audio():
    """Stress test with 10 minutes of audio."""
    lib = ctypes.CDLL(find_library())
    declare_signatures(lib)

    handle = create_pipeline(lib)

    # 10 minutes @ 16 kHz = 9,600,000 samples.
    # Write to a temporary binary file to avoid keeping it all in Python RAM.
    num_samples = 16000 * 600
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"\x00" * (num_samples * 4))
        tmp_path = f.name

    try:
        # mmap the file as float array.
        arr = (ctypes.c_float * num_samples).from_buffer_copy(
            open(tmp_path, "rb").read()
        )
        out = ctypes.c_void_p()
        out_len = ctypes.c_size_t()
        rc = lib.polyvoice_pipeline_run(
            handle, arr, num_samples, 16000, ctypes.byref(out), ctypes.byref(out_len)
        )
        assert rc == 0, f"pipeline_run returned status {rc}"
        if out:
            lib.polyvoice_free_string(out, out_len)
        lib.polyvoice_pipeline_destroy(handle)
        print("test_large_audio: PASSED")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_basic_lifecycle()
    test_null_handling()
    test_large_audio()
    print("\nAll FFI memory safety tests passed.")
