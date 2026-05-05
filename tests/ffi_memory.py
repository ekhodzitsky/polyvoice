#!/usr/bin/env python3
"""FFI memory safety tests for polyvoice.

Requires the shared library to be built with --features ffi:
    cargo build --features ffi

Usage:
    python tests/ffi_memory.py
"""

import ctypes
import os
import platform
import sys
import tempfile


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


def test_basic_lifecycle():
    """Create, run, free — basic happy path."""
    lib = ctypes.CDLL(find_library())

    lib.polyvoice_diarizer_new.argtypes = [ctypes.c_float, ctypes.c_int]
    lib.polyvoice_diarizer_new.restype = ctypes.c_void_p

    lib.polyvoice_diarizer_run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.polyvoice_diarizer_run.restype = ctypes.c_void_p

    lib.polyvoice_result_free.argtypes = [ctypes.c_void_p]
    lib.polyvoice_result_free.restype = None

    lib.polyvoice_diarizer_free.argtypes = [ctypes.c_void_p]
    lib.polyvoice_diarizer_free.restype = None

    lib.polyvoice_version.restype = ctypes.c_char_p

    version = lib.polyvoice_version()
    assert version is not None
    print(f"Library version: {version.decode()}")

    diarizer = lib.polyvoice_diarizer_new(0.5, 64)
    assert diarizer is not None, "diarizer_new returned NULL"

    # 2 seconds of silence at 16 kHz.
    samples = (ctypes.c_float * 32000)(*([0.0] * 32000))
    result = lib.polyvoice_diarizer_run(diarizer, samples, len(samples))

    # Free result first, then diarizer.
    if result:
        lib.polyvoice_result_free(result)
    lib.polyvoice_diarizer_free(diarizer)
    print("test_basic_lifecycle: PASSED")


def test_null_handling():
    """NULL pointers should be handled gracefully."""
    lib = ctypes.CDLL(find_library())

    lib.polyvoice_diarizer_run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.polyvoice_diarizer_run.restype = ctypes.c_void_p

    lib.polyvoice_diarizer_free.argtypes = [ctypes.c_void_p]
    lib.polyvoice_diarizer_free.restype = None

    lib.polyvoice_result_free.argtypes = [ctypes.c_void_p]
    lib.polyvoice_result_free.restype = None

    # run with NULL diarizer -> NULL
    samples = (ctypes.c_float * 100)(*([0.0] * 100))
    result = lib.polyvoice_diarizer_run(None, samples, 100)
    assert result is None, "run with NULL diarizer should return NULL"

    # free NULL -> no crash
    lib.polyvoice_diarizer_free(None)
    lib.polyvoice_result_free(None)
    print("test_null_handling: PASSED")


def test_large_audio():
    """Stress test with 10 minutes of audio."""
    lib = ctypes.CDLL(find_library())

    lib.polyvoice_diarizer_new.argtypes = [ctypes.c_float, ctypes.c_int]
    lib.polyvoice_diarizer_new.restype = ctypes.c_void_p

    lib.polyvoice_diarizer_run.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.polyvoice_diarizer_run.restype = ctypes.c_void_p

    lib.polyvoice_result_free.argtypes = [ctypes.c_void_p]
    lib.polyvoice_result_free.restype = None

    lib.polyvoice_diarizer_free.argtypes = [ctypes.c_void_p]
    lib.polyvoice_diarizer_free.restype = None

    diarizer = lib.polyvoice_diarizer_new(0.5, 64)
    assert diarizer is not None

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
        result = lib.polyvoice_diarizer_run(diarizer, arr, num_samples)
        if result:
            lib.polyvoice_result_free(result)
        lib.polyvoice_diarizer_free(diarizer)
        print("test_large_audio: PASSED")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_basic_lifecycle()
    test_null_handling()
    test_large_audio()
    print("\nAll FFI memory safety tests passed.")
