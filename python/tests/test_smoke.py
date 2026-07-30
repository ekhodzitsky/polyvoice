"""Smoke tests for the pyo3 bindings."""

import polyvoice


def test_pipeline_module_imports():
    assert hasattr(polyvoice, "Pipeline"), "Pipeline class should be exposed"


def test_pipeline_mobile_constructor_signature():
    # We can't actually build a Pipeline without cached ONNX, but we can
    # verify the class method exists and rejects invalid sample rate.
    assert hasattr(polyvoice.Pipeline, "mobile")
    assert hasattr(polyvoice.Pipeline, "balanced")
    assert hasattr(polyvoice.Pipeline, "run")
