"""Smoke test: validate_int8._render_report status flips on budget breach."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_int8", ROOT / "scripts" / "validate_int8.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_pass_status() -> None:
    mod = _load_module()
    results = {"der_delta": 0.3, "kl_max": 0.02, "fp32_der": 11.0, "int8_der": 11.3}
    budgets = mod.BUDGETS["powerset"]
    text = mod._render_report("powerset", results, budgets, ok=True)
    assert "PASS" in text


def test_report_fail_status() -> None:
    mod = _load_module()
    results = {"der_delta": 0.6, "kl_max": 0.02, "fp32_der": 11.0, "int8_der": 11.6}
    budgets = mod.BUDGETS["powerset"]
    text = mod._render_report("powerset", results, budgets, ok=False)
    assert "FAIL" in text


def test_budgets_contain_expected_keys() -> None:
    mod = _load_module()
    assert mod.BUDGETS["powerset"]["der_delta_max"] == 0.5
    assert mod.BUDGETS["powerset"]["kl_max"] == 0.05
    assert mod.BUDGETS["embedder"]["eer_delta_max"] == 0.30
    assert mod.BUDGETS["embedder"]["cosine_mean_min"] == 0.998
    assert mod.BUDGETS["embedder"]["cosine_p1_min"] == 0.991
