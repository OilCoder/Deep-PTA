"""Smoke tests for the inference path (structure, not accuracy) and the app builder."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from deep_pta.app.inference import (  # noqa: E402
    decode_params,
    diagnose,
    reconstruct_derivative,
)
from deep_pta.engine import boundaries as bnd  # noqa: E402
from deep_pta.engine.solution import evaluate, make_homogeneous  # noqa: E402
from deep_pta.models.resnet1d import ResNet1D  # noqa: E402


def _synthetic_series() -> tuple[np.ndarray, np.ndarray]:
    t_d = np.logspace(-1, 6, 240)
    p_wd, _ = evaluate(make_homogeneous(), bnd.infinite(), (100.0, 2.0), t_d)
    return t_d, p_wd


def test_diagnose_output_structure() -> None:
    model = ResNet1D(base_channels=16, n_blocks=4)
    t, dp = _synthetic_series()
    result = diagnose(model, t, dp)
    assert 0 <= int(result["reservoir_class"]) < 4
    assert 0 <= int(result["boundary_class"]) < 4
    assert 0.0 <= float(result["confidence"]) <= 1.0
    assert isinstance(result["params"], dict)
    assert "C_D" in result["params"] and "S" in result["params"]


def test_diagnose_with_physics_channels() -> None:
    """The v3 input path: 5-channel representation feeds a 5-channel model."""
    from deep_pta.models.tcn import TCN1D

    model = TCN1D(in_channels=5, width=16, n_levels=4)
    t, dp = _synthetic_series()
    result = diagnose(model, t, dp, extra_channels=("sep", "slope"))
    x = result["x"]
    assert isinstance(x, np.ndarray) and x.shape == (5, 256)
    assert 0 <= int(result["reservoir_class"]) < 4


def test_reconstruct_derivative_is_finite() -> None:
    params = decode_params(np.array([2.0, 3.0, 0, 0, 0, 0, 0]), 0, 0)
    recon = reconstruct_derivative(0, 0, params)
    assert recon.shape == (256,)
    assert np.isfinite(recon).all()


def test_build_demo_importable() -> None:
    pytest.importorskip("gradio")
    from deep_pta.app.app import build_demo

    demo = build_demo()
    assert demo is not None
