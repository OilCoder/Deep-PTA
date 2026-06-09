"""Analytic checks for the physics-informed extra channels (sep, slope).

On a pure fracture-linear-flow curve ``dp = A * sqrt(t)`` the Bourdet derivative is
``t d(dp)/dt = dp / 2``, so the separation channel must read exactly ``log10(2)`` and
the local log-log slope of the derivative must read exactly ``0.5`` — closed-form
ground truths independent of the engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_pta.data.representation import (
    _SEP_MEAN,
    _SEP_STD,
    N_GRID,
    build_representation,
)


def _linear_flow_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.logspace(0, 6, 400)
    dp = 3.0 * np.sqrt(t)
    deriv = dp / 2.0
    return t, dp, t, deriv


def test_legacy_shape_unchanged() -> None:
    """Default call keeps the (3, 256) contract."""
    t, dp, t_der, deriv = _linear_flow_curve()
    x = build_representation(t, dp, t_der, deriv)
    assert x.shape == (3, N_GRID)
    assert x.dtype == np.float32


def test_extra_channels_shape_and_order() -> None:
    """sep+slope append after the 3 base channels in request order."""
    t, dp, t_der, deriv = _linear_flow_curve()
    x = build_representation(t, dp, t_der, deriv, extra_channels=("sep", "slope"))
    assert x.shape == (5, N_GRID)
    # ✅ base channels identical to the legacy call
    base = build_representation(t, dp, t_der, deriv)
    np.testing.assert_allclose(x[:3], base, rtol=0, atol=0)


def test_sep_reads_log10_2_on_linear_flow() -> None:
    """De-normalized separation equals log10(2) during fracture linear flow."""
    t, dp, t_der, deriv = _linear_flow_curve()
    x = build_representation(t, dp, t_der, deriv, extra_channels=("sep",))
    sep_raw = x[3] * _SEP_STD + _SEP_MEAN
    np.testing.assert_allclose(sep_raw, np.log10(2.0), atol=1e-3)


def test_slope_reads_half_on_linear_flow() -> None:
    """Local log-log slope of a t^0.5 derivative is 0.5 across the grid."""
    t, dp, t_der, deriv = _linear_flow_curve()
    x = build_representation(t, dp, t_der, deriv, extra_channels=("slope",))
    np.testing.assert_allclose(x[3], 0.5, atol=1e-3)


def test_slope_reads_unit_on_storage() -> None:
    """Unit-slope storage line (dp' proportional to t) reads slope 1."""
    t = np.logspace(0, 4, 300)
    dp = 2.0 * t
    deriv = 2.0 * t  # on the unit-slope line dp = dp'
    x = build_representation(t, dp, t, deriv, extra_channels=("sep", "slope"))
    sep_raw = x[3] * _SEP_STD + _SEP_MEAN
    np.testing.assert_allclose(sep_raw, 0.0, atol=1e-3)  # dp == dp' -> sep 0
    np.testing.assert_allclose(x[4], 1.0, atol=1e-3)


def test_unknown_channel_raises() -> None:
    """An unknown extra-channel name fails loudly."""
    t, dp, t_der, deriv = _linear_flow_curve()
    with pytest.raises(ValueError, match="unknown extra channel"):
        build_representation(t, dp, t_der, deriv, extra_channels=("bogus",))
