"""Tests for the Bourdet derivative and Agarwal equivalent time."""

from __future__ import annotations

import numpy as np

from deep_pta.data.bourdet import agarwal_equivalent_time, bourdet_derivative


def test_agarwal_equivalent_time() -> None:
    dt = np.array([1.0, 10.0, 100.0])
    t_p = 50.0
    expected = dt * t_p / (t_p + dt)
    np.testing.assert_allclose(agarwal_equivalent_time(dt, t_p), expected)


def test_derivative_of_log_is_constant() -> None:
    # dp = a*ln(t) + b  ->  t d(dp)/dt = a  (radial-flow plateau).
    t = np.logspace(-1, 4, 300)
    a = 2.0
    dp = a * np.log(t) + 1.0
    _, deriv = bourdet_derivative(t, dp, l_window=0.2)
    np.testing.assert_allclose(deriv, a, rtol=1e-6)


def test_derivative_of_power_law() -> None:
    # dp = t^m  ->  t d(dp)/dt = m * t^m.
    t = np.logspace(-1, 3, 400)
    m = 0.5
    dp = t**m
    t_mid, deriv = bourdet_derivative(t, dp, l_window=0.1)
    np.testing.assert_allclose(deriv, m * t_mid**m, rtol=2e-2)
