"""Certify the Stehfest inversion against transforms with known closed forms."""

from __future__ import annotations

import numpy as np
import pytest

from deep_pta.engine.stehfest import stehfest_inverse, stehfest_weights

T = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])


def test_weights_sum_is_zero() -> None:
    # The Stehfest weights sum to (close to) zero for any even N.
    for n in (8, 10, 12, 14):
        assert abs(sum(stehfest_weights(n))) < 1e-6


def test_weights_reject_odd_n() -> None:
    with pytest.raises(ValueError):
        stehfest_weights(11)


def test_inverse_of_one_over_u_is_constant() -> None:
    # L^{-1}{1/u} = 1
    got = stehfest_inverse(lambda u: 1.0 / u, T)
    np.testing.assert_allclose(got, np.ones_like(T), rtol=1e-6)


def test_inverse_of_one_over_u_squared_is_t() -> None:
    # L^{-1}{1/u^2} = t
    got = stehfest_inverse(lambda u: 1.0 / u**2, T)
    np.testing.assert_allclose(got, T, rtol=1e-5)


def test_inverse_of_one_over_u_plus_a_is_exponential() -> None:
    # L^{-1}{1/(u+a)} = exp(-a t)
    a = 0.7
    got = stehfest_inverse(lambda u: 1.0 / (u + a), T)
    # Stehfest loses relative accuracy on the exponential tail (value ~1e-3 at t=10),
    # but the absolute error stays tiny; assert with an absolute floor.
    np.testing.assert_allclose(got, np.exp(-a * T), rtol=2e-3, atol=5e-4)


def test_inverse_of_one_over_sqrt_u() -> None:
    # L^{-1}{1/sqrt(u)} = 1/sqrt(pi t)
    got = stehfest_inverse(lambda u: 1.0 / np.sqrt(u), T)
    np.testing.assert_allclose(got, 1.0 / np.sqrt(np.pi * T), rtol=1e-4)
