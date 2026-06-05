"""Numerical inversion of Laplace transforms via the Stehfest algorithm.

The Stehfest algorithm [Stehfest1970]_ approximates the time-domain function
``f(t)`` from its Laplace transform ``F(u)`` as

.. math::

    f(t) \\approx \\frac{\\ln 2}{t} \\sum_{i=1}^{N} V_i \\, F\\!\\left(\\frac{i \\ln 2}{t}\\right)

with ``N`` even (8-12 typical; 12 by default). The weights ``V_i`` depend only on
``N`` and are cached.

References
----------
.. [Stehfest1970] Stehfest, H. (1970). Algorithm 368: Numerical Inversion of
   Laplace Transforms. Comm. ACM 13(1), 47-49.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from math import factorial, log

import numpy as np
from numpy.typing import NDArray

LN2 = log(2.0)


@lru_cache(maxsize=8)
def stehfest_weights(n: int) -> tuple[float, ...]:
    """Compute the Stehfest weights ``V_i`` for ``i = 1 .. n``.

    Parameters
    ----------
    n : int
        Number of terms; must be even and positive.

    Returns
    -------
    tuple of float
        The ``n`` weights ``(V_1, ..., V_n)``.

    Raises
    ------
    ValueError
        If ``n`` is not a positive even integer.
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError(f"Stehfest N must be a positive even integer, got {n}")

    half = n // 2
    weights = []
    for i in range(1, n + 1):
        acc = 0.0
        k_min = (i + 1) // 2
        k_max = min(i, half)
        for k in range(k_min, k_max + 1):
            num = (k**half) * factorial(2 * k)
            den = (
                factorial(half - k)
                * factorial(k)
                * factorial(k - 1)
                * factorial(i - k)
                * factorial(2 * k - i)
            )
            acc += num / den
        weights.append(((-1) ** (i + half)) * acc)
    return tuple(weights)


def stehfest_inverse(
    f_laplace: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    t: NDArray[np.float64],
    n: int = 12,
) -> NDArray[np.float64]:
    """Invert a Laplace-domain function to the time domain.

    Parameters
    ----------
    f_laplace : callable
        Laplace transform ``F(u)``; receives a 1-D array of positive real Laplace
        variables and returns an array of the same shape.
    t : numpy.ndarray
        Strictly positive times at which to evaluate ``f(t)``.
    n : int, optional
        Number of Stehfest terms (even), by default 12.

    Returns
    -------
    numpy.ndarray
        ``f(t)`` evaluated at each ``t``.
    """
    t = np.asarray(t, dtype=np.float64)
    if np.any(t <= 0.0):
        raise ValueError("Stehfest inversion requires strictly positive t")

    weights = stehfest_weights(n)
    acc = np.zeros_like(t)
    for i, v_i in enumerate(weights, start=1):
        u = i * LN2 / t
        acc += v_i * f_laplace(u)
    return (LN2 / t) * acc
