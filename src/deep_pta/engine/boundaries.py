"""Outer-boundary effects applied to a radial sandface solution.

Boundaries are modelled by superposition of image wells [Horne1995]_ on top of the
infinite-acting sandface solution, except the closed boundary which uses the exact
bounded-reservoir line-source solution [vanEverdingenHurst1949]_:

* **Sealing fault** — image producer at ``2 L_D`` adds a ``+K_0`` term; the radial
  derivative plateau **doubles**.
* **Constant pressure** — image injector at ``2 L_D`` subtracts a ``K_0`` term; the
  derivative **falls** toward zero.
* **Closed (no-flow circular)** — exact bounded solution; late-time pseudo-steady
  flow gives a **unit slope** on the derivative.

A boundary is a callable ``(p_rD, f_of_u) -> p_rD'`` that transforms a bare sandface
solution into a boundary-aware one (the closed boundary rebuilds the solution and
ignores the incoming ``p_rD``).

References
----------
.. [Horne1995] Horne, R.N. (1995). Modern Well Test Analysis: A Computer-Aided
   Approach (2nd ed.). Petroway.
.. [vanEverdingenHurst1949] van Everdingen, A.F. & Hurst, W. (1949). The
   Application of the Laplace Transformation to Flow Problems in Reservoirs.
   Trans. AIME 186, 305-324.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import ive, k0, kve

from deep_pta.engine.laplace_base import CouplingFn, SandfaceFn

#: A boundary transforms a bare sandface solution given the reservoir coupling.
Boundary: TypeAlias = Callable[[SandfaceFn, CouplingFn], SandfaceFn]


def infinite() -> Boundary:
    """Infinite-acting (no outer boundary): the sandface solution is unchanged.

    Returns
    -------
    callable
        Identity boundary transform.
    """

    def apply(p_rd: SandfaceFn, f_of_u: CouplingFn) -> SandfaceFn:
        return p_rd

    return apply


def _image_boundary(l_d: float, sign: float) -> Boundary:
    """Image-well boundary: add ``sign * K_0(2 L_D x) / u`` to the sandface solution."""
    if l_d <= 0.0:
        raise ValueError(f"L_D must be positive, got {l_d}")

    def apply(p_rd: SandfaceFn, f_of_u: CouplingFn) -> SandfaceFn:
        def modified(u: NDArray[np.float64]) -> NDArray[np.float64]:
            x = np.sqrt(u * f_of_u(u))
            return np.asarray(p_rd(u) + sign * k0(2.0 * l_d * x) / u, dtype=np.float64)

        return modified

    return apply


def sealing_fault(l_d: float) -> Boundary:
    """Single sealing fault at dimensionless distance ``L_D`` (derivative doubles).

    Parameters
    ----------
    l_d : float
        Dimensionless distance to the fault (> 0).

    Returns
    -------
    callable
        Boundary transform adding the image-producer term.
    """
    return _image_boundary(l_d, +1.0)


def constant_pressure(l_d: float) -> Boundary:
    """Constant-pressure linear boundary at ``L_D`` (derivative falls).

    Parameters
    ----------
    l_d : float
        Dimensionless distance to the boundary (> 0).

    Returns
    -------
    callable
        Boundary transform subtracting the image-injector term.
    """
    return _image_boundary(l_d, -1.0)


def closed(r_ed: float) -> Boundary:
    """Closed circular no-flow outer boundary at ``r_eD`` (unit-slope late time).

    Uses the exact bounded-reservoir line-source solution (sandface at ``r_D = 1``),
    evaluated with exponentially scaled Bessel functions for numerical stability.

    Parameters
    ----------
    r_ed : float
        Dimensionless external radius (> 1).

    Returns
    -------
    callable
        Boundary transform rebuilding the bounded sandface solution.

    Raises
    ------
    ValueError
        If ``r_ed`` is not greater than 1.
    """
    if r_ed <= 1.0:
        raise ValueError(f"r_eD must be greater than 1, got {r_ed}")

    def apply(p_rd: SandfaceFn, f_of_u: CouplingFn) -> SandfaceFn:
        def bounded(u: NDArray[np.float64]) -> NDArray[np.float64]:
            x = np.sqrt(u * f_of_u(u))
            a = r_ed * x  # outer-boundary argument
            b = x  # sandface argument (r_D = 1)
            e = np.exp(2.0 * (b - a))  # in (0, 1] since a >= b
            ive0b, ive1a, ive1b = ive(0, b), ive(1, a), ive(1, b)
            kve0b, kve1a, kve1b = kve(0, b), kve(1, a), kve(1, b)
            num = ive1a * kve0b + kve1a * ive0b * e
            # The u factor matches the constant-rate sandface convention
            # p_rD = K0/(u x K1); without it the bounded solution is u-times too large.
            den = u * b * (ive1a * kve1b - kve1a * ive1b * e)
            return np.asarray(num / den, dtype=np.float64)

        return bounded

    return apply
