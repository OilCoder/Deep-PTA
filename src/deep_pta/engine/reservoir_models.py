"""Reservoir coupling functions ``f(u)`` for the radial sandface solution.

The coupling ``f(u)`` enters the radial solution through :math:`x = \\sqrt{u\\,f(u)}`.
Homogeneous reservoirs use ``f(u) = 1``; naturally fractured (double-porosity)
reservoirs use the Warren-Root pseudo-steady interporosity form [WarrenRoot1963]_.

References
----------
.. [WarrenRoot1963] Warren, J.E. & Root, P.J. (1963). The Behavior of Naturally
   Fractured Reservoirs. SPEJ 3(3), 245-255.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from deep_pta.engine.laplace_base import CouplingFn


def homogeneous() -> CouplingFn:
    """Homogeneous infinite-acting reservoir: ``f(u) = 1``.

    Returns
    -------
    callable
        Coupling function returning ones with the shape of its input.
    """

    def f_of_u(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones_like(u)

    return f_of_u


def warren_root(omega: float, lam: float) -> CouplingFn:
    """Double-porosity Warren-Root coupling (pseudo-steady interporosity flow).

    .. math::

        f(u) = \\frac{\\omega(1-\\omega)\\,u + \\lambda}{(1-\\omega)\\,u + \\lambda}

    The derivative shows two radial plateaus separated by a transition valley whose
    depth scales with ``omega`` and whose position scales with ``lambda``.

    Parameters
    ----------
    omega : float
        Storativity ratio ``omega`` in ``(0, 1)``.
    lam : float
        Interporosity flow coefficient ``lambda`` (dimensionless, > 0).

    Returns
    -------
    callable
        Coupling function ``f(u)``.

    Raises
    ------
    ValueError
        If ``omega`` is not in ``(0, 1)`` or ``lam`` is not positive.
    """
    if not 0.0 < omega < 1.0:
        raise ValueError(f"omega must be in (0, 1), got {omega}")
    if lam <= 0.0:
        raise ValueError(f"lambda must be positive, got {lam}")

    def f_of_u(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return (omega * (1.0 - omega) * u + lam) / ((1.0 - omega) * u + lam)

    return f_of_u
