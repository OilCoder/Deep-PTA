"""Wellbore storage + skin wrapper and the radial sandface solution in Laplace space.

The inner boundary condition (wellbore storage ``C_D`` and skin ``S``) is applied to
*any* bare sandface reservoir solution :math:`\\bar p_{rD}(u)` through the general
relation derived from [Agarwal1970]_ / [vanEverdingenHurst1949]_ and consolidated by
[MavorCincoLey1979]_:

.. math::

    \\bar p_{wD}(u) =
    \\frac{u\\,\\bar p_{rD}(u) + S}{u\\,\\bigl[\\,1 + C_D u\\,(u\\,\\bar p_{rD}(u) + S)\\bigr]}

For the homogeneous radial case the bare sandface solution is
:math:`\\bar p_{rD}(u) = K_0(x) / (u\\,x\\,K_1(x))` with :math:`x = \\sqrt{u\\,f(u)}`,
where ``f(u)`` is the reservoir coupling function (1 for homogeneous).

References
----------
.. [Agarwal1970] Agarwal, R.G., Al-Hussainy, R. & Ramey, H.J. (1970). An
   Investigation of Wellbore Storage and Skin Effect in Unsteady Liquid Flow.
   SPEJ 10(3), 279-290.
.. [vanEverdingenHurst1949] van Everdingen, A.F. & Hurst, W. (1949). The
   Application of the Laplace Transformation to Flow Problems in Reservoirs.
   Trans. AIME 186, 305-324.
.. [MavorCincoLey1979] Mavor, M.J. & Cinco-Ley, H. (1979). Transient Pressure
   Behavior of Naturally Fractured Reservoirs. SPE 7977.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import k0e, k1e

#: A reservoir coupling function ``f(u)``: maps Laplace variable(s) to ``f(u)``.
CouplingFn: TypeAlias = Callable[[NDArray[np.float64]], NDArray[np.float64]]

#: A bare sandface Laplace solution ``p_rD(u)`` (no storage, no skin).
SandfaceFn: TypeAlias = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def radial_sandface(f_of_u: CouplingFn) -> SandfaceFn:
    """Build the bare radial sandface solution for a reservoir coupling ``f(u)``.

    Parameters
    ----------
    f_of_u : callable
        Reservoir coupling function ``f(u)`` (1 for homogeneous; Warren-Root for
        double porosity).

    Returns
    -------
    callable
        ``p_rD(u)`` with no wellbore storage or skin.
    """

    def p_rd(u: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.sqrt(u * f_of_u(u))
        # Exponentially scaled Bessel K cancel in the ratio (numerically safe for large x).
        return np.asarray(k0e(x) / (u * x * k1e(x)), dtype=np.float64)

    return p_rd


def apply_storage_skin(
    p_rd: SandfaceFn,
    c_d: float,
    s: float,
) -> SandfaceFn:
    """Wrap a bare sandface solution with wellbore storage ``C_D`` and skin ``S``.

    Parameters
    ----------
    p_rd : callable
        Bare sandface Laplace solution ``p_rD(u)`` (no storage, no skin).
    c_d : float
        Dimensionless wellbore storage coefficient.
    s : float
        Skin factor.

    Returns
    -------
    callable
        Wellbore Laplace solution ``p_wD(u)`` including storage and skin.
    """

    def p_wd(u: NDArray[np.float64]) -> NDArray[np.float64]:
        inner = u * p_rd(u) + s
        return inner / (u * (1.0 + c_d * u * inner))

    return p_wd
