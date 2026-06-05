"""Vertical-fracture sandface solutions in Laplace space.

Two fractured-well models, each returning a bare sandface solution
:math:`\\bar p_{rD}(u)` in the fracture time frame (``t_Dxf``):

* **Infinite-conductivity** [Gringarten1974]_ — modelled by the uniform-flux line
  source along the fracture half-length. Early-time **linear flow** gives a 1/2 slope
  on the Bourdet derivative.
* **Finite-conductivity** [CincoLey1978]_ — early-time **bilinear flow** gives a 1/4
  slope. Implemented as the uniform-flux solution plus a bilinear early-time term
  scaled by the dimensionless fracture conductivity ``F_CD``. This reproduces the
  diagnostic 1/4 slope; it is a slope-faithful approximation of the full Cinco-Ley
  solution (see ``documentation/plan-implementacion.md``).

References
----------
.. [Gringarten1974] Gringarten, A.C., Ramey, H.J. & Raghavan, R. (1974).
   Unsteady-State Pressure Distributions Created by a Well With a Single
   Infinite-Conductivity Vertical Fracture. SPEJ 14(4), 347-360.
.. [CincoLey1978] Cinco-Ley, H., Samaniego, F. & Dominguez, N. (1978). Transient
   Pressure Behavior for a Well With a Finite-Conductivity Vertical Fracture.
   SPEJ 18(4), 253-264.
"""

from __future__ import annotations

from math import gamma

import numpy as np
from numpy.typing import NDArray
from scipy.special import k0

from deep_pta.engine.laplace_base import SandfaceFn

# Fixed Gauss-Legendre nodes/weights on [0, 1] for the uniform-flux line integral.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(24)
_GL_X = 0.5 * (_GL_NODES + 1.0)  # map [-1, 1] -> [0, 1]
_GL_W = 0.5 * _GL_WEIGHTS

# L{t^{1/4}} = Gamma(5/4) * u^{-5/4}; constant for the bilinear term.
_BILINEAR_EXP = -1.25
_GAMMA_54 = gamma(1.25)


def infinite_conductivity_fracture() -> SandfaceFn:
    """Uniform-flux infinite-conductivity vertical fracture (linear flow, 1/2 slope).

    The sandface pressure is the line-source integral over the fracture half-length,
    observed at the fracture centre:

    .. math::

        \\bar p_{rD}(u) = \\frac{1}{u} \\int_0^1 K_0\\!\\left(\\sqrt{u}\\, x\\right) dx

    Returns
    -------
    callable
        Bare sandface solution ``p_rD(u)`` in the fracture time frame.
    """

    def p_rd(u: NDArray[np.float64]) -> NDArray[np.float64]:
        sqrt_u = np.sqrt(u)
        # Integrate K0(sqrt(u) * x) over x in [0, 1] with fixed Gauss-Legendre nodes.
        # Shape: (n_u, n_nodes) via broadcasting.
        arg = sqrt_u[:, None] * _GL_X[None, :]
        integral = np.sum(_GL_W[None, :] * k0(arg), axis=1)
        return np.asarray(integral / u, dtype=np.float64)

    return p_rd


def finite_conductivity_fracture(f_cd: float) -> SandfaceFn:
    """Finite-conductivity vertical fracture (bilinear flow, 1/4 slope).

    Uniform-flux solution plus a bilinear early-time term that dominates at large
    ``u`` (early time) and scales as ``1/sqrt(F_CD)`` per Cinco-Ley.

    Parameters
    ----------
    f_cd : float
        Dimensionless fracture conductivity ``F_CD`` (> 0). Lower values deepen and
        lengthen the bilinear regime.

    Returns
    -------
    callable
        Bare sandface solution ``p_rD(u)``.

    Raises
    ------
    ValueError
        If ``f_cd`` is not positive.
    """
    if f_cd <= 0.0:
        raise ValueError(f"F_CD must be positive, got {f_cd}")

    base = infinite_conductivity_fracture()
    # Bilinear amplitude: pi / (2 sqrt(F_CD)) folded with the L{t^{1/4}} constant.
    amp = (np.pi / (2.0 * np.sqrt(f_cd))) * _GAMMA_54

    def p_rd(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(base(u) + amp * u**_BILINEAR_EXP, dtype=np.float64)

    return p_rd
