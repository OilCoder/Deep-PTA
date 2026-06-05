"""Compose reservoir model, boundary, and wellbore into a time-domain response.

A :class:`ReservoirSpec` carries a bare sandface Laplace solution plus the reservoir
coupling ``f(u)`` (used by image-well boundaries). :func:`evaluate` wires together
model x boundary x wellbore (storage + skin), inverts to time with Stehfest, and
returns both the dimensionless wellbore pressure ``p_wD`` and its Bourdet derivative.

The Bourdet derivative is obtained analytically: since
:math:`\\mathcal{L}\\{dp_D/dt_D\\} = u\\,\\bar p_{wD}(u)`, we have
:math:`p_D' = t_D\\,\\frac{dp_D}{dt_D} = t_D\\,\\mathcal{L}^{-1}\\{u\\,\\bar p_{wD}\\}`,
avoiding any numerical differentiation of the clean curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from deep_pta.engine.boundaries import Boundary
from deep_pta.engine.fractures import (
    finite_conductivity_fracture,
    infinite_conductivity_fracture,
)
from deep_pta.engine.laplace_base import (
    CouplingFn,
    SandfaceFn,
    apply_storage_skin,
    radial_sandface,
)
from deep_pta.engine.reservoir_models import homogeneous, warren_root
from deep_pta.engine.stehfest import stehfest_inverse


@dataclass(frozen=True)
class ReservoirSpec:
    """A reservoir model: its bare sandface solution and coupling function.

    Attributes
    ----------
    sandface : callable
        Bare sandface Laplace solution ``p_rD(u)`` (no storage, no skin).
    coupling : callable
        Reservoir coupling ``f(u)`` used by image-well boundaries (``homogeneous``
        when the boundary is infinite or not radial-coupled).
    """

    sandface: SandfaceFn
    coupling: CouplingFn


def make_homogeneous() -> ReservoirSpec:
    """Homogeneous infinite-acting radial reservoir."""
    f = homogeneous()
    return ReservoirSpec(sandface=radial_sandface(f), coupling=f)


def make_double_porosity(omega: float, lam: float) -> ReservoirSpec:
    """Warren-Root double-porosity reservoir.

    Parameters
    ----------
    omega : float
        Storativity ratio in ``(0, 1)``.
    lam : float
        Interporosity flow coefficient (> 0).
    """
    f = warren_root(omega, lam)
    return ReservoirSpec(sandface=radial_sandface(f), coupling=f)


def make_infinite_conductivity_fracture() -> ReservoirSpec:
    """Infinite-conductivity vertical fracture (linear flow, 1/2 slope)."""
    return ReservoirSpec(
        sandface=infinite_conductivity_fracture(),
        coupling=homogeneous(),
    )


def make_finite_conductivity_fracture(f_cd: float) -> ReservoirSpec:
    """Finite-conductivity vertical fracture (bilinear flow, 1/4 slope).

    Parameters
    ----------
    f_cd : float
        Dimensionless fracture conductivity (> 0).
    """
    return ReservoirSpec(
        sandface=finite_conductivity_fracture(f_cd),
        coupling=homogeneous(),
    )


def evaluate(
    reservoir: ReservoirSpec,
    boundary: Boundary,
    well: tuple[float, float],
    t_d: NDArray[np.float64],
    n: int = 12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the wellbore pressure and Bourdet derivative in dimensionless time.

    Parameters
    ----------
    reservoir : ReservoirSpec
        The reservoir model.
    boundary : Boundary
        Outer-boundary transform (``infinite``, ``sealing_fault``, ...).
    well : tuple of float
        ``(C_D, S)`` — dimensionless wellbore storage and skin.
    t_d : numpy.ndarray
        Strictly positive dimensionless times.
    n : int, optional
        Number of Stehfest terms (even), by default 12.

    Returns
    -------
    p_wd : numpy.ndarray
        Dimensionless wellbore pressure ``p_wD(t_D)``.
    dp_wd : numpy.ndarray
        Bourdet derivative ``p_wD' = t_D dp_wD/dt_D``.
    """
    c_d, s = well
    p_rd = boundary(reservoir.sandface, reservoir.coupling)
    p_wd_lap = apply_storage_skin(p_rd, c_d, s)

    t_d = np.asarray(t_d, dtype=np.float64)
    p_wd = stehfest_inverse(p_wd_lap, t_d, n)
    dp_wd = t_d * stehfest_inverse(lambda u: u * p_wd_lap(u), t_d, n)
    return p_wd, dp_wd
