"""Parameter sampling, factored labels, and the domain-knowledge validity filter.

Each sample draws a reservoir class (4) and a boundary class (4), log-uniform shape
parameters for those classes, and a dimensionless time window. The **validity filter**
encodes interpreter knowledge: if a boundary's signature cannot develop within the test
duration, the boundary is unlabelable and is **relabelled as infinite** (its parameter
deactivated). Inactive parameters are masked out of the regression loss downstream.

Regression target layout (length 7), all log10 except skin:

``[log C_D, S, log omega, log lambda, log L_D, log F_CD, log r_eD]``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Reservoir and boundary class indices.
RES_HOMOGENEOUS, RES_DOUBLE_POROSITY, RES_INF_FRACTURE, RES_FIN_FRACTURE = 0, 1, 2, 3
BND_INFINITE, BND_SEALING_FAULT, BND_CONSTANT_PRESSURE, BND_CLOSED = 0, 1, 2, 3
N_RESERVOIR = 4
N_BOUNDARY = 4

PARAM_NAMES = ("log_CD", "S", "log_omega", "log_lambda", "log_LD", "log_FCD", "log_reD")
N_PARAMS = len(PARAM_NAMES)
_IDX = {name: i for i, name in enumerate(PARAM_NAMES)}


@dataclass(frozen=True)
class CurveParams:
    """A sampled curve specification with factored labels and a target mask.

    Attributes
    ----------
    reservoir_class : int
        Reservoir class index (after the validity filter).
    boundary_class : int
        Boundary class index (after the validity filter).
    targets : numpy.ndarray
        Length-``N_PARAMS`` regression targets (log10 except skin); inactive entries
        are zero.
    mask : numpy.ndarray
        Boolean length-``N_PARAMS`` array marking the active (valid) targets.
    raw : dict
        Raw physical-ish parameters used to build the engine solution.
    t_d : numpy.ndarray
        Dimensionless time grid for engine evaluation.
    """

    reservoir_class: int
    boundary_class: int
    targets: NDArray[np.float64]
    mask: NDArray[np.bool_]
    raw: dict[str, float]
    t_d: NDArray[np.float64]


def active_param_mask(reservoir_class: int, boundary_class: int) -> NDArray[np.bool_]:
    """Return the active-parameter mask for a (reservoir, boundary) class pair.

    Storage and skin are always active; reservoir- and boundary-specific parameters are
    active only for their classes. Used at inference to decode the parameters a model
    is allowed to report for its predicted classes.

    Parameters
    ----------
    reservoir_class : int
        Reservoir class index.
    boundary_class : int
        Boundary class index.

    Returns
    -------
    numpy.ndarray
        Boolean length-``N_PARAMS`` mask.
    """
    mask = np.zeros(N_PARAMS, dtype=np.bool_)
    mask[_IDX["log_CD"]] = True
    mask[_IDX["S"]] = True
    if reservoir_class == RES_DOUBLE_POROSITY:
        mask[_IDX["log_omega"]] = True
        mask[_IDX["log_lambda"]] = True
    elif reservoir_class == RES_FIN_FRACTURE:
        mask[_IDX["log_FCD"]] = True
    if boundary_class in (BND_SEALING_FAULT, BND_CONSTANT_PRESSURE):
        mask[_IDX["log_LD"]] = True
    elif boundary_class == BND_CLOSED:
        mask[_IDX["log_reD"]] = True
    return mask


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(10.0 ** rng.uniform(np.log10(low), np.log10(high)))


def sample_curve(rng: np.random.Generator, n_time: int = 200) -> CurveParams:
    """Draw one curve specification (classes, parameters, time window, mask).

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded random generator for reproducibility.
    n_time : int, optional
        Number of dimensionless time points to evaluate, by default 200.

    Returns
    -------
    CurveParams
        The sampled specification after applying the validity filter.
    """
    reservoir_class = int(rng.integers(N_RESERVOIR))
    boundary_class = int(rng.integers(N_BOUNDARY))

    targets = np.zeros(N_PARAMS, dtype=np.float64)
    mask = np.zeros(N_PARAMS, dtype=np.bool_)
    raw: dict[str, float] = {}

    # Wellbore storage and skin are always active.
    c_d = _loguniform(rng, 1.0, 1e4)
    s = float(rng.uniform(-3.0, 10.0))
    raw["C_D"], raw["S"] = c_d, s
    targets[_IDX["log_CD"]], mask[_IDX["log_CD"]] = np.log10(c_d), True
    targets[_IDX["S"]], mask[_IDX["S"]] = s, True

    # Reservoir-specific parameters.
    if reservoir_class == RES_DOUBLE_POROSITY:
        omega = _loguniform(rng, 0.01, 0.5)
        lam = _loguniform(rng, 1e-9, 1e-4)
        raw["omega"], raw["lambda"] = omega, lam
        targets[_IDX["log_omega"]], mask[_IDX["log_omega"]] = np.log10(omega), True
        targets[_IDX["log_lambda"]], mask[_IDX["log_lambda"]] = np.log10(lam), True
    elif reservoir_class == RES_FIN_FRACTURE:
        f_cd = _loguniform(rng, 0.1, 500.0)
        raw["F_CD"] = f_cd
        targets[_IDX["log_FCD"]], mask[_IDX["log_FCD"]] = np.log10(f_cd), True

    # Time window (dimensionless): start in storage, end somewhere in/after radial.
    t_min = _loguniform(rng, 1e-2, 1.0)
    t_max = _loguniform(rng, 1e3, 1e8)
    t_d = np.logspace(np.log10(t_min), np.log10(t_max), n_time)

    # Boundary-specific parameters + validity filter.
    if boundary_class in (BND_SEALING_FAULT, BND_CONSTANT_PRESSURE):
        l_d = _loguniform(rng, 300.0, 5000.0)
        # Boundary signature appears around t_D ~ L_D^2; if beyond the window, unlabelable.
        if l_d**2 > t_max:
            boundary_class = BND_INFINITE
        else:
            raw["L_D"] = l_d
            targets[_IDX["log_LD"]], mask[_IDX["log_LD"]] = np.log10(l_d), True
    elif boundary_class == BND_CLOSED:
        r_ed = _loguniform(rng, 200.0, 3000.0)
        # Pseudo-steady state appears around t_D ~ r_eD^2 / 2.
        if 0.5 * r_ed**2 > t_max:
            boundary_class = BND_INFINITE
        else:
            raw["r_eD"] = r_ed
            targets[_IDX["log_reD"]], mask[_IDX["log_reD"]] = np.log10(r_ed), True

    return CurveParams(
        reservoir_class=reservoir_class,
        boundary_class=boundary_class,
        targets=targets,
        mask=mask,
        raw=raw,
        t_d=t_d,
    )
