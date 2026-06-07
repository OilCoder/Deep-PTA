"""Parameter sampling, factored labels, and the domain-knowledge validity filter.

Each sample draws a reservoir class (4) and a boundary class (4), log-uniform shape
parameters for those classes, and a dimensionless time window. The **validity filter**
encodes interpreter knowledge: a boundary is labelable only if its signature develops
*after* wellbore storage ends and *within* the observed window; otherwise it is
indistinguishable from infinite-acting and is **relabelled as infinite** (its parameter
deactivated). Inactive parameters are masked out of the regression loss downstream.

Distribution note (changed 2026-06-07): the time window ``t_max`` is now conditioned on
the sampled parameters — always past storage (``~50·C_D``), and for boundary cases
observable ~82% of draws (``SHORT_WINDOW_PROB`` undeveloped). This is a domain-aware
curriculum, not a leak: train/val/test share the same conditioning, and the disjoint
``C_D`` band split (see ``generator.split_of``) still measures generalization on unseen
storage. It replaces the earlier unconditional ``t_max ~ loguniform[1e3,1e8]`` which made
~70% of finite boundaries fall outside the window and collapse to "infinite".

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

# Probability that a finite-boundary draw gets a window too short for the
# signature to develop, yielding a genuine (physically correct) "infinite" label.
SHORT_WINDOW_PROB = 0.18

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

    # Wellbore storage masks the early response until ~t_D ≈ 50·C_D (end of storage).
    # The observed window must extend past that for any regime to be discriminable.
    storage_end = 50.0 * c_d
    t_min = _loguniform(rng, 1e-2, 1.0)

    # Boundary parameter, drawn before the time window so the window can be
    # conditioned on whether the boundary signature is observable (a domain-aware
    # curriculum, not a leak — train/val/test share the same conditioning).
    t_boundary: float | None = None
    bnd_param: float = 0.0
    bnd_log_key: str = ""
    bnd_raw_key: str = ""
    if boundary_class in (BND_SEALING_FAULT, BND_CONSTANT_PRESSURE):
        bnd_param = _loguniform(rng, 300.0, 5000.0)  # L_D
        t_boundary = bnd_param**2  # signature appears around t_D ~ L_D^2
        bnd_log_key, bnd_raw_key = "log_LD", "L_D"
    elif boundary_class == BND_CLOSED:
        bnd_param = _loguniform(rng, 200.0, 3000.0)  # r_eD
        t_boundary = 0.5 * bnd_param**2  # pseudo-steady state at t_D ~ r_eD^2 / 2
        bnd_log_key, bnd_raw_key = "log_reD", "r_eD"

    # Condition t_max so the response is meaningful: always past storage, and for
    # boundary cases observable ~82% of the time, deliberately undeveloped ~18%
    # (the undeveloped ones become genuine "infinite" via the validity filter).
    if t_boundary is None:
        t_max = _loguniform(rng, max(1e3, 20.0 * storage_end), 1e8)
    elif rng.random() < SHORT_WINDOW_PROB:
        t_max = _loguniform(rng, 20.0 * storage_end, max(40.0 * storage_end, 0.5 * t_boundary))
    else:
        lo = max(2.0 * t_boundary, 20.0 * storage_end)
        t_max = _loguniform(rng, lo, 50.0 * lo)
    t_d = np.logspace(np.log10(t_min), np.log10(t_max), n_time)

    # Validity filter: a boundary is labelable only if its signature develops
    # after storage ends and within the observed window; otherwise it is
    # indistinguishable from infinite-acting and is relabelled accordingly.
    if t_boundary is not None:
        if storage_end < t_boundary < t_max:
            raw[bnd_raw_key] = bnd_param
            targets[_IDX[bnd_log_key]], mask[_IDX[bnd_log_key]] = np.log10(bnd_param), True
        else:
            boundary_class = BND_INFINITE

    return CurveParams(
        reservoir_class=reservoir_class,
        boundary_class=boundary_class,
        targets=targets,
        mask=mask,
        raw=raw,
        t_d=t_d,
    )
