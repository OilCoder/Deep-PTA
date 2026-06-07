"""On-the-fly synthetic curve generator: engine -> realism -> Bourdet -> representation.

Produces labelled 3-channel log-log samples with perfect labels by construction. The
generator is NumPy-only (no PyTorch) so it stays usable on the CPU build; a thin
``torch.utils.data.Dataset`` wrapper lives in :mod:`deep_pta.train`.

A **disjoint-range split** holds out a band of wellbore storage ``C_D`` for the test
set, so test performance measures genuine generalization (not just unseen noise).
"""

from __future__ import annotations

import h5py
import numpy as np
from numpy.typing import NDArray

from deep_pta.data.bourdet import bourdet_derivative
from deep_pta.data.realism import apply_realism
from deep_pta.data.representation import N_GRID, build_representation
from deep_pta.data.sampling import (
    BND_CLOSED,
    BND_CONSTANT_PRESSURE,
    BND_INFINITE,
    BND_SEALING_FAULT,
    N_BOUNDARY,
    N_RESERVOIR,
    RES_DOUBLE_POROSITY,
    RES_FIN_FRACTURE,
    RES_HOMOGENEOUS,
    RES_INF_FRACTURE,
    CurveParams,
    sample_curve,
)
from deep_pta.engine import boundaries as bnd
from deep_pta.engine.solution import (
    ReservoirSpec,
    evaluate,
    make_double_porosity,
    make_finite_conductivity_fracture,
    make_homogeneous,
    make_infinite_conductivity_fracture,
)

# Disjoint held-out C_D bands (log10): validation and test occupy distinct bands
# so HPO / early-stopping never observe the test storage regime. Train is the rest.
_VAL_CD_BAND = (2.5, 3.0)
_TEST_CD_BAND = (3.0, 3.5)


def _build_reservoir(cp: CurveParams) -> ReservoirSpec:
    if cp.reservoir_class == RES_HOMOGENEOUS:
        return make_homogeneous()
    if cp.reservoir_class == RES_DOUBLE_POROSITY:
        return make_double_porosity(cp.raw["omega"], cp.raw["lambda"])
    if cp.reservoir_class == RES_INF_FRACTURE:
        return make_infinite_conductivity_fracture()
    if cp.reservoir_class == RES_FIN_FRACTURE:
        return make_finite_conductivity_fracture(cp.raw["F_CD"])
    raise ValueError(f"unknown reservoir class {cp.reservoir_class}")


def _build_boundary(cp: CurveParams) -> bnd.Boundary:
    if cp.boundary_class == BND_INFINITE:
        return bnd.infinite()
    if cp.boundary_class == BND_SEALING_FAULT:
        return bnd.sealing_fault(cp.raw["L_D"])
    if cp.boundary_class == BND_CONSTANT_PRESSURE:
        return bnd.constant_pressure(cp.raw["L_D"])
    if cp.boundary_class == BND_CLOSED:
        return bnd.closed(cp.raw["r_eD"])
    raise ValueError(f"unknown boundary class {cp.boundary_class}")


def split_of(cp: CurveParams) -> str:
    """Assign a sample to ``"train"``, ``"val"``, or ``"test"`` by disjoint ``C_D`` bands."""
    log_cd = np.log10(cp.raw["C_D"])
    if _TEST_CD_BAND[0] <= log_cd < _TEST_CD_BAND[1]:
        return "test"
    if _VAL_CD_BAND[0] <= log_cd < _VAL_CD_BAND[1]:
        return "val"
    return "train"


def generate_sample(rng: np.random.Generator, max_retries: int = 20) -> dict[str, object]:
    """Generate one labelled sample, retrying on degenerate draws.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator.
    max_retries : int, optional
        Maximum resamples before giving up, by default 8.

    Returns
    -------
    dict
        Keys: ``x`` (3 x 256 float32), ``y_reservoir`` (int), ``y_boundary`` (int),
        ``targets`` (float64 [7]), ``mask`` (bool [7]), ``split`` (str).

    Raises
    ------
    RuntimeError
        If no valid sample is produced within ``max_retries``.
    """
    for _ in range(max_retries):
        cp = sample_curve(rng)
        try:
            reservoir = _build_reservoir(cp)
            boundary = _build_boundary(cp)
            p_wd, _ = evaluate(reservoir, boundary, (cp.raw["C_D"], cp.raw["S"]), cp.t_d)
            if not np.all(np.isfinite(p_wd)):
                continue
            # Stehfest inversion leaves tiny negatives on the near-zero early storage
            # region; clip that roundoff (drawdown pressure is non-negative).
            p_wd = np.clip(p_wd, 1e-8, None)
            t_obs, p_obs = apply_realism(rng, cp.t_d, p_wd)
            l_window = float(rng.uniform(0.1, 0.3))
            t_der, deriv = bourdet_derivative(t_obs, p_obs, l_window)
            if t_der.size < N_GRID // 8:
                continue
            x = build_representation(t_obs, p_obs, t_der, deriv)
        except (ValueError, KeyError, ZeroDivisionError):
            continue
        if not np.all(np.isfinite(x)):
            continue
        return {
            "x": x,
            "y_reservoir": cp.reservoir_class,
            "y_boundary": cp.boundary_class,
            "targets": cp.targets,
            "mask": cp.mask,
            "split": split_of(cp),
        }
    raise RuntimeError("failed to generate a valid sample within max_retries")


def export_frozen_test_set(path: str, n: int, seed: int = 0, split: str = "test") -> int:
    """Generate ``n`` samples from one split and write them to an HDF5 file.

    Parameters
    ----------
    path : str
        Output ``.h5`` path.
    n : int
        Number of samples to collect.
    seed : int, optional
        Base seed, by default 0.
    split : str, optional
        Which disjoint split to draw from, by default ``"test"``.

    Returns
    -------
    int
        Number of samples written (``n``).
    """
    rng = np.random.default_rng(seed)
    xs: list[NDArray[np.float32]] = []
    y_res, y_bnd, tgts, masks = [], [], [], []
    while len(xs) < n:
        sample = generate_sample(rng)
        if sample["split"] != split:
            continue
        xs.append(sample["x"])  # type: ignore[arg-type]
        y_res.append(sample["y_reservoir"])
        y_bnd.append(sample["y_boundary"])
        tgts.append(sample["targets"])
        masks.append(sample["mask"])

    _write_h5(path, xs, y_res, y_bnd, tgts, masks)
    return n


def export_stratified_set(
    path: str,
    n_per_cell: int,
    seed: int = 0,
    split: str = "test",
    max_draws: int = 2_000_000,
) -> int:
    """Export a class-balanced frozen set (equal samples per reservoir×boundary cell).

    Balancing both heads removes the imbalance that makes raw accuracy misleading,
    so balanced metrics computed on this set are trustworthy. Reuses the same
    generation path as :func:`export_frozen_test_set`.

    Parameters
    ----------
    path : str
        Output ``.h5`` path.
    n_per_cell : int
        Target samples for each of the ``N_RESERVOIR * N_BOUNDARY`` class cells.
    seed : int, optional
        Base seed, by default 0.
    split : str, optional
        Which disjoint split to draw from (``"train"``/``"val"``/``"test"``),
        by default ``"test"``.
    max_draws : int, optional
        Safety cap on total draws before giving up, by default 2_000_000.

    Returns
    -------
    int
        Number of samples written.

    Raises
    ------
    RuntimeError
        If some cells stay unfilled within ``max_draws`` (reports the shortfall).
    """
    rng = np.random.default_rng(seed)
    counts: dict[tuple[int, int], int] = {}
    xs: list[NDArray[np.float32]] = []
    y_res, y_bnd, tgts, masks = [], [], [], []
    target_cells = N_RESERVOIR * N_BOUNDARY

    draws = 0
    while len([c for c in counts.values() if c >= n_per_cell]) < target_cells:
        if draws >= max_draws:
            missing = {
                (r, b): n_per_cell - counts.get((r, b), 0)
                for r in range(N_RESERVOIR)
                for b in range(N_BOUNDARY)
                if counts.get((r, b), 0) < n_per_cell
            }
            raise RuntimeError(f"stratified export hit max_draws with cells short: {missing}")
        draws += 1
        sample = generate_sample(rng)
        if sample["split"] != split:
            continue
        cell = (int(sample["y_reservoir"]), int(sample["y_boundary"]))  # type: ignore[call-overload]
        if counts.get(cell, 0) >= n_per_cell:
            continue
        counts[cell] = counts.get(cell, 0) + 1
        xs.append(sample["x"])  # type: ignore[arg-type]
        y_res.append(sample["y_reservoir"])
        y_bnd.append(sample["y_boundary"])
        tgts.append(sample["targets"])
        masks.append(sample["mask"])

    _write_h5(path, xs, y_res, y_bnd, tgts, masks)
    return len(xs)


def _write_h5(
    path: str,
    xs: list[NDArray[np.float32]],
    y_res: list[object],
    y_bnd: list[object],
    tgts: list[object],
    masks: list[object],
) -> None:
    """Write the collected sample arrays to an HDF5 file."""
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.asarray(xs, dtype=np.float32))
        f.create_dataset("y_reservoir", data=np.asarray(y_res, dtype=np.int64))
        f.create_dataset("y_boundary", data=np.asarray(y_bnd, dtype=np.int64))
        f.create_dataset("targets", data=np.asarray(tgts, dtype=np.float64))
        f.create_dataset("mask", data=np.asarray(masks, dtype=np.bool_))
