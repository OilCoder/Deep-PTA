"""Certify the analytical engine against known diagnostic signatures.

The engine is certified by the diagnostic slopes and plateaus its Bourdet derivative
must reproduce (the senior interpreter's reading), plus an exact analytical reference:
the line-source (Theis) infinite-acting solution ``p_D = 1/2 E_1(1/(4 t_D))``.

Point-by-point matching against digitized published type curves
([Bourdet1983]_, [Gringarten1974]_) is a future addition once the reference CSVs are
digitized into ``tests/data/typecurves/``; the diagnostic slopes certified here are
the physics those curves encode.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import exp1

from deep_pta.engine import boundaries as bnd
from deep_pta.engine.solution import (
    evaluate,
    make_double_porosity,
    make_finite_conductivity_fracture,
    make_homogeneous,
    make_infinite_conductivity_fracture,
)


def _loglog_slope(t: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    return float(np.polyfit(np.log10(t), np.log10(y), 1)[0])


def test_homogeneous_matches_theis_line_source() -> None:
    # No storage/skin: late-time pressure must match the Theis line-source solution.
    t_d = np.logspace(2, 5, 30)
    p_wd, _ = evaluate(make_homogeneous(), bnd.infinite(), (0.0, 0.0), t_d)
    theis = 0.5 * exp1(1.0 / (4.0 * t_d))
    # The engine uses a cylindrical (finite-wellbore) source; it converges to the
    # Theis line source at large t_D, with ~0.5% difference near t_D = 100.
    np.testing.assert_allclose(p_wd, theis, rtol=1e-2)


def test_homogeneous_radial_derivative_plateau() -> None:
    # Radial flow: Bourdet derivative stabilizes at 0.5.
    t_d = np.logspace(1, 6, 40)
    _, dp = evaluate(make_homogeneous(), bnd.infinite(), (0.0, 0.0), t_d)
    np.testing.assert_allclose(dp[-10:], 0.5, rtol=1e-2)


def test_double_porosity_two_plateaus_and_valley() -> None:
    # Warren-Root: two ~0.5 plateaus separated by a transition valley.
    t_d = np.logspace(0, 7, 160)
    _, dp = evaluate(make_double_porosity(0.05, 1e-6), bnd.infinite(), (0.0, 0.0), t_d)
    assert np.isclose(np.median(dp[-10:]), 0.5, atol=0.03)
    assert dp.min() < 0.25  # transition valley well below the plateau


def test_infinite_conductivity_fracture_half_slope() -> None:
    # Linear flow: early-time derivative slope is 1/2.
    t_d = np.logspace(-4, -1, 40)
    _, dp = evaluate(make_infinite_conductivity_fracture(), bnd.infinite(), (0.0, 0.0), t_d)
    assert abs(_loglog_slope(t_d[:12], dp[:12]) - 0.5) < 0.05


def test_finite_conductivity_fracture_bilinear_slope() -> None:
    # Bilinear flow (low F_CD): early-time derivative slope near 1/4, below the
    # 1/2 linear slope.
    t_d = np.logspace(-4, -1, 50)
    _, dp = evaluate(make_finite_conductivity_fracture(0.5), bnd.infinite(), (0.0, 0.0), t_d)
    slope = _loglog_slope(t_d[:12], dp[:12])
    assert 0.2 < slope < 0.38


def test_sealing_fault_doubles_plateau() -> None:
    # Sealing fault: derivative rises from the 0.5 radial plateau toward 1.0.
    t_d = np.logspace(0, 9, 120)
    _, dp = evaluate(make_homogeneous(), bnd.sealing_fault(1000.0), (0.0, 0.0), t_d)
    assert dp[-1] > 0.85  # heading to the doubled (1.0) plateau


def test_constant_pressure_derivative_falls() -> None:
    # Constant-pressure boundary: derivative drops below the radial plateau.
    t_d = np.logspace(0, 8, 120)
    _, dp = evaluate(make_homogeneous(), bnd.constant_pressure(1000.0), (0.0, 0.0), t_d)
    assert dp[-1] < 0.25


def test_closed_boundary_unit_slope() -> None:
    # Closed reservoir: late-time pseudo-steady gives a unit-slope derivative.
    t_d = np.logspace(0, 8, 120)
    _, dp = evaluate(make_homogeneous(), bnd.closed(1000.0), (0.0, 0.0), t_d)
    late = t_d > 1e7
    assert abs(_loglog_slope(t_d[late], dp[late]) - 1.0) < 0.1


def test_storage_skin_shifts_pressure_not_late_derivative() -> None:
    # Wellbore storage + skin: early unit-slope storage, late radial plateau 0.5.
    # Large C_D delays radial flow, so the window must reach late dimensionless time.
    t_d = np.logspace(-2, 9, 100)
    _, dp = evaluate(make_homogeneous(), bnd.infinite(), (1000.0, 5.0), t_d)
    np.testing.assert_allclose(dp[-10:], 0.5, rtol=2e-2)
