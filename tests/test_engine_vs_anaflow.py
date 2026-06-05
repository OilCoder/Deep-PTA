"""Cross-check the engine against AnaFlow's Theis solution (when installed).

Skipped cleanly when the optional ``[validation]`` extra (``anaflow``) is absent, so
the suite stays green on the CPU build; run locally with ``uv sync --extra validation``.
"""

from __future__ import annotations

import numpy as np
import pytest

from deep_pta.engine import boundaries as bnd
from deep_pta.engine.solution import evaluate, make_homogeneous

anaflow = pytest.importorskip("anaflow")


def test_homogeneous_matches_anaflow_theis() -> None:
    # Theis problem; convert AnaFlow drawdown to dimensionless p_D and compare to the
    # engine's homogeneous (no storage/skin) solution. With t_D = T t / (S r^2),
    # the Theis well function gives p_D = 1/2 * W(1/(4 t_D)).
    transmissivity = 1e-3
    storage = 1e-4
    rad = 1.0
    rate = -1e-3
    times = np.geomspace(1e2, 1e6, 25)

    drawdown = anaflow.theis(
        time=times,
        rad=rad,
        storage=storage,
        transmissivity=transmissivity,
        rate=rate,
    )
    drawdown = np.asarray(drawdown, dtype=float).ravel()
    # AnaFlow returns a signed head change (negative for the negative production rate);
    # dividing by the signed rate yields the positive dimensionless pressure drop.
    p_d_anaflow = 2.0 * np.pi * transmissivity * drawdown / rate

    t_d = transmissivity * times / (storage * rad**2)
    p_d_engine, _ = evaluate(make_homogeneous(), bnd.infinite(), (0.0, 0.0), t_d)

    np.testing.assert_allclose(p_d_engine, p_d_anaflow, rtol=1e-3)
