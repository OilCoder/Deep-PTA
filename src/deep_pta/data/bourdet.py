"""Bourdet derivative with a variable smoothing window, and Agarwal equivalent time.

The Bourdet derivative [Bourdet1989]_ is ``t * d(Delta p)/dt = d(Delta p)/d(ln t)``,
computed with the three-point weighted rule in log-time using a smoothing window ``L``
(in log cycles). ``L`` is deliberately variable (0.1-0.3) because the real
interpretation problem depends on the chosen smoothing.

Buildup data are mapped onto drawdown type curves via Agarwal equivalent time
[Agarwal1980]_: ``dt_e = dt * t_p / (t_p + dt)``.

References
----------
.. [Bourdet1989] Bourdet, D., Ayoub, J.A. & Pirard, Y.M. (1989). Use of Pressure
   Derivative in Well-Test Interpretation. SPEFE 4(2), 293-302.
.. [Agarwal1980] Agarwal, R.G. (1980). A New Method To Account for Producing Time
   Effects When Drawdown Type Curves Are Used To Analyze Pressure Buildup and Other
   Test Data. SPE 9289.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def agarwal_equivalent_time(dt: NDArray[np.float64], t_p: float) -> NDArray[np.float64]:
    """Agarwal equivalent time for buildup analysis.

    Parameters
    ----------
    dt : numpy.ndarray
        Shut-in time(s).
    t_p : float
        Producing time before shut-in (> 0).

    Returns
    -------
    numpy.ndarray
        Equivalent time ``dt * t_p / (t_p + dt)``.
    """
    return dt * t_p / (t_p + dt)


def bourdet_derivative(
    t: NDArray[np.float64],
    dp: NDArray[np.float64],
    l_window: float = 0.2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the Bourdet derivative with the three-point log-window rule.

    Parameters
    ----------
    t : numpy.ndarray
        Strictly positive, increasing times.
    dp : numpy.ndarray
        Pressure change at each time.
    l_window : float, optional
        Smoothing window in natural-log cycles (0.1-0.3 typical), by default 0.2.

    Returns
    -------
    t_mid : numpy.ndarray
        Times where the derivative is defined (interior points with a full window).
    deriv : numpy.ndarray
        Bourdet derivative ``t d(dp)/dt`` at ``t_mid``.
    """
    ln_t = np.log(t)
    n = t.size
    t_out, d_out = [], []
    for i in range(1, n - 1):
        # Left point: farthest j < i within the window, else nearest.
        j = i - 1
        while j > 0 and (ln_t[i] - ln_t[j]) < l_window:
            j -= 1
        # Right point: farthest k > i within the window, else nearest.
        k = i + 1
        while k < n - 1 and (ln_t[k] - ln_t[i]) < l_window:
            k += 1

        dx_l = ln_t[i] - ln_t[j]
        dx_r = ln_t[k] - ln_t[i]
        if dx_l <= 0.0 or dx_r <= 0.0:
            continue
        slope_l = (dp[i] - dp[j]) / dx_l
        slope_r = (dp[k] - dp[i]) / dx_r
        deriv = (slope_l * dx_r + slope_r * dx_l) / (dx_l + dx_r)
        t_out.append(t[i])
        d_out.append(deriv)

    return np.asarray(t_out, dtype=np.float64), np.asarray(d_out, dtype=np.float64)
