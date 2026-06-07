"""Build the 3-channel log-log representation consumed by the neural models.

The pressure change and its Bourdet derivative are interpolated onto a fixed
256-point logarithmic time grid spanning the overlapping observed window, stacked with
a third absolute-time channel as ``[log10(Delta p), log10(t d(Delta p)/dt), log10(t_D)]``.
The first two channels are per-curve standardized (zero mean, unit variance) so the model
learns the scale-invariant log-log **shape**. The third channel is normalized by **fixed
global constants** (not per-curve), so it preserves the *absolute* dimensionless-time
position — the context that per-curve standardization destroys, which the model needs to
tell an early wellbore-storage hump apart from a late boundary signature.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

N_GRID = 256
_FLOOR = 1e-6

# Fixed normalization for the absolute-time channel: log10(t_D) spans ~[-2, 8]
# across the sampled windows, so center at 1e3 and scale by ~one decade-block.
_LOG_T_MEAN = 3.0
_LOG_T_STD = 3.0


def _safe_log10(y: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log10(np.clip(y, _FLOOR, None))


def _standardize(channel: NDArray[np.float64]) -> NDArray[np.float64]:
    mu = float(channel.mean())
    sd = float(channel.std())
    return (channel - mu) / sd if sd > 1e-12 else channel - mu


def build_representation(
    t_p: NDArray[np.float64],
    dp: NDArray[np.float64],
    t_der: NDArray[np.float64],
    deriv: NDArray[np.float64],
    n_grid: int = N_GRID,
) -> NDArray[np.float32]:
    """Interpolate and stack pressure, derivative, and absolute time into a 3-channel tensor.

    Parameters
    ----------
    t_p, dp : numpy.ndarray
        Observed times and pressure change.
    t_der, deriv : numpy.ndarray
        Times and Bourdet derivative values.
    n_grid : int, optional
        Number of log-time grid points, by default 256.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3, n_grid)`` and dtype ``float32``. Channels 0-1
        (pressure, derivative) are standardized per curve; channel 2
        (absolute log-time) is normalized by fixed global constants.
    """
    lo = max(float(t_p.min()), float(t_der.min()))
    hi = min(float(t_p.max()), float(t_der.max()))
    if not hi > lo:
        raise ValueError("pressure and derivative time spans do not overlap")

    grid = np.logspace(np.log10(lo), np.log10(hi), n_grid)
    ch_p = _standardize(_safe_log10(np.interp(grid, t_p, dp)))
    ch_d = _standardize(_safe_log10(np.interp(grid, t_der, deriv)))
    ch_t = (np.log10(grid) - _LOG_T_MEAN) / _LOG_T_STD
    return np.stack([ch_p, ch_d, ch_t], axis=0).astype(np.float32)
