"""Build the multi-channel log-log representation consumed by the neural models.

The pressure change and its Bourdet derivative are interpolated onto a fixed
256-point logarithmic time grid spanning the overlapping observed window, stacked with
a third absolute-time channel as ``[log10(Delta p), log10(t d(Delta p)/dt), log10(t_D)]``.
The first two channels are per-curve standardized (zero mean, unit variance) so the model
learns the scale-invariant log-log **shape**. The third channel is normalized by **fixed
global constants** (not per-curve), so it preserves the *absolute* dimensionless-time
position — the context that per-curve standardization destroys, which the model needs to
tell an early wellbore-storage hump apart from a late boundary signature.

Optional physics-informed extra channels (v3) restore information the per-curve
standardization of channels 0-1 destroys:

* ``"sep"`` — the pressure/derivative separation ``log10(dp) - log10(dp')``, the
  classic manual discriminator: ``0`` on the unit-slope storage line, exactly
  ``log10(2)`` during fracture linear flow, and growing like ``log10(ln t)`` in
  radial flow. Normalized by fixed constants, NOT per curve.
* ``"slope"`` — the local log-log slope of the derivative,
  ``d log10(dp')/d log10(t)``, smoothed and clipped to ``[-2, 2]``. Reads the flow
  regime directly (1 storage, 0.5 linear, 0.25 bilinear, 0 radial) and is invariant
  to log-time shifts, which helps high-storage extrapolation.
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

# Fixed normalization for the separation channel: 0 (storage) .. ~2 (late radial),
# so center at 0.5 and scale by 0.5. NOT per-curve, the absolute level IS the signal.
_SEP_MEAN = 0.5
_SEP_STD = 0.5

# The slope channel is clipped to this physical band (unit slope = 1, noise spikes
# beyond +/-2 carry no regime information) and used unscaled: it is already O(1).
_SLOPE_CLIP = 2.0
_SLOPE_SMOOTH = 9  # moving-average window (grid points) before differentiating

#: Canonical order of the optional extra channels (index = 3 + position).
EXTRA_CHANNELS = ("sep", "slope")


def _safe_log10(y: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.log10(np.clip(y, _FLOOR, None))


def _standardize(channel: NDArray[np.float64]) -> NDArray[np.float64]:
    mu = float(channel.mean())
    sd = float(channel.std())
    return (channel - mu) / sd if sd > 1e-12 else channel - mu


def _smooth(y: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Moving average with edge padding (keeps length and endpoints stable)."""
    if window <= 1:
        return y
    padded = np.pad(y, window // 2, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: y.size]


def _slope_channel(log_t: NDArray[np.float64], log_d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Local log-log slope of the derivative, smoothed and clipped to +/- _SLOPE_CLIP.

    Differentiates first (``np.gradient`` is exact for linear trends, including the
    one-sided edges) and smooths after, so a constant-slope regime reads its exact
    slope across the whole grid — edge-padded smoothing of a constant is a no-op.
    """
    slope = _smooth(np.gradient(log_d, log_t), _SLOPE_SMOOTH)
    return np.clip(slope, -_SLOPE_CLIP, _SLOPE_CLIP)


def build_representation(
    t_p: NDArray[np.float64],
    dp: NDArray[np.float64],
    t_der: NDArray[np.float64],
    deriv: NDArray[np.float64],
    n_grid: int = N_GRID,
    extra_channels: tuple[str, ...] = (),
) -> NDArray[np.float32]:
    """Interpolate and stack pressure, derivative, time, and optional physics channels.

    Parameters
    ----------
    t_p, dp : numpy.ndarray
        Observed times and pressure change.
    t_der, deriv : numpy.ndarray
        Times and Bourdet derivative values.
    n_grid : int, optional
        Number of log-time grid points, by default 256.
    extra_channels : tuple of str, optional
        Physics-informed channels appended after the 3 base ones, in the order
        given. Supported: ``"sep"`` (pressure/derivative separation, fixed-constant
        normalized) and ``"slope"`` (local log-log derivative slope, clipped).
        Default ``()`` keeps the legacy 3-channel output.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3 + len(extra_channels), n_grid)`` and dtype ``float32``.
        Channels 0-1 (pressure, derivative) are standardized per curve; channel 2
        (absolute log-time) and the extras use fixed global normalization so their
        absolute level survives as signal.

    Raises
    ------
    ValueError
        If the time spans do not overlap or an extra channel name is unknown.
    """
    lo = max(float(t_p.min()), float(t_der.min()))
    hi = min(float(t_p.max()), float(t_der.max()))
    if not hi > lo:
        raise ValueError("pressure and derivative time spans do not overlap")

    grid = np.logspace(np.log10(lo), np.log10(hi), n_grid)
    log_p = _safe_log10(np.interp(grid, t_p, dp))
    log_d = _safe_log10(np.interp(grid, t_der, deriv))
    log_t = np.log10(grid)

    channels = [_standardize(log_p), _standardize(log_d), (log_t - _LOG_T_MEAN) / _LOG_T_STD]
    for name in extra_channels:
        if name == "sep":
            channels.append((log_p - log_d - _SEP_MEAN) / _SEP_STD)
        elif name == "slope":
            channels.append(_slope_channel(log_t, log_d))
        else:
            raise ValueError(f"unknown extra channel {name!r}")
    return np.stack(channels, axis=0).astype(np.float32)
