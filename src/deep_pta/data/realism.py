"""Turn a clean dimensionless pressure curve into a realistic gauge record.

Applied in order to the clean ``(t_D, p_wD)`` curve:

1. **Irregular time sampling** — denser early, sparser late (gauge cadence).
2. **Gauge noise** — Gaussian, amplitude relative to the signal.
3. **Thermal drift** — a smooth random walk added to the pressure.
4. **Truncation** — the test may be cut short.
5. **Outliers** — a small fraction of spike points.

All randomness comes from a seeded :class:`numpy.random.Generator`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def apply_realism(
    rng: np.random.Generator,
    t_d: NDArray[np.float64],
    p_wd: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply realistic acquisition effects to a clean pressure curve.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator.
    t_d : numpy.ndarray
        Clean dimensionless times (monotone increasing).
    p_wd : numpy.ndarray
        Clean dimensionless wellbore pressure on ``t_d``.

    Returns
    -------
    t_obs : numpy.ndarray
        Irregular observed times (subset/jitter of ``t_d``).
    p_obs : numpy.ndarray
        Noisy observed pressure on ``t_obs``.
    """
    # 1. Irregular time sampling on a jittered log grid within the clean span.
    n_obs = int(rng.integers(100, 601))
    log_lo, log_hi = np.log10(t_d[0]), np.log10(t_d[-1])
    base = np.linspace(log_lo, log_hi, n_obs)
    jitter = rng.normal(0.0, 0.3 * (log_hi - log_lo) / n_obs, n_obs)
    log_t = np.sort(np.clip(base + jitter, log_lo, log_hi))
    t_obs = 10.0**log_t
    p_obs = np.interp(t_obs, t_d, p_wd)

    # 2. Gauge noise: Gaussian relative to the signal level.
    frac = rng.uniform(0.003, 0.03)
    sigma = frac * float(np.median(np.abs(p_wd))) + 1e-3
    p_obs = p_obs + rng.normal(0.0, sigma, t_obs.shape)

    # 3. Thermal drift: smooth random walk scaled to a small fraction of the signal.
    walk = np.cumsum(rng.normal(0.0, 1.0, t_obs.shape))
    walk = walk / (np.max(np.abs(walk)) + 1e-12)
    p_obs = p_obs + rng.uniform(0.0, 0.02) * float(np.median(np.abs(p_wd))) * walk

    # 4. Truncation: occasionally keep only an early fraction of the test.
    if rng.uniform() < 0.3:
        keep = int(rng.uniform(0.5, 0.95) * t_obs.size)
        keep = max(keep, 50)
        t_obs, p_obs = t_obs[:keep], p_obs[:keep]

    # 5. Outliers: a small fraction of spike points.
    n_out = int(rng.uniform(0.0, 0.02) * t_obs.size)
    if n_out > 0:
        idx = rng.choice(t_obs.size, size=n_out, replace=False)
        p_obs[idx] += rng.normal(0.0, 8.0 * sigma, n_out)

    return t_obs, p_obs
