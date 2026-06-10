"""Load real pressure-transient cases and turn them into model input.

Real cases live in ``data/real/`` as ``(t, p)`` CSV files with an optional ground-truth
JSON. This module reuses the synthetic pipeline's Bourdet derivative and representation
so real and synthetic inputs are processed identically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from deep_pta.data.bourdet import bourdet_derivative
from deep_pta.data.representation import build_representation


def representation_from_pressure(
    t: NDArray[np.float64],
    dp: NDArray[np.float64],
    l_window: float = 0.2,
    extra_channels: tuple[str, ...] = (),
) -> NDArray[np.float32]:
    """Build the model input representation from a pressure-change series.

    Parameters
    ----------
    t : numpy.ndarray
        Strictly positive, increasing times.
    dp : numpy.ndarray
        Pressure change at each time.
    l_window : float, optional
        Bourdet smoothing window, by default 0.2.
    extra_channels : tuple of str, optional
        Physics-informed channels appended to the base 3 (``"sep"``, ``"slope"``);
        computed from ``(t, dp)`` + Bourdet only, so real cases need nothing extra.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(3 + len(extra_channels), 256)`` and dtype ``float32``.
    """
    t_der, deriv = bourdet_derivative(t, dp, l_window)
    return build_representation(t, dp, t_der, deriv, extra_channels=extra_channels)


def load_real_case(csv_path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load a ``(t, p)`` CSV case.

    Parameters
    ----------
    csv_path : str
        Path to a two-column CSV (time, pressure change), optional header.

    Returns
    -------
    t : numpy.ndarray
        Times.
    dp : numpy.ndarray
        Pressure change.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=None, skip_header=0)
    data = np.atleast_2d(data)
    # Skip a header row if the first row failed to parse as numbers.
    if np.isnan(data[0]).any():
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        data = np.atleast_2d(data)
    return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)


def load_ground_truth(json_path: str) -> dict[str, object]:
    """Load the optional ground-truth JSON for a real case.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed ground-truth dictionary (empty if the file is missing).
    """
    path = Path(json_path)
    if not path.exists():
        return {}
    return cast("dict[str, object]", json.loads(path.read_text()))
