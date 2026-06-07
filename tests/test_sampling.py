"""Tests for parameter sampling, factored labels, masks, and the validity filter."""

from __future__ import annotations

import numpy as np

from deep_pta.data.sampling import (
    BND_INFINITE,
    N_BOUNDARY,
    N_PARAMS,
    N_RESERVOIR,
    sample_curve,
)


def test_sample_structure_and_ranges() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        cp = sample_curve(rng)
        assert 0 <= cp.reservoir_class < N_RESERVOIR
        assert 0 <= cp.boundary_class < N_BOUNDARY
        assert cp.targets.shape == (N_PARAMS,)
        assert cp.mask.shape == (N_PARAMS,)
        assert cp.t_d[0] < cp.t_d[-1]


def test_inactive_targets_are_zero() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        cp = sample_curve(rng)
        assert np.all(cp.targets[~cp.mask] == 0.0)
        # Storage and skin (indices 0, 1) are always active.
        assert cp.mask[0] and cp.mask[1]


def test_validity_filter_relabels_infinite_when_param_absent() -> None:
    # When the boundary parameter was not stored, the class must be infinite.
    rng = np.random.default_rng(2)
    for _ in range(300):
        cp = sample_curve(rng)
        if "L_D" not in cp.raw and "r_eD" not in cp.raw:
            assert cp.boundary_class == BND_INFINITE


def test_boundary_distribution_is_balanced() -> None:
    # The conditioned-window sampler must not collapse most finite boundaries to
    # "infinite"; each class should retain a sizeable share (was ~70% infinite).
    rng = np.random.default_rng(3)
    counts = np.zeros(N_BOUNDARY, dtype=np.int64)
    n = 2000
    for _ in range(n):
        counts[sample_curve(rng).boundary_class] += 1
    frac = counts / n
    assert frac[BND_INFINITE] < 0.5, f"infinite still dominates: {frac}"
    assert np.all(frac > 0.1), f"a boundary class is starved: {frac}"


def test_reproducible_with_seed() -> None:
    a = sample_curve(np.random.default_rng(7))
    b = sample_curve(np.random.default_rng(7))
    assert a.reservoir_class == b.reservoir_class
    assert a.boundary_class == b.boundary_class
    np.testing.assert_array_equal(a.targets, b.targets)
