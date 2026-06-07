"""Tests for the on-the-fly generator and the frozen HDF5 test-set export."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from deep_pta.data.generator import export_frozen_test_set, generate_sample
from deep_pta.data.sampling import N_PARAMS


def test_generate_sample_shapes_and_labels() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = generate_sample(rng)
        x = s["x"]
        assert x.shape == (3, 256)
        assert np.isfinite(x).all()
        assert 0 <= int(s["y_reservoir"]) < 4
        assert 0 <= int(s["y_boundary"]) < 4
        assert s["targets"].shape == (N_PARAMS,)
        assert s["mask"].shape == (N_PARAMS,)
        assert s["split"] in ("train", "val", "test")


def test_generate_sample_reproducible() -> None:
    a = generate_sample(np.random.default_rng(123))
    b = generate_sample(np.random.default_rng(123))
    np.testing.assert_array_equal(a["x"], b["x"])
    assert a["y_reservoir"] == b["y_reservoir"]
    assert a["y_boundary"] == b["y_boundary"]


def test_export_frozen_test_set(tmp_path: Path) -> None:
    path = str(tmp_path / "synthetic_test.h5")
    n = 12
    written = export_frozen_test_set(path, n=n, seed=1)
    assert written == n
    with h5py.File(path, "r") as f:
        assert f["x"].shape == (n, 3, 256)
        assert f["y_reservoir"].shape == (n,)
        assert f["targets"].shape == (n, N_PARAMS)
        assert f["mask"].shape == (n, N_PARAMS)
