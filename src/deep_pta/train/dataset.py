"""PyTorch dataset wrappers around the NumPy generator.

Two datasets:

* :class:`OnTheFlyDataset` — an ``IterableDataset`` that generates fresh labelled
  curves each step (infinite augmentation), seeded per worker for reproducibility and
  filtered to a chosen split.
* :class:`FrozenH5Dataset` — a map-style dataset over the frozen HDF5 test set.
"""

from __future__ import annotations

from typing import cast

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from deep_pta.data.generator import generate_sample


def _to_tensors(sample: dict[str, object]) -> dict[str, Tensor]:
    return {
        "x": torch.from_numpy(np.asarray(sample["x"], dtype=np.float32)),
        "y_reservoir": torch.tensor(cast(int, sample["y_reservoir"]), dtype=torch.long),
        "y_boundary": torch.tensor(cast(int, sample["y_boundary"]), dtype=torch.long),
        "targets": torch.from_numpy(np.asarray(sample["targets"], dtype=np.float32)),
        "mask": torch.from_numpy(np.asarray(sample["mask"], dtype=np.float32)),
    }


class OnTheFlyDataset(IterableDataset[dict[str, Tensor]]):
    """Infinite on-the-fly generator of labelled samples for one split.

    Parameters
    ----------
    epoch_size : int
        Number of samples yielded per epoch (split across workers).
    split : str, optional
        Which split to keep (``"train"`` or ``"test"``), by default ``"train"``.
    seed : int, optional
        Base seed; each worker offsets it, by default 0.
    """

    def __init__(self, epoch_size: int, split: str = "train", seed: int = 0) -> None:
        super().__init__()
        self.epoch_size = epoch_size
        self.split = split
        self.seed = seed

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Yield ``epoch_size`` samples for this worker, seeded reproducibly."""
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        n_workers = 1 if info is None else info.num_workers
        rng = np.random.default_rng(self.seed + 1000 * (worker_id + 1))
        n = self.epoch_size // n_workers
        produced = 0
        while produced < n:
            try:
                sample = generate_sample(rng)
            except RuntimeError:
                # A rare degenerate draw must not kill the worker; skip it.
                continue
            if sample["split"] != self.split:
                continue
            produced += 1
            yield _to_tensors(sample)


class FrozenH5Dataset(Dataset[dict[str, Tensor]]):
    """Map-style dataset over a frozen HDF5 test set.

    Parameters
    ----------
    path : str
        Path to the ``.h5`` file produced by ``export_frozen_test_set``.
    """

    def __init__(self, path: str) -> None:
        with h5py.File(path, "r") as f:
            self.x = f["x"][:]
            self.y_reservoir = f["y_reservoir"][:]
            self.y_boundary = f["y_boundary"][:]
            self.targets = f["targets"][:]
            self.mask = f["mask"][:]

    def __len__(self) -> int:
        """Return the number of stored samples."""
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return the sample at ``idx`` as tensors."""
        return {
            "x": torch.from_numpy(self.x[idx].astype(np.float32)),
            "y_reservoir": torch.tensor(int(self.y_reservoir[idx]), dtype=torch.long),
            "y_boundary": torch.tensor(int(self.y_boundary[idx]), dtype=torch.long),
            "targets": torch.from_numpy(self.targets[idx].astype(np.float32)),
            "mask": torch.from_numpy(self.mask[idx].astype(np.float32)),
        }
