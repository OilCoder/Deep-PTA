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
        Which split to keep (``"train"``, ``"val"``, ``"test"``, ``"extrap"``),
        by default ``"train"``.
    seed : int, optional
        Base seed; each worker offsets it, by default 0.
    res_accept : tuple of float, optional
        Per-reservoir-class acceptance probabilities in ``[0, 1]`` (length
        ``N_RESERVOIR``). Samples of class ``c`` are kept with probability
        ``res_accept[c]``; lowering easy classes oversamples the hard ones. ``None``
        (default) accepts every class (uniform).
    cd_max_log_schedule : tuple of float, optional
        ``(start_log, full_log, ramp_frac)`` for a low→high storage curriculum: the
        ``log10`` cap on ``C_D`` ramps linearly from ``start_log`` to ``full_log`` over
        the first ``ramp_frac`` of the epoch, then stays at ``full_log``. ``None``
        (default) disables the curriculum (full range throughout).
    total_steps : int, optional
        Total optimization steps, used to map per-worker progress to the curriculum
        ramp; defaults to ``epoch_size``-derived progress when ``None``.
    """

    def __init__(
        self,
        epoch_size: int,
        split: str = "train",
        seed: int = 0,
        res_accept: tuple[float, ...] | None = None,
        cd_max_log_schedule: tuple[float, float, float] | None = None,
        total_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.epoch_size = epoch_size
        self.split = split
        self.seed = seed
        self.res_accept = res_accept
        self.cd_max_log_schedule = cd_max_log_schedule
        self.total_steps = total_steps

    def _cd_max(self, progress: float) -> float | None:
        """Curriculum ``C_D`` cap for a given epoch ``progress`` in ``[0, 1)``."""
        if self.cd_max_log_schedule is None:
            return None
        start_log, full_log, ramp_frac = self.cd_max_log_schedule
        frac = 1.0 if ramp_frac <= 0 else min(1.0, progress / ramp_frac)
        return float(10.0 ** (start_log + (full_log - start_log) * frac))

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Yield ``epoch_size`` samples for this worker, seeded reproducibly."""
        info = get_worker_info()
        worker_id = 0 if info is None else info.id
        n_workers = 1 if info is None else info.num_workers
        rng = np.random.default_rng(self.seed + 1000 * (worker_id + 1))
        n = self.epoch_size // n_workers
        produced = 0
        while produced < n:
            cd_max = self._cd_max(produced / n)
            try:
                sample = generate_sample(rng, cd_max=cd_max)
            except RuntimeError:
                # A rare degenerate draw must not kill the worker; skip it.
                continue
            if sample["split"] != self.split:
                continue
            if self.res_accept is not None:
                accept = self.res_accept[int(sample["y_reservoir"])]  # type: ignore[call-overload]
                if accept < 1.0 and rng.random() >= accept:
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
