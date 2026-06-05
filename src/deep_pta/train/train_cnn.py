"""Training loop for the multi-task CNN baseline.

Trains :class:`~deep_pta.models.resnet1d.ResNet1D` on the on-the-fly generator, with
mixed precision on CUDA. Evaluation reports per-head accuracy and confusion matrices on
the frozen test set; a checkpoint and diagnostic figures are written to ``models/`` and
``outputs/``. Also exposes :func:`overfit_one_batch` for the CPU smoke test.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deep_pta.data.sampling import N_BOUNDARY, N_RESERVOIR
from deep_pta.models.losses import LossWeights, multitask_loss
from deep_pta.models.resnet1d import ResNet1D
from deep_pta.train.dataset import FrozenH5Dataset, OnTheFlyDataset


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_one_batch(batch_size: int, seed: int = 0) -> dict[str, Tensor]:
    """Build a single in-memory batch from the on-the-fly generator (for tests)."""
    ds = OnTheFlyDataset(epoch_size=batch_size, split="train", seed=seed)
    loader = DataLoader(ds, batch_size=batch_size)
    return next(iter(loader))


def overfit_one_batch(steps: int = 150, batch_size: int = 16, seed: int = 0) -> float:
    """Overfit a single batch on CPU and return the final loss.

    A sharp drop validates the forward/backward path and the masked loss without a GPU.

    Parameters
    ----------
    steps : int, optional
        Optimization steps, by default 150.
    batch_size : int, optional
        Batch size, by default 16.
    seed : int, optional
        Seed for the batch and model init, by default 0.

    Returns
    -------
    float
        The loss after the final step.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = ResNet1D(base_channels=16, n_blocks=4).to(device)
    batch = _move(make_one_batch(batch_size, seed), device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    model.train()
    loss_val = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        out = model(batch["x"])
        loss, _ = multitask_loss(
            out, batch["y_reservoir"], batch["y_boundary"], batch["targets"], batch["mask"]
        )
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
    return loss_val


def _infinite(loader: DataLoader[dict[str, Tensor]]) -> Iterator[dict[str, Tensor]]:
    while True:
        yield from loader


def train(
    n_steps: int = 3000,
    batch_size: int = 128,
    lr: float = 1e-3,
    base_channels: int = 32,
    n_blocks: int = 6,
    num_workers: int = 4,
    log_every: int = 200,
    seed: int = 0,
) -> ResNet1D:
    """Train the CNN baseline on the on-the-fly generator.

    Parameters
    ----------
    n_steps : int, optional
        Number of optimization steps, by default 3000.
    batch_size : int, optional
        Batch size, by default 128.
    lr : float, optional
        Adam learning rate, by default 1e-3.
    base_channels, n_blocks : int, optional
        Model capacity, by default 32 and 6.
    num_workers : int, optional
        DataLoader workers, by default 4.
    log_every : int, optional
        Logging cadence in steps, by default 200.
    seed : int, optional
        Random seed, by default 0.

    Returns
    -------
    ResNet1D
        The trained model (on its training device).
    """
    torch.manual_seed(seed)
    device = get_device()
    model = ResNet1D(base_channels=base_channels, n_blocks=n_blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weights = LossWeights()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]

    ds = OnTheFlyDataset(epoch_size=n_steps * batch_size, split="train", seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    model.train()
    print(f"training on {device} for {n_steps} steps (batch {batch_size})")
    for step, batch in zip(range(n_steps), _infinite(loader), strict=False):
        batch = _move(batch, device)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore[attr-defined]
            out = model(batch["x"])
            loss, parts = multitask_loss(
                out,
                batch["y_reservoir"],
                batch["y_boundary"],
                batch["targets"],
                batch["mask"],
                weights,
            )
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.step(opt)
        scaler.update()
        if step % log_every == 0:
            print(
                f"step {step:5d} | loss {float(loss.detach()):.4f} | "
                f"ce_res {parts['ce_reservoir']:.3f} ce_bnd {parts['ce_boundary']:.3f} "
                f"reg {parts['reg_nll']:.3f}"
            )
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module, test_h5: str, device: torch.device | None = None
) -> dict[str, object]:
    """Evaluate per-head accuracy and confusion matrices on the frozen test set.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    test_h5 : str
        Path to the frozen test-set HDF5 file.
    device : torch.device, optional
        Evaluation device; defaults to the model's device.

    Returns
    -------
    dict
        ``acc_reservoir``, ``acc_boundary`` floats and ``cm_reservoir``,
        ``cm_boundary`` confusion matrices, plus regression ``mae``.
    """
    device = device or next(model.parameters()).device
    ds = FrozenH5Dataset(test_h5)
    loader = DataLoader(ds, batch_size=256)
    model.eval()

    cm_res = np.zeros((N_RESERVOIR, N_RESERVOIR), dtype=np.int64)
    cm_bnd = np.zeros((N_BOUNDARY, N_BOUNDARY), dtype=np.int64)
    abs_err, n_masked = 0.0, 0.0
    correct_res = correct_bnd = total = 0
    for batch in loader:
        batch = _move(batch, device)
        out = model(batch["x"])
        pred_res = out.logits_reservoir.argmax(1)
        pred_bnd = out.logits_boundary.argmax(1)
        for t, p in zip(batch["y_reservoir"].tolist(), pred_res.tolist(), strict=False):
            cm_res[t, p] += 1
        for t, p in zip(batch["y_boundary"].tolist(), pred_bnd.tolist(), strict=False):
            cm_bnd[t, p] += 1
        correct_res += int((pred_res == batch["y_reservoir"]).sum())
        correct_bnd += int((pred_bnd == batch["y_boundary"]).sum())
        total += batch["x"].shape[0]
        m = batch["mask"]
        abs_err += float((torch.abs(out.params - batch["targets"]) * m).sum())
        n_masked += float(m.sum())

    return {
        "acc_reservoir": correct_res / max(total, 1),
        "acc_boundary": correct_bnd / max(total, 1),
        "mae": abs_err / max(n_masked, 1.0),
        "cm_reservoir": cm_res,
        "cm_boundary": cm_bnd,
    }


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save the model state dict, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
