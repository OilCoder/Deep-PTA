"""Training loop for the multi-task CNN baseline.

Trains :class:`~deep_pta.models.resnet1d.ResNet1D` on the on-the-fly generator, with
mixed precision on CUDA. Evaluation reports per-head accuracy and confusion matrices on
the frozen test set; a checkpoint and diagnostic figures are written to ``models/`` and
``outputs/``. Also exposes :func:`overfit_one_batch` for the CPU smoke test.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deep_pta.data.representation import EXTRA_CHANNELS
from deep_pta.data.sampling import N_BOUNDARY, N_RESERVOIR
from deep_pta.models.losses import LossWeights, multitask_loss
from deep_pta.models.resnet1d import ResNet1D
from deep_pta.train.config import TrainConfig
from deep_pta.train.dataset import FrozenH5Dataset, OnTheFlyDataset


def channel_index(channels: tuple[str, ...] | None) -> tuple[int, ...]:
    """Map a config ``channels`` tuple to stored-channel indices of a superset H5.

    The canonical storage layout is the 3 base channels followed by
    :data:`~deep_pta.data.representation.EXTRA_CHANNELS` in order, so e.g.
    ``("sep", "slope")`` maps to ``(0, 1, 2, 3, 4)`` and ``None`` (legacy
    3-channel input) maps to ``(0, 1, 2)``.

    Parameters
    ----------
    channels : tuple of str or None
        Extra channel names from :class:`~deep_pta.train.config.TrainConfig`.

    Returns
    -------
    tuple of int
        Indices into the stored ``x`` channel axis.
    """
    extras = tuple(3 + EXTRA_CHANNELS.index(name) for name in (channels or ()))
    return (0, 1, 2, *extras)


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


def _class_metrics(cm: np.ndarray) -> dict[str, object]:
    """Derive balanced metrics from a confusion matrix.

    The confusion matrix follows the ``cm[true, pred]`` convention, so the
    row sums are the per-class support. Balanced accuracy is the mean per-class
    recall, which is unaffected by class imbalance (unlike raw accuracy).

    Parameters
    ----------
    cm : numpy.ndarray
        Square confusion matrix of integer counts, shape ``(C, C)``.

    Returns
    -------
    dict
        ``recall`` (per-class), ``support`` (per-class), ``balanced_accuracy``
        (mean recall), and ``macro_f1`` (mean per-class F1).
    """
    support = cm.sum(axis=1)
    pred_total = cm.sum(axis=0)
    diag = np.diag(cm).astype(np.float64)

    recall = np.divide(diag, support, out=np.zeros_like(diag), where=support > 0)
    precision = np.divide(diag, pred_total, out=np.zeros_like(diag), where=pred_total > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(diag), where=denom > 0)

    return {
        "recall": recall.tolist(),
        "support": support.astype(np.int64).tolist(),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_h5: str,
    device: torch.device | None = None,
    channel_idx: tuple[int, ...] | None = None,
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
    channel_idx : tuple of int, optional
        Stored channels to keep (for running a reduced-channel model on a
        channel-superset H5). ``None`` keeps all stored channels.

    Returns
    -------
    dict
        ``acc_reservoir``, ``acc_boundary`` (raw accuracy) and ``cm_reservoir``,
        ``cm_boundary`` confusion matrices, plus regression ``mae``. Also
        includes honest, imbalance-robust metrics per head: ``bal_acc_reservoir``,
        ``bal_acc_boundary`` (balanced accuracy), ``macro_f1_reservoir``,
        ``macro_f1_boundary``, and ``recall_reservoir``, ``recall_boundary``,
        ``support_reservoir``, ``support_boundary``.
    """
    device = device or next(model.parameters()).device
    ds = FrozenH5Dataset(test_h5, channel_idx=channel_idx)
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

    res_metrics = _class_metrics(cm_res)
    bnd_metrics = _class_metrics(cm_bnd)
    return {
        "acc_reservoir": correct_res / max(total, 1),
        "acc_boundary": correct_bnd / max(total, 1),
        "bal_acc_reservoir": res_metrics["balanced_accuracy"],
        "bal_acc_boundary": bnd_metrics["balanced_accuracy"],
        "macro_f1_reservoir": res_metrics["macro_f1"],
        "macro_f1_boundary": bnd_metrics["macro_f1"],
        "recall_reservoir": res_metrics["recall"],
        "recall_boundary": bnd_metrics["recall"],
        "support_reservoir": res_metrics["support"],
        "support_boundary": bnd_metrics["support"],
        "mae": abs_err / max(n_masked, 1.0),
        "cm_reservoir": cm_res,
        "cm_boundary": cm_bnd,
    }


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save the model state dict, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _warmup_cosine(step: int, n_steps: int, warmup_frac: float) -> float:
    """LR multiplier: linear warmup then cosine decay to ~zero."""
    warmup = max(1, int(n_steps * warmup_frac))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, n_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _class_weights_from_h5(path: str, n_classes: int, key: str, device: torch.device) -> Tensor:
    """Inverse-frequency cross-entropy weights from a frozen set's label support."""
    with h5py.File(path, "r") as f:
        labels = f[key][:]
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    inv = 1.0 / np.clip(counts, 1.0, None)
    weights = inv * n_classes / inv.sum()  # mean weight ~ 1.0
    return torch.tensor(weights, dtype=torch.float32, device=device)


def fit(
    cfg: TrainConfig,
    on_eval: Callable[[int, float], None] | None = None,
    model: nn.Module | None = None,
) -> dict[str, object]:
    """Train a model from a :class:`TrainConfig` with val-based early stopping.

    Uses AdamW with linear-warmup cosine decay, optional dropout, gradient clipping,
    periodic validation on the frozen val set, best-on-val checkpointing, optional
    TensorBoard logging, and a single final evaluation on the frozen test set.

    Parameters
    ----------
    cfg : TrainConfig
        Full run configuration.
    on_eval : Callable[[int, float], None], optional
        Called after each validation as ``on_eval(step, score)`` where ``score`` is
        the mean balanced accuracy; may raise to abort (used for HPO pruning).
    model : torch.nn.Module, optional
        Model to train. If ``None`` (default), a :class:`~deep_pta.models.resnet1d.ResNet1D`
        is built from ``cfg``; pass an instance (e.g. PatchTST) to train another arch
        through the identical loop.

    Returns
    -------
    dict
        ``best_val`` (metrics of the best checkpoint), ``test`` (final test metrics),
        ``best_step``, and ``history`` (list of per-eval val balanced-accuracy means).
    """
    torch.manual_seed(cfg.seed)
    device = get_device()
    ch_idx = channel_index(cfg.channels)
    if model is None:
        model = ResNet1D(
            in_channels=len(ch_idx),
            base_channels=cfg.base_channels,
            n_blocks=cfg.n_blocks,
            dropout=cfg.dropout,
        )
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _warmup_cosine(s, cfg.n_steps, cfg.warmup_frac)
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]

    cw_res = cw_bnd = None
    if cfg.use_class_weights:
        cw_res = _class_weights_from_h5(cfg.val_h5, N_RESERVOIR, "y_reservoir", device)
        cw_bnd = _class_weights_from_h5(cfg.val_h5, N_BOUNDARY, "y_boundary", device)

    writer = _make_tb_writer(cfg.tb_logdir)
    loader: DataLoader[dict[str, Tensor]]
    if cfg.train_h5 is not None:
        # Frozen training set: shuffle and cycle (fast; fixed augmentation).
        loader = DataLoader(
            FrozenH5Dataset(cfg.train_h5, channel_idx=ch_idx),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=True,
        )
    else:
        # On-the-fly generation: infinite augmentation, CPU-bound (workers are the
        # throughput knob; persistent workers + prefetch keep the GPU fed).
        on_the_fly = OnTheFlyDataset(
            epoch_size=cfg.n_steps * cfg.batch_size,
            split="train",
            seed=cfg.seed,
            res_accept=cfg.res_sample_weights,
            cd_max_log_schedule=cfg.cd_max_log_schedule,
            total_steps=cfg.n_steps,
            extra_channels=cfg.channels or (),
        )
        loader = DataLoader(
            on_the_fly,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=4 if cfg.num_workers > 0 else None,
        )

    best_score, best_step, since_best = -1.0, -1, 0
    history: list[float] = []
    print(f"fit: {device}, {cfg.n_steps} steps, batch {cfg.batch_size}")
    model.train()
    for step, batch in zip(range(cfg.n_steps), _infinite(loader), strict=False):
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
                cfg.weights,
                cw_res,
                cw_bnd,
            )
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        if cfg.grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        # Step the scheduler only when the optimizer actually stepped (AMP may skip
        # an early step on inf/nan, which would otherwise warn and desync the schedule).
        if scaler.get_scale() >= scale_before:
            scheduler.step()

        if writer is not None and step % cfg.log_every == 0:
            writer.add_scalar("train/loss", float(loss.detach()), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

        if step > 0 and step % cfg.eval_every == 0:
            val = evaluate(model, cfg.val_h5, device=device, channel_idx=ch_idx)
            score = 0.5 * (
                float(val["bal_acc_reservoir"]) + float(val["bal_acc_boundary"])  # type: ignore[arg-type]
            )
            history.append(score)
            print(
                f"step {step:6d} | val bal_res {val['bal_acc_reservoir']:.3f} "
                f"bal_bnd {val['bal_acc_boundary']:.3f} | score {score:.3f}"
            )
            if writer is not None:
                writer.add_scalar("val/bal_acc_reservoir", val["bal_acc_reservoir"], step)
                writer.add_scalar("val/bal_acc_boundary", val["bal_acc_boundary"], step)
                writer.add_scalar("val/score", score, step)
            if score > best_score:
                best_score, best_step, since_best = score, step, 0
                save_checkpoint(model, cfg.ckpt_path)
            else:
                since_best += 1
                if cfg.patience > 0 and since_best >= cfg.patience:
                    print(f"early stopping at step {step} (best {best_score:.3f} @ {best_step})")
                    break
            if on_eval is not None:
                on_eval(step, score)
            model.train()

    if best_step < 0:  # never evaluated (tiny run): save final state
        save_checkpoint(model, cfg.ckpt_path)
    model.load_state_dict(torch.load(cfg.ckpt_path, map_location=device))
    best_val = evaluate(model, cfg.val_h5, device=device, channel_idx=ch_idx)
    test = evaluate(model, cfg.test_h5, device=device, channel_idx=ch_idx)
    if writer is not None:
        writer.close()
    return {"best_val": best_val, "test": test, "best_step": best_step, "history": history}


def _make_tb_writer(logdir: str | None):  # type: ignore[no-untyped-def]
    """Return a TensorBoard ``SummaryWriter`` or ``None`` if disabled/unavailable."""
    if logdir is None:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not available; continuing without tracking")
        return None
    return SummaryWriter(logdir)
