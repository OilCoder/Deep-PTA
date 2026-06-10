"""Honest CNN-vs-Transformer comparison under identical conditions.

Trains the ResNet-1D and the hand-built PatchTST with the same generator, seed, and
optimization budget, evaluates both on the frozen test set, and saves attention maps
over the derivative to inspect whether the Transformer attends to regime transitions.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from deep_pta.models.losses import multitask_loss
from deep_pta.models.patchtst import PatchTST1D
from deep_pta.models.resnet1d import ResNet1D
from deep_pta.train.dataset import OnTheFlyDataset
from deep_pta.train.train_cnn import evaluate, get_device


def train_model(
    model: nn.Module,
    n_steps: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    num_workers: int = 4,
    seed: int = 0,
) -> nn.Module:
    """Train any multi-task model on the on-the-fly generator.

    Parameters
    ----------
    model : nn.Module
        A model returning :class:`~deep_pta.models.resnet1d.ModelOutput`.
    n_steps, batch_size, lr, num_workers, seed
        Optimization settings (shared across compared models).

    Returns
    -------
    nn.Module
        The trained model.
    """
    torch.manual_seed(seed)
    device = get_device()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = OnTheFlyDataset(epoch_size=n_steps * batch_size, split="train", seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    model.train()
    for _, batch in zip(range(n_steps), loader, strict=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        opt.zero_grad()
        out = model(batch["x"])
        loss, _ = multitask_loss(
            out, batch["y_reservoir"], batch["y_boundary"], batch["targets"], batch["mask"]
        )
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    return model


def compare(test_h5: str, n_steps: int = 2000, seed: int = 0) -> dict[str, dict[str, object]]:
    """Train and evaluate both models under identical conditions.

    Parameters
    ----------
    test_h5 : str
        Path to the frozen test set.
    n_steps : int, optional
        Shared training budget, by default 2000.
    seed : int, optional
        Shared seed, by default 0.

    Returns
    -------
    dict
        ``{"cnn": metrics, "transformer": metrics}``.
    """
    cnn = train_model(ResNet1D(), n_steps=n_steps, seed=seed)
    tst = train_model(PatchTST1D(), n_steps=n_steps, seed=seed)
    return {
        "cnn": evaluate(cnn, test_h5),
        "transformer": evaluate(tst, test_h5),
    }


@torch.no_grad()
def save_attention_figure(model: PatchTST1D, x: Tensor, path: str) -> None:
    """Save a heatmap of the last-block mean attention for one sample.

    Parameters
    ----------
    model : PatchTST1D
        A trained Transformer.
    x : torch.Tensor
        A single-sample batch of shape ``(1, 3, 256)``.
    path : str
        Output image path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = model.attention_maps(x)
    attn = maps[-1][0].mean(0).cpu().numpy()  # (n_patches, n_patches)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    # House sequential cmap per the style guide (docs/guia_estilos.html).
    im = ax.imshow(attn, cmap="Blues")
    ax.set_xlabel("key patch")
    ax.set_ylabel("query patch")
    ax.set_title("PatchTST attention (last block, head-mean)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
