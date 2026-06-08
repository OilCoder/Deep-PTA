"""Single entrypoint: optional HPO, final training, evaluation, and figures.

Run with ``python -m deep_pta.train``. Builds a :class:`TrainConfig`, optionally runs
an Optuna study to pick the best config, trains the final model with early stopping,
evaluates once on the frozen test set, and writes ``outputs/metrics.json`` plus
row-normalized confusion-matrix figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from deep_pta.train.config import TrainConfig
from deep_pta.train.train_cnn import fit


def _json_metrics(m: dict[str, Any]) -> dict[str, Any]:
    """Keep only the JSON-serializable scalar/list fields from an evaluate() dict."""
    return {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}


def save_confusion_figure(cm: np.ndarray, labels: list[str], title: str, path: str) -> None:
    """Save a row-normalized confusion-matrix heatmap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros(cm.shape, dtype=float), where=row > 0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


RES_LABELS = ["homogeneous", "double poros", "inf fracture", "fin fracture"]
BND_LABELS = ["infinite", "sealing fault", "const press", "closed"]


def main() -> None:
    """Train the final CNN (optionally after HPO) and write metrics + figures."""
    parser = argparse.ArgumentParser(description="Deep PTA CNN training entrypoint")
    parser.add_argument("--steps", type=int, default=20_000, help="final-run steps")
    parser.add_argument("--hpo-trials", type=int, default=0, help="Optuna trials (0 skips HPO)")
    parser.add_argument("--hpo-steps", type=int, default=2000, help="steps per HPO trial")
    parser.add_argument("--tb", action="store_true", help="enable TensorBoard logging")
    parser.add_argument("--train-h5", default=None, help="frozen train set (else on-the-fly)")
    parser.add_argument("--val-h5", default="data/synthetic_val.h5")
    parser.add_argument("--test-h5", default="data/synthetic_test_stratified.h5")
    parser.add_argument("--ckpt", default="models/cnn_best.pt")
    parser.add_argument("--class-weights", action="store_true", help="inverse-freq CE weights")
    args = parser.parse_args()

    cfg = TrainConfig(
        n_steps=args.steps,
        train_h5=args.train_h5,
        val_h5=args.val_h5,
        test_h5=args.test_h5,
        ckpt_path=args.ckpt,
        use_class_weights=args.class_weights,
        tb_logdir="outputs/tb" if args.tb else None,
    )

    if args.hpo_trials > 0:
        from deep_pta.train.hpo import best_config, run_study

        study = run_study(base=cfg, n_trials=args.hpo_trials, n_steps=args.hpo_steps)
        print(f"best HPO value {study.best_value:.4f} with {study.best_params}")
        cfg = best_config(study, base=cfg)

    result = fit(cfg)
    test: dict[str, Any] = result["test"]  # type: ignore[assignment]
    best_val: dict[str, Any] = result["best_val"]  # type: ignore[assignment]

    # Report the held-out extrapolation (high-C_D) stress test separately — it is never
    # used for model selection; the in-distribution test is the headline metric.
    extrap: dict[str, Any] | None = None
    if cfg.extrap_test_h5 is not None and Path(cfg.extrap_test_h5).exists():
        import torch

        from deep_pta.models.resnet1d import ResNet1D
        from deep_pta.train.train_cnn import evaluate

        m = ResNet1D(base_channels=cfg.base_channels, n_blocks=cfg.n_blocks)
        m.load_state_dict(torch.load(cfg.ckpt_path, map_location="cpu"))
        extrap = evaluate(m, cfg.extrap_test_h5)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump(
            {
                "best_step": result["best_step"],
                "val": _json_metrics(best_val),
                "test": _json_metrics(test),
                "test_extrap": _json_metrics(extrap) if extrap is not None else None,
            },
            f,
            indent=2,
        )

    save_confusion_figure(
        np.asarray(test["cm_reservoir"]),
        RES_LABELS,
        "CNN reservoir (test)",
        "outputs/cm_cnn_reservoir.png",
    )
    save_confusion_figure(
        np.asarray(test["cm_boundary"]),
        BND_LABELS,
        "CNN boundary (test)",
        "outputs/cm_cnn_boundary.png",
    )
    print(
        f"done | test bal_res {float(test['bal_acc_reservoir']):.3f} "
        f"bal_bnd {float(test['bal_acc_boundary']):.3f}"
    )


if __name__ == "__main__":
    main()
