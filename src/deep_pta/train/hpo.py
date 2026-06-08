"""Hyperparameter optimization for the CNN with Optuna.

Searches optimization, regularization, capacity, and loss-weight knobs, training a
short run per trial and maximizing **balanced** classification accuracy on the frozen
**validation** set (never the test set, which is touched once at the end). Uses median
pruning and SQLite persistence so studies resume. Intended to run locally on the GPU.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import optuna

from deep_pta.models.losses import LossWeights
from deep_pta.train.config import TrainConfig
from deep_pta.train.train_cnn import fit


def _trial_config(trial: optuna.Trial, base: TrainConfig, n_steps: int) -> TrainConfig:
    """Build a TrainConfig for a trial from the proposed hyperparameters."""
    return replace(
        base,
        n_steps=n_steps,
        lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        warmup_frac=trial.suggest_float("warmup_frac", 0.0, 0.15),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        # Capacity search widened for the on-the-fly (unlimited-data) regime, where
        # bigger models stop overfitting the frozen set and can become justified.
        base_channels=trial.suggest_categorical("base_channels", [32, 48, 64, 96, 128]),
        n_blocks=trial.suggest_int("n_blocks", 4, 12),
        batch_size=trial.suggest_categorical("batch_size", [128, 256]),
        weights=LossWeights(
            reservoir=trial.suggest_float("w_reservoir", 0.5, 2.0),
            boundary=trial.suggest_float("w_boundary", 0.5, 2.0),
            params=trial.suggest_float("w_params", 0.2, 1.5),
        ),
        # Short trials: evaluate often, no early-stop, scratch checkpoint.
        eval_every=max(1, n_steps // 5),
        patience=0,
        ckpt_path="models/_hpo_trial.pt",
        tb_logdir=None,
    )


def objective(trial: optuna.Trial, base: TrainConfig, n_steps: int = 2000) -> float:
    """Optuna objective: mean balanced accuracy on the validation set.

    Parameters
    ----------
    trial : optuna.Trial
        The trial proposing hyperparameters.
    base : TrainConfig
        Base config supplying the val/test paths and fixed settings.
    n_steps : int, optional
        Steps per trial, by default 2000.

    Returns
    -------
    float
        Mean of reservoir and boundary balanced accuracy on the val set.
    """
    cfg = _trial_config(trial, base, n_steps)

    def _report(step: int, score: float) -> None:
        trial.report(score, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    result = fit(cfg, on_eval=_report)
    val: dict[str, Any] = result["best_val"]  # type: ignore[assignment]
    return 0.5 * (float(val["bal_acc_reservoir"]) + float(val["bal_acc_boundary"]))


def run_study(
    base: TrainConfig | None = None,
    n_trials: int = 40,
    n_steps: int = 2000,
    storage: str = "sqlite:///outputs/hpo.db",
    study_name: str = "deep_pta_cnn",
) -> optuna.Study:
    """Run (or resume) an Optuna study and return it.

    Parameters
    ----------
    base : TrainConfig, optional
        Base config; defaults to a fresh :class:`TrainConfig`.
    n_trials : int, optional
        Number of trials, by default 40.
    n_steps : int, optional
        Steps per trial, by default 2000.
    storage : str, optional
        Optuna storage URL for persistence/resume, by default a local SQLite file.
    study_name : str, optional
        Study name, by default ``"deep_pta_cnn"``.

    Returns
    -------
    optuna.Study
        The study after optimization (maximize, median pruning).
    """
    base = base or TrainConfig()
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(lambda t: objective(t, base, n_steps), n_trials=n_trials)
    return study


def best_config(study: optuna.Study, base: TrainConfig | None = None) -> TrainConfig:
    """Map a study's best parameters onto a full :class:`TrainConfig` for the final run."""
    base = base or TrainConfig()
    p = study.best_params
    return replace(
        base,
        lr=p["lr"],
        weight_decay=p["weight_decay"],
        warmup_frac=p["warmup_frac"],
        dropout=p["dropout"],
        base_channels=p["base_channels"],
        n_blocks=p["n_blocks"],
        batch_size=p["batch_size"],
        weights=LossWeights(p["w_reservoir"], p["w_boundary"], p["w_params"]),
    )
