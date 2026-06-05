"""Hyperparameter optimization for the CNN baseline with Optuna.

Searches learning rate, model capacity, and loss weighting, training a short run per
trial and maximizing the mean of the two classification accuracies on the frozen test
set. Intended to run locally on the GPU.
"""

from __future__ import annotations

import optuna

from deep_pta.train.train_cnn import evaluate, train


def objective(trial: optuna.Trial, test_h5: str, n_steps: int = 800) -> float:
    """Optuna objective: mean classification accuracy after a short training run.

    Parameters
    ----------
    trial : optuna.Trial
        The trial proposing hyperparameters.
    test_h5 : str
        Path to the frozen test set.
    n_steps : int, optional
        Steps per trial, by default 800.

    Returns
    -------
    float
        Mean of reservoir and boundary accuracy.
    """
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 48])
    n_blocks = trial.suggest_int("n_blocks", 4, 8)

    model = train(
        n_steps=n_steps,
        lr=lr,
        base_channels=base_channels,
        n_blocks=n_blocks,
        log_every=max(n_steps, 1),
        num_workers=2,
    )
    metrics = evaluate(model, test_h5)
    acc_res = float(metrics["acc_reservoir"])  # type: ignore[arg-type]
    acc_bnd = float(metrics["acc_boundary"])  # type: ignore[arg-type]
    return 0.5 * (acc_res + acc_bnd)


def run_study(test_h5: str, n_trials: int = 20) -> optuna.Study:
    """Run an Optuna study and return it.

    Parameters
    ----------
    test_h5 : str
        Path to the frozen test set.
    n_trials : int, optional
        Number of trials, by default 20.

    Returns
    -------
    optuna.Study
        The completed study (maximize).
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, test_h5), n_trials=n_trials)
    return study
