"""Smoke tests for the config-driven training pipeline and HPO study."""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_pta.data.generator import export_frozen_test_set
from deep_pta.train.config import TrainConfig
from deep_pta.train.train_cnn import fit


def _tiny_config(tmp_path: Path) -> TrainConfig:
    """Build tiny frozen val/test sets and a minimal config pointing at them."""
    val = str(tmp_path / "val.h5")
    test = str(tmp_path / "test.h5")
    export_frozen_test_set(val, n=16, seed=1)
    export_frozen_test_set(test, n=16, seed=2)
    return TrainConfig(
        n_steps=40,
        batch_size=16,
        num_workers=0,
        eval_every=20,
        log_every=20,
        patience=0,
        base_channels=16,
        n_blocks=4,
        val_h5=val,
        test_h5=test,
        ckpt_path=str(tmp_path / "ckpt.pt"),
    )


def test_fit_runs_and_returns_metrics(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    result = fit(cfg)
    assert Path(cfg.ckpt_path).exists()
    for key in ("best_val", "test", "best_step", "history"):
        assert key in result
    test = result["test"]
    assert 0.0 <= float(test["bal_acc_reservoir"]) <= 1.0  # type: ignore[call-overload,index]
    assert 0.0 <= float(test["bal_acc_boundary"]) <= 1.0  # type: ignore[call-overload,index]


def test_hpo_study_runs_one_trial(tmp_path: Path) -> None:
    optuna = pytest.importorskip("optuna")
    from deep_pta.train.hpo import run_study

    base = _tiny_config(tmp_path)
    storage = f"sqlite:///{tmp_path / 'hpo.db'}"
    study = run_study(base=base, n_trials=1, n_steps=20, storage=storage, study_name="smoke")
    assert isinstance(study, optuna.Study)
    assert len(study.trials) == 1
