"""Reproducible training configuration for the multi-task CNN.

A single :class:`TrainConfig` dataclass captures every knob (optimization, schedule,
regularization, capacity, paths) so a run is fully described by one object. The HPO
study writes its best parameters here, and the entrypoint trains the final model from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deep_pta.models.losses import LossWeights


@dataclass
class TrainConfig:
    """All settings for one training run of :class:`~deep_pta.models.resnet1d.ResNet1D`.

    Attributes
    ----------
    n_steps : int
        Optimization steps.
    batch_size : int
        Mini-batch size.
    lr : float
        Peak AdamW learning rate (after warmup).
    weight_decay : float
        AdamW weight decay.
    warmup_frac : float
        Fraction of ``n_steps`` spent in linear LR warmup before cosine decay.
    dropout : float
        Dropout before the heads.
    grad_clip : float
        Max global grad norm (``0`` disables clipping).
    base_channels, n_blocks : int
        Model capacity.
    num_workers : int
        DataLoader workers.
    seed : int
        Random seed.
    log_every, eval_every : int
        Cadence (in steps) for logging and validation evaluation.
    patience : int
        Early-stopping patience in evaluations (``0`` disables it).
    use_class_weights : bool
        Apply inverse-frequency cross-entropy weights from the val support.
    res_sample_weights : tuple of float or None
        Per-reservoir-class acceptance probabilities for on-the-fly oversampling of hard
        classes (length 4). ``None`` keeps the uniform class draw. Use this OR
        ``use_class_weights``, not both.
    cd_max_log_schedule : tuple of float or None
        ``(start_log, full_log, ramp_frac)`` low→high ``C_D`` curriculum for on-the-fly
        training; ``None`` disables it.
    weights : LossWeights
        Relative weights of the three loss terms.
    train_h5 : str or None
        Frozen training set path. If set, train from it (fast, fixed augmentation);
        if ``None``, generate curves on the fly (infinite augmentation, CPU-bound).
    val_h5, test_h5 : str
        Frozen validation / in-distribution test set paths.
    extrap_test_h5 : str or None
        Frozen extrapolation stress-test set (held-out high-``C_D`` band); reported
        separately and never used for model selection. ``None`` skips it.
    ckpt_path : str
        Where to save the best-on-val checkpoint.
    tb_logdir : str or None
        TensorBoard log directory; ``None`` disables tracking.
    """

    n_steps: int = 20_000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    dropout: float = 0.1
    grad_clip: float = 1.0
    base_channels: int = 32
    n_blocks: int = 6
    num_workers: int = 4
    seed: int = 0
    log_every: int = 500
    eval_every: int = 1000
    patience: int = 8
    use_class_weights: bool = False
    res_sample_weights: tuple[float, ...] | None = None
    cd_max_log_schedule: tuple[float, float, float] | None = None
    weights: LossWeights = field(default_factory=LossWeights)
    train_h5: str | None = None
    val_h5: str = "data/synthetic_val.h5"
    test_h5: str = "data/synthetic_test_stratified.h5"
    extrap_test_h5: str | None = "data/synthetic_test_extrapolation.h5"
    ckpt_path: str = "models/cnn_best.pt"
    tb_logdir: str | None = None
