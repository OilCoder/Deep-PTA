"""Inference utilities: diagnose a pressure test and reconstruct the model curve.

Turns a model's head outputs into a human-facing diagnosis (reservoir class, boundary
class, decoded parameters, and a confidence from the classification entropy) and
reconstructs the clean engine derivative for the predicted classes to overlay on the
data in normalized log-log space.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from deep_pta.data.real_cases import representation_from_pressure
from deep_pta.data.representation import build_representation
from deep_pta.data.sampling import (
    BND_CLOSED,
    BND_CONSTANT_PRESSURE,
    BND_SEALING_FAULT,
    RES_DOUBLE_POROSITY,
    RES_FIN_FRACTURE,
    active_param_mask,
)
from deep_pta.engine import boundaries as bnd
from deep_pta.engine.solution import (
    evaluate,
    make_double_porosity,
    make_finite_conductivity_fracture,
    make_homogeneous,
    make_infinite_conductivity_fracture,
)

RES_NAMES = ("homogeneous", "double porosity", "inf-cond fracture", "fin-cond fracture")
BND_NAMES = ("infinite", "sealing fault", "constant pressure", "closed")

# Shipped interpreter (v3 benchmark winner): TCN trained on 5M 5-channel curves —
# chosen for out-of-distribution robustness (extrap 0.62/0.81, near in-dist).
DEFAULT_CHECKPOINT = "models/best.pt"
DEFAULT_ARCH = "tcn"
DEFAULT_EXTRA_CHANNELS = ("sep", "slope")
_PARAM_KEYS = ("log_CD", "S", "log_omega", "log_lambda", "log_LD", "log_FCD", "log_reD")
# Map each target name to the engine parameter name it decodes to.
_OUT_KEY = {
    "log_CD": "C_D",
    "S": "S",
    "log_omega": "omega",
    "log_lambda": "lambda",
    "log_LD": "L_D",
    "log_FCD": "F_CD",
    "log_reD": "r_eD",
}


def _entropy_confidence(logits: Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    entropy = float(-(p * torch.log(p + 1e-12)).sum())
    max_entropy = float(np.log(p.shape[-1]))
    return 1.0 - entropy / max_entropy


def decode_params(
    params: NDArray[np.float64], reservoir_class: int, boundary_class: int
) -> dict[str, float]:
    """Decode active regression outputs into physical-ish parameter values.

    Parameters
    ----------
    params : numpy.ndarray
        Length-7 regression output (log10 except skin).
    reservoir_class, boundary_class : int
        Predicted classes, used to select active parameters.

    Returns
    -------
    dict
        Active parameters: ``C_D``, ``S`` and any of ``omega``, ``lambda``, ``F_CD``,
        ``L_D``, ``r_eD`` that apply.
    """
    mask = active_param_mask(reservoir_class, boundary_class)
    out: dict[str, float] = {}
    for i, key in enumerate(_PARAM_KEYS):
        if not mask[i]:
            continue
        value = float(params[i])
        out[_OUT_KEY[key]] = value if key == "S" else float(10.0**value)
    return out


def reconstruct_derivative(
    reservoir_class: int, boundary_class: int, params: dict[str, float]
) -> NDArray[np.float32]:
    """Rebuild the clean engine derivative for the predicted classes (normalized).

    Parameters
    ----------
    reservoir_class, boundary_class : int
        Predicted classes.
    params : dict
        Decoded parameters from :func:`decode_params`.

    Returns
    -------
    numpy.ndarray
        The derivative channel on the 256-point grid, standardized like the input.
    """
    if reservoir_class == RES_DOUBLE_POROSITY:
        spec = make_double_porosity(params.get("omega", 0.1), params.get("lambda", 1e-6))
    elif reservoir_class == RES_FIN_FRACTURE:
        spec = make_finite_conductivity_fracture(params.get("F_CD", 1.0))
    elif reservoir_class == 2:
        spec = make_infinite_conductivity_fracture()
    else:
        spec = make_homogeneous()

    if boundary_class == BND_SEALING_FAULT:
        boundary = bnd.sealing_fault(params.get("L_D", 1000.0))
    elif boundary_class == BND_CONSTANT_PRESSURE:
        boundary = bnd.constant_pressure(params.get("L_D", 1000.0))
    elif boundary_class == BND_CLOSED:
        boundary = bnd.closed(params.get("r_eD", 1000.0))
    else:
        boundary = bnd.infinite()

    t_d = np.logspace(-1, 7, 256)
    _, dp = evaluate(spec, boundary, (params.get("C_D", 100.0), params.get("S", 0.0)), t_d)
    # Reuse representation standardization by stacking with itself (derivative channel).
    rep = build_representation(t_d, np.clip(dp, 1e-6, None), t_d, np.clip(dp, 1e-6, None))
    return np.asarray(rep[1], dtype=np.float32)


@torch.no_grad()
def diagnose(
    model: nn.Module,
    t: NDArray[np.float64],
    dp: NDArray[np.float64],
    extra_channels: tuple[str, ...] = (),
) -> dict[str, object]:
    """Diagnose a pressure test from its ``(t, dp)`` series.

    Parameters
    ----------
    model : nn.Module
        A trained model returning ``ModelOutput``.
    t, dp : numpy.ndarray
        Times and pressure change.
    extra_channels : tuple of str, optional
        Physics-informed channels the model was trained with (the shipped v3
        interpreter uses :data:`DEFAULT_EXTRA_CHANNELS`). Default ``()`` keeps
        the legacy 3-channel input.

    Returns
    -------
    dict
        ``reservoir``, ``boundary`` (names), ``reservoir_class``, ``boundary_class``,
        ``confidence`` (0-1), ``params`` (decoded), and ``x`` (the model input).
    """
    x = representation_from_pressure(t, dp, extra_channels=extra_channels)
    device = next(model.parameters()).device
    model.eval()
    out = model(torch.from_numpy(x).unsqueeze(0).to(device))
    res_class = int(out.logits_reservoir.argmax(1))
    bnd_class = int(out.logits_boundary.argmax(1))
    conf = 0.5 * (
        _entropy_confidence(out.logits_reservoir[0]) + _entropy_confidence(out.logits_boundary[0])
    )
    params_np = out.params[0].cpu().numpy().astype(np.float64)
    return {
        "reservoir": RES_NAMES[res_class],
        "boundary": BND_NAMES[bnd_class],
        "reservoir_class": res_class,
        "boundary_class": bnd_class,
        "confidence": conf,
        "params": decode_params(params_np, res_class, bnd_class),
        "x": x,
    }


def load_model(
    checkpoint: str = DEFAULT_CHECKPOINT,
    arch: str = DEFAULT_ARCH,
    in_channels: int = 3 + len(DEFAULT_EXTRA_CHANNELS),
    base_channels: int = 32,
    n_blocks: int = 6,
) -> nn.Module:
    """Load a trained interpreter from a checkpoint onto CPU.

    Parameters
    ----------
    checkpoint : str, optional
        Path to the saved state dict; defaults to the shipped v3 interpreter.
    arch : str, optional
        One of ``"tcn"`` (shipped default), ``"resnet1d"``, ``"patchtst"``,
        ``"patchtst_big"``.
    in_channels : int, optional
        Input channel count the checkpoint was trained with (5 for v3).
    base_channels, n_blocks : int, optional
        ResNet capacity (only used for ``arch="resnet1d"``).

    Returns
    -------
    nn.Module
        The model in eval mode.

    Raises
    ------
    ValueError
        If ``arch`` is unknown.
    """
    model: nn.Module
    if arch == "tcn":
        from deep_pta.models.tcn import TCN1D

        model = TCN1D(in_channels=in_channels)
    elif arch == "resnet1d":
        from deep_pta.models.resnet1d import ResNet1D

        model = ResNet1D(in_channels=in_channels, base_channels=base_channels, n_blocks=n_blocks)
    elif arch == "patchtst":
        from deep_pta.models.patchtst import PatchTST1D

        model = PatchTST1D(in_channels=in_channels)
    elif arch == "patchtst_big":
        from deep_pta.models.patchtst import PatchTST1D

        model = PatchTST1D(in_channels=in_channels, d_model=256, n_heads=8, depth=8)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model
