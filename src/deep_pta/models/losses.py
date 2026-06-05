"""Multi-task loss for PTA diagnosis: two classifications plus masked regression.

The total loss combines cross-entropy on the reservoir and boundary heads with a
**masked heteroscedastic Gaussian negative log-likelihood** on the parameter head. The
mask (from :mod:`deep_pta.data.sampling`) ensures only parameters that are physically
active for the true class contribute to the regression loss. The predicted log-variance
turns the regression into an uncertainty-aware objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from deep_pta.models.resnet1d import ModelOutput


@dataclass
class LossWeights:
    """Relative weights of the three loss terms.

    Attributes
    ----------
    reservoir : float
        Weight on the reservoir cross-entropy.
    boundary : float
        Weight on the boundary cross-entropy.
    params : float
        Weight on the masked regression NLL.
    """

    reservoir: float = 1.0
    boundary: float = 1.0
    params: float = 1.0


def multitask_loss(
    output: ModelOutput,
    y_reservoir: Tensor,
    y_boundary: Tensor,
    targets: Tensor,
    mask: Tensor,
    weights: LossWeights | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the combined multi-task loss.

    Parameters
    ----------
    output : ModelOutput
        Model head outputs.
    y_reservoir, y_boundary : torch.Tensor
        Integer class labels, shape ``(B,)``.
    targets : torch.Tensor
        Regression targets, shape ``(B, 7)``.
    mask : torch.Tensor
        Boolean/float mask of active targets, shape ``(B, 7)``.
    weights : LossWeights, optional
        Term weights; defaults to all ones.

    Returns
    -------
    loss : torch.Tensor
        Scalar total loss.
    parts : dict
        The three term values as floats, for logging.
    """
    weights = weights or LossWeights()
    ce_res = F.cross_entropy(output.logits_reservoir, y_reservoir)
    ce_bnd = F.cross_entropy(output.logits_boundary, y_boundary)

    mask_f = mask.to(targets.dtype)
    # Heteroscedastic Gaussian NLL: 0.5 * (exp(-logvar) * err^2 + logvar), masked.
    err2 = (output.params - targets) ** 2
    nll = 0.5 * (torch.exp(-output.params_logvar) * err2 + output.params_logvar)
    denom = mask_f.sum().clamp_min(1.0)
    reg = (nll * mask_f).sum() / denom

    loss = weights.reservoir * ce_res + weights.boundary * ce_bnd + weights.params * reg
    parts = {
        "ce_reservoir": float(ce_res.detach()),
        "ce_boundary": float(ce_bnd.detach()),
        "reg_nll": float(reg.detach()),
    }
    return loss, parts
