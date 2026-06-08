"""Logit/precision-fusing ensemble over models sharing the :class:`ModelOutput` API.

Combines several trained PTA models (e.g. a ResNet1D and a PatchTST) into one module
that exposes the same :class:`~deep_pta.models.resnet1d.ModelOutput`, so ``evaluate`` and
the diagnostic scripts work unchanged. Class heads are fused by **averaging softmax
probabilities** (robust to differently-scaled logits); the parameter head is fused by
**inverse-variance (precision) weighting** using each model's heteroscedastic log-variance.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from deep_pta.models.resnet1d import ModelOutput


class Ensemble(nn.Module):
    """Average a set of multi-task PTA models into a single :class:`ModelOutput`.

    Parameters
    ----------
    models : list of torch.nn.Module
        Trained models, each returning a :class:`ModelOutput` for input ``(B, 3, 256)``.
    """

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        if not models:
            raise ValueError("Ensemble needs at least one model")
        self.models = nn.ModuleList(models)

    def forward(self, x: Tensor) -> ModelOutput:
        """Fuse the member outputs for batch ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, 3, 256)``.

        Returns
        -------
        ModelOutput
            Fused heads: class logits as ``log`` of the mean softmax (argmax-equivalent),
            parameters as the precision-weighted mean, and the combined log-variance.
        """
        outs = [m(x) for m in self.models]

        # Class heads: average softmax probabilities, return as log-probs (argmax-stable).
        prob_res = torch.stack([torch.softmax(o.logits_reservoir, dim=-1) for o in outs]).mean(0)
        prob_bnd = torch.stack([torch.softmax(o.logits_boundary, dim=-1) for o in outs]).mean(0)
        logits_res = torch.log(prob_res.clamp_min(1e-12))
        logits_bnd = torch.log(prob_bnd.clamp_min(1e-12))

        # Param head: inverse-variance (precision) weighted mean of the per-model means.
        params = torch.stack([o.params for o in outs])
        precision = torch.stack([torch.exp(-o.params_logvar) for o in outs])
        total_precision = precision.sum(0)
        fused_params = (precision * params).sum(0) / total_precision.clamp_min(1e-12)
        fused_logvar = -torch.log(total_precision.clamp_min(1e-12))

        return ModelOutput(
            logits_reservoir=logits_res,
            logits_boundary=logits_bnd,
            params=fused_params,
            params_logvar=fused_logvar,
        )
