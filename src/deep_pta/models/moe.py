"""Mixture-of-Experts 1-D network for PTA diagnosis.

A dense (soft-gated) mixture in the original Jacobs et al. sense: a small gating
network reads the curve and weights the feature vectors of ``n_experts`` independent
convolutional towers; the blended feature feeds the four shared PTA heads. Dense
gating avoids the expert-collapse pathology of sparse top-k routing without auxiliary
load-balancing losses — sparse routing is a compute optimization for giant models,
unnecessary at this scale.

The physical motivation: flow regimes make PTA curves nearly disjoint sub-problems
(storage-dominated vs early-fracture vs late-boundary), so conditional specialization
may beat a monolithic encoder. ``last_gates`` exposes the routing weights so the
benchmark can test whether the gate learns the physical taxonomy on its own.

References
----------
.. [Jacobs1991] Jacobs, R.A., Jordan, M.I., Nowlan, S.J. & Hinton, G.E. (1991).
   Adaptive Mixtures of Local Experts. Neural Computation 3(1), 79-87.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from deep_pta.data.sampling import N_BOUNDARY, N_PARAMS, N_RESERVOIR
from deep_pta.models.resnet1d import ModelOutput, _ResidualBlock1D


class _ExpertTower(nn.Module):
    """A small convolutional expert: stem + two residual stages + global pool."""

    def __init__(self, in_channels: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, 7, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            _ResidualBlock1D(width, width, stride=2),
            _ResidualBlock1D(width, 2 * width, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = 2 * width

    def forward(self, x: Tensor) -> Tensor:
        """Return the pooled feature vector, shape ``(B, 2 * width)``."""
        return cast(Tensor, self.net(x).flatten(1))


class MoE1D(nn.Module):
    """Soft-gated mixture of convolutional experts with the four PTA heads.

    Parameters
    ----------
    in_channels : int, optional
        Input channels, by default 3.
    n_experts : int, optional
        Number of expert towers, by default 4.
    expert_width : int, optional
        Stem width of each expert (feature dim is ``2 * expert_width``),
        by default 32.
    gate_width : int, optional
        Width of the gating encoder, by default 16.
    dropout : float, optional
        Dropout on the blended feature before the heads, by default 0.0.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_experts: int = 4,
        expert_width: int = 32,
        gate_width: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [_ExpertTower(in_channels, expert_width) for _ in range(n_experts)]
        )
        feat_dim = 2 * expert_width
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, gate_width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(gate_width),
            nn.ReLU(inplace=True),
            nn.Conv1d(gate_width, gate_width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(gate_width),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(gate_width, n_experts),
        )
        self.dropout = nn.Dropout(dropout)
        self.head_reservoir = nn.Linear(feat_dim, N_RESERVOIR)
        self.head_boundary = nn.Linear(feat_dim, N_BOUNDARY)
        self.head_params = nn.Linear(feat_dim, N_PARAMS)
        self.head_logvar = nn.Linear(feat_dim, N_PARAMS)

        #: Routing weights of the most recent forward pass, shape ``(B, n_experts)``.
        self.last_gates: Tensor | None = None

    def forward(self, x: Tensor) -> ModelOutput:
        """Run the gated mixture forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, C, 256)``.

        Returns
        -------
        ModelOutput
            The four head outputs.
        """
        gates = torch.softmax(self.gate(x), dim=1)
        self.last_gates = gates.detach()
        feats = torch.stack([expert(x) for expert in self.experts], dim=1)
        blended = self.dropout((gates.unsqueeze(-1) * feats).sum(dim=1))
        return ModelOutput(
            logits_reservoir=self.head_reservoir(blended),
            logits_boundary=self.head_boundary(blended),
            params=self.head_params(blended),
            params_logvar=torch.clamp(self.head_logvar(blended), -8.0, 8.0),
        )
