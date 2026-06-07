"""Multi-task 1-D ResNet for PTA diagnosis.

A residual CNN over the 3-channel x 256 log-log representation with four heads:
reservoir class (4), boundary class (4), parameter regression (7), and a per-parameter
log-variance head for heteroscedastic (aleatoric) uncertainty. The CNN baseline follows
the architecture shown to work for PTA in [DiscoverAppSci2024]_ and [JPSE2021]_.

Out-of-taxonomy inputs are flagged at inference by the softmax entropy of the class
heads together with the predicted regression variance.

References
----------
.. [DiscoverAppSci2024] Enhancing pressure transient analysis through deep learning
   neural networks. Discover Applied Sciences (2024).
.. [JPSE2021] Application of deep learning on well-test interpretation. JPSE (2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from deep_pta.data.sampling import N_BOUNDARY, N_PARAMS, N_RESERVOIR


@dataclass
class ModelOutput:
    """Container for the four head outputs of :class:`ResNet1D`.

    Attributes
    ----------
    logits_reservoir : torch.Tensor
        Reservoir-class logits, shape ``(B, 4)``.
    logits_boundary : torch.Tensor
        Boundary-class logits, shape ``(B, 4)``.
    params : torch.Tensor
        Parameter regression means, shape ``(B, 7)``.
    params_logvar : torch.Tensor
        Predicted log-variance per parameter, shape ``(B, 7)``.
    """

    logits_reservoir: Tensor
    logits_boundary: Tensor
    params: Tensor
    params_logvar: Tensor


class _ResidualBlock1D(nn.Module):
    """A 1-D residual block (two convolutions, BN, ReLU, projected skip)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block."""
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return cast(Tensor, self.act(out + self.skip(x)))


class ResNet1D(nn.Module):
    """Multi-task 1-D ResNet over the 3-channel log-log representation.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 3.
    base_channels : int, optional
        Channel count of the first stage, by default 32.
    n_blocks : int, optional
        Number of residual blocks (channels double and length halves every two
        blocks), by default 6.
    dropout : float, optional
        Dropout probability applied to the pooled features before the heads,
        by default 0.0 (no dropout).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        n_blocks: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        ch = base_channels
        for i in range(n_blocks):
            out_ch = ch * 2 if (i > 0 and i % 2 == 0) else ch
            stride = 2 if (i % 2 == 0) else 1
            blocks.append(_ResidualBlock1D(ch, out_ch, stride))
            ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.head_reservoir = nn.Linear(ch, N_RESERVOIR)
        self.head_boundary = nn.Linear(ch, N_BOUNDARY)
        self.head_params = nn.Linear(ch, N_PARAMS)
        self.head_logvar = nn.Linear(ch, N_PARAMS)

    def forward(self, x: Tensor) -> ModelOutput:
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, 3, 256)``.

        Returns
        -------
        ModelOutput
            The four head outputs.
        """
        feat = self.dropout(self.pool(self.blocks(self.stem(x))).flatten(1))
        return ModelOutput(
            logits_reservoir=self.head_reservoir(feat),
            logits_boundary=self.head_boundary(feat),
            params=self.head_params(feat),
            params_logvar=torch.clamp(self.head_logvar(feat), -8.0, 8.0),
        )
