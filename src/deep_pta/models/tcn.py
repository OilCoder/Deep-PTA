"""Temporal Convolutional Network (1-D, dilated) for PTA diagnosis.

A stack of residual blocks with exponentially growing dilation, so the receptive
field covers the full 256-point log grid in a few layers while keeping per-layer
locality. Non-causal (centered) padding — the task is whole-curve diagnosis, not
forecasting, so future context is legitimate.

References
----------
.. [Bai2018] Bai, S., Kolter, J.Z. & Koltun, V. (2018). An Empirical Evaluation of
   Generic Convolutional and Recurrent Networks for Sequence Modeling.
   arXiv:1803.01271.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from deep_pta.data.sampling import N_BOUNDARY, N_PARAMS, N_RESERVOIR
from deep_pta.models.resnet1d import ModelOutput


class _TemporalBlock(nn.Module):
    """Two dilated convolutions with BN/ReLU and a projected residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, padding=pad, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, padding=pad, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.skip: nn.Module = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the dilated residual block."""
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return cast(Tensor, self.act(out + self.skip(x)))


class TCN1D(nn.Module):
    """Dilated temporal CNN with the four PTA heads.

    Parameters
    ----------
    in_channels : int, optional
        Input channels, by default 3.
    width : int, optional
        Channel count of every temporal block, by default 64.
    n_levels : int, optional
        Number of blocks; dilation doubles per level (receptive field
        ``~ kernel_size * 2^n_levels``), by default 7.
    kernel_size : int, optional
        Convolution kernel size (odd), by default 5.
    dropout : float, optional
        Dropout on the pooled feature before the heads, by default 0.0.
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        n_levels: int = 7,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks = []
        ch = in_channels
        for level in range(n_levels):
            blocks.append(_TemporalBlock(ch, width, kernel_size, dilation=2**level))
            ch = width
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.head_reservoir = nn.Linear(width, N_RESERVOIR)
        self.head_boundary = nn.Linear(width, N_BOUNDARY)
        self.head_params = nn.Linear(width, N_PARAMS)
        self.head_logvar = nn.Linear(width, N_PARAMS)

    def forward(self, x: Tensor) -> ModelOutput:
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, C, 256)``.

        Returns
        -------
        ModelOutput
            The four head outputs.
        """
        feat = self.dropout(self.pool(self.blocks(x)).flatten(1))
        return ModelOutput(
            logits_reservoir=self.head_reservoir(feat),
            logits_boundary=self.head_boundary(feat),
            params=self.head_params(feat),
            params_logvar=torch.clamp(self.head_logvar(feat), -8.0, 8.0),
        )
