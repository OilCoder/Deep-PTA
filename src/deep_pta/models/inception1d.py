"""InceptionTime-style 1-D network for PTA diagnosis.

Inception modules apply parallel convolutions with very different kernel sizes
(short/medium/long) plus a max-pool branch, so each layer sees the curve at several
log-time scales at once — a natural fit for PTA, where the diagnostic regimes
(storage hump, half-slope, radial plateau, boundary tail) live at different widths
of the 256-point log grid. Residual shortcuts every ``residual_every`` modules
follow the InceptionTime recipe.

References
----------
.. [Ismail2020] Ismail Fawaz, H. et al. (2020). InceptionTime: Finding AlexNet for
   time series classification. Data Mining and Knowledge Discovery 34, 1936-1962.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from deep_pta.data.sampling import N_BOUNDARY, N_PARAMS, N_RESERVOIR
from deep_pta.models.resnet1d import ModelOutput


class _InceptionModule(nn.Module):
    """Bottleneck + three parallel kernel sizes + max-pool branch, concatenated."""

    def __init__(self, in_ch: int, n_filters: int, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.bottleneck = (
            nn.Conv1d(in_ch, n_filters, 1, bias=False) if in_ch > n_filters else nn.Identity()
        )
        mid_ch = n_filters if in_ch > n_filters else in_ch
        self.convs = nn.ModuleList(
            [nn.Conv1d(mid_ch, n_filters, k, padding=k // 2, bias=False) for k in kernel_sizes]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, n_filters, 1, bias=False),
        )
        out_ch = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.out_ch = out_ch

    def forward(self, x: Tensor) -> Tensor:
        """Apply the parallel branches and concatenate along channels."""
        mid = self.bottleneck(x)
        branches = [conv(mid) for conv in self.convs] + [self.pool_branch(x)]
        return cast(Tensor, self.act(self.bn(torch.cat(branches, dim=1))))


class InceptionTime1D(nn.Module):
    """InceptionTime-style multi-scale CNN with the four PTA heads.

    Parameters
    ----------
    in_channels : int, optional
        Input channels, by default 3.
    n_filters : int, optional
        Filters per branch (module output is ``4 * n_filters``), by default 32.
    depth : int, optional
        Number of inception modules, by default 6.
    kernel_sizes : tuple of int, optional
        Branch kernel sizes, by default ``(9, 19, 39)``.
    residual_every : int, optional
        Add a residual shortcut every this many modules, by default 3.
    dropout : float, optional
        Dropout on the pooled feature before the heads, by default 0.0.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_filters: int = 32,
        depth: int = 6,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
        residual_every: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.residual_every = residual_every
        self.modules_list = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        ch = in_channels
        res_ch = in_channels
        for i in range(depth):
            module = _InceptionModule(ch, n_filters, kernel_sizes)
            self.modules_list.append(module)
            ch = module.out_ch
            if (i + 1) % residual_every == 0:
                self.shortcuts.append(
                    nn.Sequential(nn.Conv1d(res_ch, ch, 1, bias=False), nn.BatchNorm1d(ch))
                )
                res_ch = ch
        self.act = nn.ReLU(inplace=True)
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
            Input batch of shape ``(B, C, 256)``.

        Returns
        -------
        ModelOutput
            The four head outputs.
        """
        h = x
        residual = x
        shortcut_idx = 0
        for i, module in enumerate(self.modules_list):
            h = module(h)
            if (i + 1) % self.residual_every == 0:
                h = self.act(h + self.shortcuts[shortcut_idx](residual))
                residual = h
                shortcut_idx += 1
        feat = self.dropout(self.pool(h).flatten(1))
        return ModelOutput(
            logits_reservoir=self.head_reservoir(feat),
            logits_boundary=self.head_boundary(feat),
            params=self.head_params(feat),
            params_logvar=torch.clamp(self.head_logvar(feat), -8.0, 8.0),
        )
