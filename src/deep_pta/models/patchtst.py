"""Hand-built PatchTST-style 1-D Transformer for PTA diagnosis.

The self-attention is implemented from scratch (no ``nn.Transformer`` / ``nn.MultiheadAttention``)
because building the attention by hand is a declared learning goal of the project. The
sequence of 256 points is split into patches [Nie2023]_, linearly embedded, given a
learned positional encoding, and processed by Transformer blocks; a mean-pooled
representation feeds the same four heads as the CNN baseline.

A flow regime is defined by its relation to what comes before and after it (a long-range
dependency), which motivates global self-attention over the derivative.

References
----------
.. [Nie2023] Nie, Y. et al. (2023). A Time Series is Worth 64 Words: Long-term
   Forecasting with Transformers (PatchTST). ICLR 2023.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from deep_pta.data.sampling import N_BOUNDARY, N_PARAMS, N_RESERVOIR
from deep_pta.models.resnet1d import ModelOutput


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention, implemented from scratch.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads (must divide ``d_model``).
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.last_attn: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention over the patch sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, d_model)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, d_model)``.
        """
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, H, T, d_head)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn.detach()
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        return cast(Tensor, self.proj(out))


class _TransformerBlock(nn.Module):
    """A pre-norm Transformer block (attention + MLP with residuals)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the pre-norm attention and MLP residual sublayers."""
        x = x + self.attn(self.norm1(x))
        return cast(Tensor, x + self.mlp(self.norm2(x)))


class PatchTST1D(nn.Module):
    """Patch-based 1-D Transformer with the four PTA heads.

    Parameters
    ----------
    in_channels : int, optional
        Input channels, by default 2.
    seq_len : int, optional
        Input length, by default 256.
    patch_len : int, optional
        Patch size (must divide ``seq_len``), by default 16.
    d_model : int, optional
        Embedding dimension, by default 64.
    n_heads : int, optional
        Attention heads, by default 4.
    depth : int, optional
        Number of Transformer blocks, by default 3.
    """

    def __init__(
        self,
        in_channels: int = 2,
        seq_len: int = 256,
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len")
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.embed = nn.Linear(in_channels * patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head_reservoir = nn.Linear(d_model, N_RESERVOIR)
        self.head_boundary = nn.Linear(d_model, N_BOUNDARY)
        self.head_params = nn.Linear(d_model, N_PARAMS)
        self.head_logvar = nn.Linear(d_model, N_PARAMS)

    def forward(self, x: Tensor) -> ModelOutput:
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, 2, 256)``.

        Returns
        -------
        ModelOutput
            The four head outputs.
        """
        b, c, _ = x.shape
        # Patchify: (B, C, n_patches, patch_len) -> (B, n_patches, C*patch_len).
        patches = x.reshape(b, c, self.n_patches, self.patch_len)
        patches = patches.permute(0, 2, 1, 3).reshape(b, self.n_patches, -1)
        h = self.embed(patches) + self.pos
        for block in self.blocks:
            h = block(h)
        feat = self.norm(h).mean(dim=1)
        return ModelOutput(
            logits_reservoir=self.head_reservoir(feat),
            logits_boundary=self.head_boundary(feat),
            params=self.head_params(feat),
            params_logvar=torch.clamp(self.head_logvar(feat), -8.0, 8.0),
        )

    def attention_maps(self, x: Tensor) -> list[Tensor]:
        """Return the per-block attention tensors for a forward pass on ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape ``(B, 2, 256)``.

        Returns
        -------
        list of torch.Tensor
            One ``(B, n_heads, n_patches, n_patches)`` attention map per block.
        """
        self.forward(x)
        maps = []
        for block in self.blocks:
            assert isinstance(block, _TransformerBlock)
            if block.attn.last_attn is not None:
                maps.append(block.attn.last_attn)
        return maps
