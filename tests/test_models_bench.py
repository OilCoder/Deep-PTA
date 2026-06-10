"""Forward-pass smoke tests for the architecture-benchmark contenders.

Every model must consume the 5-channel 256-point representation and return a
:class:`~deep_pta.models.resnet1d.ModelOutput` with the four PTA heads, so the
shared training loop and evaluator work unchanged.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deep_pta.data.sampling import N_BOUNDARY, N_PARAMS, N_RESERVOIR  # noqa: E402
from deep_pta.models.inception1d import InceptionTime1D  # noqa: E402
from deep_pta.models.moe import MoE1D  # noqa: E402
from deep_pta.models.patchtst import PatchTST1D  # noqa: E402
from deep_pta.models.tcn import TCN1D  # noqa: E402

BATCH, CHANNELS, GRID = 4, 5, 256


def _check_output(model: torch.nn.Module) -> None:
    x = torch.randn(BATCH, CHANNELS, GRID)
    out = model(x)
    assert out.logits_reservoir.shape == (BATCH, N_RESERVOIR)
    assert out.logits_boundary.shape == (BATCH, N_BOUNDARY)
    assert out.params.shape == (BATCH, N_PARAMS)
    assert out.params_logvar.shape == (BATCH, N_PARAMS)
    assert torch.isfinite(out.logits_reservoir).all()


def test_moe_forward_and_gates() -> None:
    """MoE returns the four heads and exposes normalized routing weights."""
    model = MoE1D(in_channels=CHANNELS)
    _check_output(model)
    assert model.last_gates is not None
    assert model.last_gates.shape == (BATCH, 4)
    torch.testing.assert_close(model.last_gates.sum(dim=1), torch.ones(BATCH), rtol=0, atol=1e-5)


def test_inception_forward() -> None:
    """InceptionTime consumes 5 channels and returns the four heads."""
    _check_output(InceptionTime1D(in_channels=CHANNELS))


def test_tcn_forward() -> None:
    """TCN consumes 5 channels and returns the four heads."""
    _check_output(TCN1D(in_channels=CHANNELS))


def test_patchtst_big_forward() -> None:
    """The large PatchTST variant builds and runs at benchmark capacity."""
    model = PatchTST1D(in_channels=CHANNELS, d_model=256, n_heads=8, depth=8)
    _check_output(model)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 4_000_000  # genuinely a capacity test, not a tweak
