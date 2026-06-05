"""CPU smoke test: the model can overfit a single batch (forward/backward + masks).

Skipped cleanly when the optional ``[ml]`` extra (``torch``) is absent, so the suite
stays green on the bare CPU build.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from deep_pta.train.train_cnn import overfit_one_batch  # noqa: E402


def test_overfit_one_batch_drops_loss() -> None:
    # A correctly wired model overfits a single small batch: the loss must drop sharply.
    final_loss = overfit_one_batch(steps=120, batch_size=12, seed=0)
    assert final_loss < 1.0
