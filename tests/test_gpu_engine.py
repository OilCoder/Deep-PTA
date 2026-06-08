"""Equivalence test: the GPU engine must match the certified CPU engine.

Skipped when CUDA is unavailable. Tolerance is 1e-5 relative — far below the realism
noise floor (~1e-2) and the engine's type-curve certification tolerance, while leaving
headroom for torch-vs-scipy float64 Bessel differences (~1e-7 observed).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

from deep_pta.data import sampling  # noqa: E402
from deep_pta.data.generator import _build_boundary, _build_reservoir  # noqa: E402
from deep_pta.engine.gpu_engine import evaluate_pwd_batch  # noqa: E402
from deep_pta.engine.solution import evaluate as cpu_evaluate  # noqa: E402


def test_gpu_matches_cpu_engine() -> None:
    rng = np.random.default_rng(0)
    cps = [sampling.sample_curve(rng) for _ in range(160)]

    cpu = []
    for cp in cps:
        res, bnd = _build_reservoir(cp), _build_boundary(cp)
        p, _ = cpu_evaluate(res, bnd, (cp.raw["C_D"], cp.raw["S"]), cp.t_d)
        cpu.append(p)
    cpu_arr = np.stack(cpu)

    def col(key: str, default: float) -> np.ndarray:
        return np.array([cp.raw.get(key, default) for cp in cps], dtype=np.float64)

    gpu_arr = evaluate_pwd_batch(
        np.array([cp.reservoir_class for cp in cps], dtype=np.int64),
        np.array([cp.boundary_class for cp in cps], dtype=np.int64),
        np.stack([cp.t_d for cp in cps]),
        col("C_D", 1.0),
        col("S", 0.0),
        col("omega", 0.1),
        col("lambda", 1e-6),
        col("F_CD", 1.0),
        col("L_D", 1.0),
        col("r_eD", 2.0),
    )
    rel = np.abs(gpu_arr - cpu_arr) / (np.abs(cpu_arr) + 1e-12)
    assert float(np.max(rel)) < 1e-5
