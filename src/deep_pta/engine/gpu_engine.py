"""GPU (PyTorch) batched port of the analytical engine, validated against the CPU engine.

Reproduces :func:`deep_pta.engine.solution.evaluate`'s dimensionless wellbore pressure
``p_wD`` to ~1e-7 relative error (far below the realism noise floor), but evaluates
thousands of curves at once on the GPU instead of one at a time on the CPU. Used only to
accelerate dataset generation; the certified CPU engine remains the reference of record
(equivalence is checked in ``tests/test_gpu_engine.py``).

The closed-boundary solution needs a scaled modified Bessel ``I`` (``ive``), which torch
lacks, so it is implemented here: exact via ``exp(-x) I(x)`` where ``I`` does not overflow,
and an asymptotic series for large argument (where it is accurate to ~1e-13).
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from deep_pta.engine.stehfest import stehfest_weights

# Class indices mirror :mod:`deep_pta.data.sampling`.
RES_HOMO, RES_DP, RES_INF, RES_FIN = 0, 1, 2, 3
BND_INF, BND_SEAL, BND_CONST, BND_CLOSED = 0, 1, 2, 3

_N_STEH = 12
_LN2 = math.log(2.0)
_GLN, _GLW = np.polynomial.legendre.leggauss(24)
_GAMMA_54 = math.gamma(1.25)
_BILINEAR_EXP = -1.25


class _Consts:
    """Cache of device-resident constant tensors (Stehfest weights, GL nodes)."""

    def __init__(self, device: torch.device) -> None:
        self.v = torch.tensor(stehfest_weights(_N_STEH), dtype=torch.float64, device=device)
        self.gl_x = torch.tensor(0.5 * (_GLN + 1.0), dtype=torch.float64, device=device)
        self.gl_w = torch.tensor(0.5 * _GLW, dtype=torch.float64, device=device)


def _ive(nu: int, x: torch.Tensor) -> torch.Tensor:
    """Scaled modified Bessel I, ``exp(-x) I_nu(x)`` for ``nu`` in ``{0, 1}``."""
    small = x < 700.0
    xs = torch.where(small, x, torch.ones_like(x))
    i_fn = torch.special.modified_bessel_i0 if nu == 0 else torch.special.modified_bessel_i1
    small_val = torch.exp(-xs) * i_fn(xs)
    mu = 4.0 * nu * nu
    t8 = 1.0 / (8.0 * x)
    c1 = -(mu - 1.0)
    c2 = (mu - 1.0) * (mu - 9.0) / 2.0
    c3 = -(mu - 1.0) * (mu - 9.0) * (mu - 25.0) / 6.0
    large_val = (1.0 + c1 * t8 + c2 * t8 * t8 + c3 * t8 * t8 * t8) / torch.sqrt(2.0 * math.pi * x)
    return torch.where(small, small_val, large_val)


def _kve(nu: int, x: torch.Tensor) -> torch.Tensor:
    """Scaled modified Bessel K, ``exp(x) K_nu(x)`` for ``nu`` in ``{0, 1}``."""
    if nu == 0:
        return cast(torch.Tensor, torch.special.scaled_modified_bessel_k0(x))
    return cast(torch.Tensor, torch.special.scaled_modified_bessel_k1(x))


def _f_of_u(u: torch.Tensor, rc: int, omega: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Reservoir coupling ``f(u)``: Warren-Root for double porosity, else 1."""
    if rc == RES_DP:
        return (omega * (1.0 - omega) * u + lam) / ((1.0 - omega) * u + lam)
    return torch.ones_like(u)


def _sandface(
    u: torch.Tensor, rc: int, omega: torch.Tensor, lam: torch.Tensor, fcd: torch.Tensor, c: _Consts
) -> torch.Tensor:
    """Bare sandface ``p_rD(u)`` for the reservoir class."""
    if rc in (RES_HOMO, RES_DP):
        f = _f_of_u(u, rc, omega, lam)
        x = torch.sqrt(u * f)
        return _kve(0, x) / (u * x * _kve(1, x))
    sqrt_u = torch.sqrt(u)
    arg = sqrt_u.unsqueeze(-1) * c.gl_x
    integral = (torch.special.modified_bessel_k0(arg) * c.gl_w).sum(dim=-1)
    base = integral / u
    if rc == RES_FIN:
        amp = (math.pi / (2.0 * torch.sqrt(fcd))) * _GAMMA_54
        return cast(torch.Tensor, base + amp * u**_BILINEAR_EXP)
    return cast(torch.Tensor, base)


def _laplace_pwd(
    u: torch.Tensor,
    rc: int,
    bc: int,
    omega: torch.Tensor,
    lam: torch.Tensor,
    fcd: torch.Tensor,
    l_d: torch.Tensor,
    r_ed: torch.Tensor,
    c_d: torch.Tensor,
    s: torch.Tensor,
    c: _Consts,
) -> torch.Tensor:
    """Full wellbore Laplace solution ``p_wD(u)``: sandface -> boundary -> storage/skin."""
    if bc == BND_CLOSED:
        f = _f_of_u(u, rc, omega, lam)
        x = torch.sqrt(u * f)
        a, b = r_ed * x, x
        e = torch.exp(2.0 * (b - a))
        num = _ive(1, a) * _kve(0, b) + _kve(1, a) * _ive(0, b) * e
        den = u * b * (_ive(1, a) * _kve(1, b) - _kve(1, a) * _ive(1, b) * e)
        p_rd = num / den
    else:
        p_rd = _sandface(u, rc, omega, lam, fcd, c)
        if bc in (BND_SEAL, BND_CONST):
            f = _f_of_u(u, rc, omega, lam)
            x = torch.sqrt(u * f)
            sign = 1.0 if bc == BND_SEAL else -1.0
            p_rd = p_rd + sign * torch.special.modified_bessel_k0(2.0 * l_d * x) / u
    inner = u * p_rd + s
    return inner / (u * (1.0 + c_d * u * inner))


def _group_pwd(
    rc: int,
    bc: int,
    t_d: torch.Tensor,
    omega: torch.Tensor,
    lam: torch.Tensor,
    fcd: torch.Tensor,
    l_d: torch.Tensor,
    r_ed: torch.Tensor,
    c_d: torch.Tensor,
    s: torch.Tensor,
    c: _Consts,
) -> torch.Tensor:
    """Stehfest-invert ``p_wD`` to time for a group of curves sharing classes ``(rc, bc)``."""
    acc = torch.zeros_like(t_d)
    for i in range(1, _N_STEH + 1):
        u = i * _LN2 / t_d
        acc = acc + c.v[i - 1] * _laplace_pwd(u, rc, bc, omega, lam, fcd, l_d, r_ed, c_d, s, c)
    return (_LN2 / t_d) * acc


def evaluate_pwd_batch(
    reservoir_class: NDArray[np.int64],
    boundary_class: NDArray[np.int64],
    t_d: NDArray[np.float64],
    c_d: NDArray[np.float64],
    s: NDArray[np.float64],
    omega: NDArray[np.float64],
    lam: NDArray[np.float64],
    f_cd: NDArray[np.float64],
    l_d: NDArray[np.float64],
    r_ed: NDArray[np.float64],
    device: torch.device | None = None,
) -> NDArray[np.float64]:
    """Evaluate ``p_wD`` for a heterogeneous batch of curves on the GPU.

    Curves are grouped by ``(reservoir_class, boundary_class)`` and each group is
    evaluated in one batched call. Inactive parameters (e.g. ``omega`` for a fracture)
    are ignored by the relevant branch, so any finite placeholder is fine.

    Parameters
    ----------
    reservoir_class, boundary_class : numpy.ndarray
        Integer class arrays, shape ``(B,)``.
    t_d : numpy.ndarray
        Dimensionless times, shape ``(B, T)`` (each curve's own grid).
    c_d, s, omega, lam, f_cd, l_d, r_ed : numpy.ndarray
        Per-curve parameters, shape ``(B,)``.
    device : torch.device, optional
        Compute device; defaults to CUDA if available, else CPU.

    Returns
    -------
    numpy.ndarray
        ``p_wD`` of shape ``(B, T)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = _Consts(device)
    t = torch.tensor(t_d, dtype=torch.float64, device=device)

    def col(a: NDArray[np.float64]) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float64, device=device).unsqueeze(1)

    cols = {
        k: col(v)
        for k, v in dict(c_d=c_d, s=s, omega=omega, lam=lam, fcd=f_cd, l_d=l_d, r_ed=r_ed).items()
    }
    out = torch.empty_like(t)
    rc_t = torch.tensor(reservoir_class)
    bc_t = torch.tensor(boundary_class)
    for rc in range(4):
        for bc in range(4):
            idx = torch.nonzero((rc_t == rc) & (bc_t == bc), as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            ix = idx.to(device)
            out[ix] = _group_pwd(
                rc,
                bc,
                t[ix],
                cols["omega"][ix],
                cols["lam"][ix],
                cols["fcd"][ix],
                cols["l_d"][ix],
                cols["r_ed"][ix],
                cols["c_d"][ix],
                cols["s"][ix],
                c,
            )
    return out.cpu().numpy()
