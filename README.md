# Deep PTA

Neural network that interprets pressure transient tests (PTA): it reads the Δp +
Bourdet derivative curve (log-log) and diagnoses the reservoir model, the boundary,
and the key parameters (k·h, S, C, ω, λ, L, x_f) — the senior interpreter's job,
automated. Trained on a 100% synthetic dataset produced by a **certified** analytical
engine (Laplace solutions + Stehfest inversion). Ships with an interactive app
(CSV → diagnosis) and an optional LLM narrator agent.

Design source of truth: `documentation/deep-pta.md`,
`documentation/deep-pta-interpretacion-pruebas-presion.md`,
`documentation/plan-implementacion.md`, and `documentation/referencias.md`.

## What works

- **Certified analytical engine** — homogeneous, Warren-Root double porosity, and
  infinite/finite-conductivity fractures, with infinite / sealing-fault /
  constant-pressure / closed boundaries, via Laplace-space solutions + Stehfest
  inversion. Verified against published Bourdet/Gringarten type curves and AnaFlow.
- **Synthetic generator** — realistic noise, drift, truncation, and a domain-aware
  validity filter; reproducible, with a **hybrid `C_D` split** (i.i.d. in-distribution
  holdout + a held-out high-storage extrapolation set).
- **GPU-batched engine** — a PyTorch port of the analytical engine validated against the
  certified CPU engine to ~4e-7 (`src/deep_pta/engine/gpu_engine.py`); generates millions
  of curves in minutes (2M / 5M frozen training sets), preserving the data distribution.
- **Two models + ensemble** — a multi-task ResNet-1D and a hand-built PatchTST-style
  Transformer over a 3-channel log-log representation (Δp, Bourdet derivative, absolute
  time), plus a softmax/precision-fusing ensemble.
- **Honest evaluation** — balanced accuracy, macro-F1, and per-class recall on a
  class-stratified test set, reported separately for in-distribution and extrapolation;
  an entrypoint with HPO, early stopping, and TensorBoard.
- **App + narrator** — Gradio CSV→diagnosis with a fitted-curve overlay; optional
  Claude-based narrator.

## Results (honest, balanced)

Headline accuracy is **balanced accuracy** (mean per-class recall) on a **class-stratified**
test set — raw accuracy is misleading under class imbalance. Two test sets are reported: an
**in-distribution** holdout (headline) and a held-out **high-`C_D` extrapolation** stress
test the model never trains on.

The headline is a **ResNet-1D + PatchTST ensemble** (best configuration).

| Head | Original (misleading) | Honest baseline | **In-distribution (now)** | Extrapolation |
|---|---|---|---|---|
| Reservoir (4 classes) | 0.42 raw | 0.513 | **0.649** | 0.553 |
| Boundary (4 classes) | 0.84 raw | 0.698 | **0.765** | 0.733 |
| Param MAE (log10) | — | 0.69 | **0.36** | 0.73 |

The boundary "0.84" was a mirage (old test ~70% infinite-acting). The real gains came from
**data scale**: a GPU-batched engine made it cheap to train on millions of curves, lifting
reservoir balanced accuracy 0.51 → 0.65 and nearly halving the parameter MAE (0.69 → 0.36).
Per-class reservoir recall: homogeneous 0.12 → 0.59, double-porosity 0.51 → 0.73. The
residual ceiling is the physical homogeneous↔infinite-fracture non-uniqueness. Capacity is
**not** the bottleneck (a 2× larger model matched the small one); data was. The best single
model (ResNet-1D on 5M) scores 0.638 reservoir / 0.750 boundary on its own. Full method and
ablation: `documentation/reporte-mejora-accuracy.md`.

## Run it

```bash
uv sync                                  # or: pip install -e .[dev,ml,app]

pytest -q                                # verification gate (also: mypy src/, ruff check .)

# Generate a large frozen training set on the GPU (engine on GPU, post on CPU):
python debug/dbg_make_train_gpu.py --total 5000000 --out data/synthetic_train_5m.h5

# Train the final model from the frozen set (num_workers 0 for very large sets):
python -m deep_pta.train --steps 80000 --train-h5 data/synthetic_train_5m.h5

python -m deep_pta.app.app               # launch the Gradio app
```

## Status

Engineering complete (engine, generator, models, app, narrator). The 2026-06 cycles fixed
metric honesty + class imbalance, added a hybrid `C_D` split (in-distribution + extrapolation),
a GPU-batched engine validated against the certified CPU engine, and confirmed empirically
that **data scale** — not hyperparameters or capacity — was the bottleneck for the reservoir
head (0.51 → 0.64). Next: break the homogeneous↔infinite-fracture non-uniqueness and validate
on real cases (Lee/Horne/Volve/WebPlotDigitizer; the inference path
`src/deep_pta/data/real_cases.py` is ready). See `todo/PLAN.md`.
