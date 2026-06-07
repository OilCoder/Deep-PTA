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
  validity filter; reproducible, with disjoint `C_D` bands for train/val/test.
- **Two models** — a multi-task ResNet-1D and a hand-built PatchTST-style Transformer,
  over a 3-channel log-log representation (Δp, Bourdet derivative, absolute time).
- **Honest evaluation** — balanced accuracy, macro-F1, and per-class recall on a
  class-stratified test set; an entrypoint with HPO, early stopping, and TensorBoard.
- **App + narrator** — Gradio CSV→diagnosis with a fitted-curve overlay; optional
  Claude-based narrator.

## Results (honest, balanced)

Headline accuracy is reported as **balanced accuracy** (mean per-class recall) on a
**class-stratified** test set — raw accuracy is misleading under class imbalance.

| Head | Baseline (raw / balanced) | Current (balanced) |
|---|---|---|
| Reservoir (4 classes) | 0.42 / 0.43 | **0.51** |
| Boundary (4 classes) | 0.84 / 0.61 | **0.70** |

The boundary "0.84" was a mirage: the old test set was ~70% infinite-acting, so the
metric rode the majority class. On a balanced test the model genuinely improves the
classes that matter (sealing fault 0.35→0.59, constant pressure 0.40→0.60, closed
0.71→0.85) and the homogeneous reservoir recovers from 0.12 to 0.33. Full method and
ablation: `documentation/reporte-mejora-accuracy.md` and
`documentation/reporte-no-unicidad-confusiones.md`.

## Run it

```bash
uv sync                                  # or: pip install -e .[dev,ml,app]

pytest -q                                # verification gate (also: mypy src/, ruff check .)

# Train (frozen train set optional; on-the-fly if omitted). HPO + final run:
python -m deep_pta.train --hpo-trials 20 --steps 20000 \
    --train-h5 data/synthetic_train.h5 --class-weights

python -m deep_pta.app.app               # launch the Gradio app
```

## Status

Engineering complete (engine, generator, models, app, narrator). Latest cycle fixed the
metric honesty and class imbalance and added a reproducible HPO/training pipeline. Real
case validation (Lee/Horne/Volve/WebPlotDigitizer) is future work — the inference path
(`src/deep_pta/data/real_cases.py`) is ready. See `todo/PLAN.md`.
